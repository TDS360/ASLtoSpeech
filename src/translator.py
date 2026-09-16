"""
ASL Sign Language Translator - Step 4: Live Translator (auto-speech)
-----------------------------------------------------------------
Loads your trained model (models/letter_classifier.pkl) and turns
your hand poses into SPOKEN words automatically -- no keypress
needed to hear it. Built for a Raspberry Pi deployment: the core
behavior never depends on a keyboard, screen, or mouse.

HOW IT WORKS:
- Hold a letter steady for about LETTER_HOLD_SECONDS and it gets
  added to the word you're currently building. Predictions are
  smoothed over the last SMOOTHING_FRAMES frames (majority vote)
  before the hold timer even starts, so one glitchy frame can't
  flip the letter being read.
- Once a letter is added, change your hand pose (or drop your hand)
  before signing that SAME letter again, so double letters (like
  the two L's in "HELLO") don't blur into one hold.
- When you drop your hand out of frame and keep it out for about
  WORD_PAUSE_SECONDS, whatever word you were building is spell-
  checked, spoken out loud, and added to the sentence.

Both delays are measured with a real clock (time.time()), not a
frame count, so they take the same number of seconds whether this
runs on a fast desktop or a slower Raspberry Pi.

ALL OF THE ABOVE IS TUNABLE without touching this file -- see
config.json in the project root (copy the one from this project if
you don't have it yet; sensible defaults are used if it's missing).

CONTROLS (all optional -- the core behavior above is automatic):
  SPACE      - manually force the current word to finish right now
  BACKSPACE  - delete the last letter of the word you're building
  c          - clear everything
  s          - manually speak everything typed so far
  q          - quit
  (Keyboard controls only work when "headless" is false in
  config.json, since they rely on the OpenCV video window to
  capture key presses. In headless mode, wire a physical button to
  a GPIO pin and set "gpio_button_pin" in config.json to clear the
  current sentence instead.)

Run this AFTER collect_data.py/process_dataset.py and train_model.py
-- it needs models/letter_classifier.pkl to already exist.
"""

import cv2
import pickle
import os
import sys
import json
import subprocess
import platform
import threading
import time
import logging
from collections import deque, Counter
from datetime import datetime

import pyttsx3
from normalize import normalize_landmarks
from mp_setup import create_landmarker, detect, draw_landmarks

MODEL_PATH = "../models/letter_classifier.pkl"
CONFIG_PATH = "../config.json"

DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "use_picamera2": False,        # true = use the Pi Camera Module (ribbon) via picamera2
    "headless": False,             # true = no video window (for a Pi with no monitor)
    "confidence_threshold": 0.6,   # ignore predictions the model isn't confident about
    "letter_hold_seconds": 0.2,    # how long a smoothed pose must hold before it's added
    "word_pause_seconds": 0.45,    # how long your hand must be gone to auto-finish a word
    "smoothing_frames": 5,         # majority-vote window to fight single-frame flicker
    "spell_correct": True,         # nearest-dictionary-word correction before speaking
    "beep_feedback": True,         # short tone on letter-lock and word-finish
    "gpio_button_pin": None,       # e.g. 2 -- BCM pin number for a "clear" button, or null
    "log_level": "INFO",           # DEBUG shows per-frame detail; INFO is normal running
    "session_log_path": "../docs/session_log.md",
    "spell_extra_words": ["asl"],  # words the spell-checker should never "correct" away
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config.update(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] Couldn't read {CONFIG_PATH} ({e}) -- using built-in defaults.")
    else:
        print(f"[config] No {CONFIG_PATH} found -- using built-in defaults. "
              "Copy config.json into the project root to customize thresholds, "
              "camera settings, headless mode, etc.")
    return config


config = load_config()

logging.basicConfig(
    level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("translator")

CONFIDENCE_THRESHOLD = config["confidence_threshold"]
LETTER_HOLD_SECONDS = config["letter_hold_seconds"]
WORD_PAUSE_SECONDS = config["word_pause_seconds"]
HEADLESS = bool(config["headless"])
SMOOTHING_FRAMES = max(1, int(config["smoothing_frames"]))
SPELL_CORRECT = bool(config["spell_correct"])
BEEP_FEEDBACK = bool(config["beep_feedback"])
SESSION_LOG_PATH = config["session_log_path"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "No trained model found at models/letter_classifier.pkl. "
        "Run collect_data.py or process_dataset.py, then train_model.py, "
        "before running this."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Optional spell-correction on finished words -----------------------
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    # Without this, pyspellchecker "corrects" ASL itself to "all" -- add any
    # project-specific words you fingerspell a lot to spell_extra_words in
    # config.json so they're never mangled.
    _spell.word_frequency.load_words(config.get("spell_extra_words", []))
except ImportError:
    _spell = None
    if SPELL_CORRECT:
        log.warning("pyspellchecker not installed -- spell correction disabled. "
                     "Run: pip install pyspellchecker")


def maybe_correct(word):
    """Nudges a finished word toward the nearest dictionary word, but only
    if it doesn't already look like a real word -- this avoids mangling
    names or intentional non-words from a fingerspelling demo."""
    if not SPELL_CORRECT or _spell is None or not word:
        return word
    lower = word.lower()
    if lower in _spell:
        return word
    correction = _spell.correction(lower)
    if correction and correction != lower:
        log.info(f"[spell] '{word}' -> '{correction}'")
        return correction
    return word


# --- Speech (non-blocking, platform-aware) ------------------------------
def speak(text):
    """
    Speaks text out loud without blocking the camera loop.

    On Windows, pyttsx3's SAPI5 driver can conflict with OpenCV's video
    window when both run in the same process, so speak_worker.py is
    launched as a separate process there (the original, Windows-safe
    approach). On Linux -- including a Raspberry Pi -- that conflict
    doesn't apply, and spawning a whole new Python interpreter per word
    is slow on a Pi's CPU, so speech instead runs in a background thread
    in this same process.
    """
    text = text.strip()
    if not text:
        return

    if platform.system() == "Windows":
        log.info(f"[speak] launching speech process for: '{text}'")
        subprocess.Popen([sys.executable, "speak_worker.py", text])
    else:
        log.info(f"[speak] speaking in background thread: '{text}'")
        threading.Thread(target=_speak_in_thread, args=(text,), daemon=True).start()


def _speak_in_thread(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        log.error(f"[speak] error while speaking: {e}")


# --- Short audio tones for non-visual feedback --------------------------
def play_beep(frequency=880, duration_ms=90):
    """
    Fire-and-forget tone, useful once the video window is off in headless
    mode so the signer still gets confirmation a letter or word landed.
    No extra dependency: Windows uses the built-in winsound module;
    Linux (including Raspberry Pi OS) generates a short WAV in memory and
    plays it with `aplay`, which ships with the OS by default.
    """
    if not BEEP_FEEDBACK:
        return

    def _beep():
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(int(frequency), int(duration_ms))
            else:
                import wave
                import struct
                import math
                import tempfile

                framerate = 44100
                n_frames = int(framerate * duration_ms / 1000)
                tmp_path = tempfile.mktemp(suffix=".wav")
                with wave.open(tmp_path, "w") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(framerate)
                    for i in range(n_frames):
                        value = int(32767 * math.sin(2 * math.pi * frequency * i / framerate))
                        wav_file.writeframes(struct.pack("<h", value))
                subprocess.run(["aplay", "-q", tmp_path], check=False,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(tmp_path)
        except Exception as e:
            log.debug(f"[beep] skipped (non-fatal): {e}")

    threading.Thread(target=_beep, daemon=True).start()


# --- Session transcript logging (handy for a demo write-up) -------------
def log_session_event(word):
    try:
        os.makedirs(os.path.dirname(SESSION_LOG_PATH), exist_ok=True)
        with open(SESSION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- {word}\n")
    except OSError as e:
        log.debug(f"[session log] couldn't write ({e})")


# --- Camera abstraction: USB webcam (OpenCV) or Pi Camera (picamera2) ---
class Camera:
    """
    Wraps either cv2.VideoCapture (USB webcam, works everywhere) or
    picamera2 (the official Raspberry Pi camera ribbon, which
    cv2.VideoCapture often can't open on current Raspberry Pi OS
    releases). Always hands back BGR frames either way, so the rest of
    this script doesn't need to know which backend is in use.
    """

    def __init__(self, cfg):
        self.picam = None
        self.cap = None

        if cfg["use_picamera2"]:
            try:
                from picamera2 import Picamera2
                self.picam = Picamera2()
                video_config = self.picam.create_video_configuration(
                    main={"size": (cfg["camera_width"], cfg["camera_height"]),
                          "format": "BGR888"}
                )
                self.picam.configure(video_config)
                self.picam.start()
                log.info("Using Raspberry Pi Camera Module via picamera2.")
            except Exception as e:
                log.warning(f"picamera2 unavailable ({e}); falling back to "
                            "cv2.VideoCapture (USB webcam).")
                self.picam = None

        if self.picam is None:
            self.cap = cv2.VideoCapture(cfg["camera_index"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open the camera (device {cfg['camera_index']}). "
                    "On a Raspberry Pi, check that a camera is connected/enabled "
                    "and that no other program is using it. If you're using the "
                    "ribbon-cable Pi Camera Module, set \"use_picamera2\": true "
                    "in config.json instead."
                )

    def read(self):
        if self.picam is not None:
            frame = self.picam.capture_array()
            return True, frame
        return self.cap.read()

    def release(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


# --- Optional GPIO button (headless control without a keyboard) --------
gpio_clear_event = threading.Event()
if config.get("gpio_button_pin") is not None:
    try:
        from gpiozero import Button
        gpio_button = Button(config["gpio_button_pin"], pull_up=True)
        gpio_button.when_pressed = gpio_clear_event.set
        log.info(f"GPIO clear button enabled on BCM pin {config['gpio_button_pin']}.")
    except Exception as e:
        log.warning(f"Could not set up GPIO button on pin {config['gpio_button_pin']} "
                    f"({e}); continuing without it.")

# ASL fingerspelling only ever uses one hand, so tracking a single hand
# (instead of two) cuts MediaPipe's per-frame work roughly in half.
landmarker = create_landmarker(num_hands=1)
camera = Camera(config)

sentence = ""                      # finished words, each followed by a space
current_word = ""                  # letters built up since the last finished word
last_confirmed_letter = None       # last letter added -- blocks accidental repeats
stable_letter = None                # the (smoothed) letter currently being held
stable_since = None                  # timestamp when the current hold started
letter_confirmed_this_hold = False   # only add one letter per hold, not one per frame
hold_duration = 0.0                   # seconds the current pose has been held (display)
no_hand_since = None                   # timestamp when the hand first disappeared
no_hand_duration = 0.0                 # seconds the hand has been gone (display)
pause_handled = False                   # true once the current pause already triggered speech
previous_hand_present = None            # tracks changes so we only log on transitions
recent_predictions = deque(maxlen=SMOOTHING_FRAMES)  # majority-vote smoothing window


def finish_word():
    """Shared by the automatic pause-trigger and the manual SPACE key so
    the spell-check / speak / beep / log steps only live in one place."""
    global sentence, current_word
    if not current_word:
        return
    corrected = maybe_correct(current_word)
    sentence += corrected + " "
    speak(corrected)
    play_beep(660, 100)  # lower tone: word finished
    log_session_event(corrected)
    current_word = ""


log.info("Translator started.")
log.info(f"Sign letters to build a word. Drop your hand for ~{WORD_PAUSE_SECONDS:.2g}s "
         "to speak it automatically.")
if not HEADLESS:
    log.info("Controls: SPACE=finish word now, BACKSPACE=delete, c=clear, s=speak all, q=quit")
else:
    log.info("Running headless (no video window). Ctrl+C to quit."
             + (" GPIO button clears the sentence." if config.get("gpio_button_pin") else ""))

try:
    while True:
        success, frame = camera.read()
        if not success:
            log.error("Couldn't read from the camera -- stopping.")
            break

        now = time.time()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detect(landmarker, rgb_frame)

        hand_present = bool(result.hand_landmarks)
        predicted_letter = None
        confidence = 0.0

        if hand_present != previous_hand_present:
            log.debug(f"hand_present changed to: {hand_present} (current_word='{current_word}')")
            previous_hand_present = hand_present

        if hand_present:
            if not HEADLESS:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            # ASL fingerspelling is one-handed -- always classify the
            # first detected hand, even if a second hand is in frame.
            features = normalize_landmarks(result.hand_landmarks[0])
            probabilities = model.predict_proba([features])[0]
            best_index = probabilities.argmax()
            predicted_letter = model.classes_[best_index]
            confidence = probabilities[best_index]

            recent_predictions.append(predicted_letter if confidence >= CONFIDENCE_THRESHOLD else None)
            no_hand_since = None
            no_hand_duration = 0.0
            pause_handled = False
        else:
            recent_predictions.append(None)
            if no_hand_since is None:
                no_hand_since = now
            no_hand_duration = now - no_hand_since

        # --- Smooth the last few frames' predictions (majority vote) ---
        non_none = [p for p in recent_predictions if p is not None]
        smoothed_letter = Counter(non_none).most_common(1)[0][0] if non_none else None

        # --- Stability / debounce logic for adding letters (on the SMOOTHED letter) ---
        if smoothed_letter:
            if smoothed_letter == stable_letter:
                hold_duration = now - stable_since
            else:
                stable_letter = smoothed_letter
                stable_since = now
                hold_duration = 0.0
                letter_confirmed_this_hold = False

            if hold_duration >= LETTER_HOLD_SECONDS and not letter_confirmed_this_hold:
                if smoothed_letter != last_confirmed_letter:
                    current_word += smoothed_letter
                    last_confirmed_letter = smoothed_letter
                    play_beep(1200, 50)  # higher tone: letter locked in
                    log.debug(f"letter confirmed: {smoothed_letter} (word so far: '{current_word}')")
                letter_confirmed_this_hold = True
        else:
            stable_letter = None
            stable_since = None
            hold_duration = 0.0
            letter_confirmed_this_hold = False
            last_confirmed_letter = None

        # --- GPIO clear button (checked every frame, works even headless) ---
        if gpio_clear_event.is_set():
            sentence = ""
            current_word = ""
            gpio_clear_event.clear()
            log.info("[gpio] cleared via button")

        # --- Automatic word-finish-and-speak when the hand pauses ---
        if (not hand_present and current_word and not pause_handled
                and no_hand_duration >= WORD_PAUSE_SECONDS):
            finish_word()
            pause_handled = True

        # --- On-screen display (skipped entirely in headless mode) ---
        if not HEADLESS:
            if hand_present and predicted_letter:
                progress = min(hold_duration / LETTER_HOLD_SECONDS, 1.0)
                cv2.putText(frame, f"{predicted_letter.upper()} ({confidence:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                bar_width = int(200 * progress)
                cv2.rectangle(frame, (10, 40), (10 + bar_width, 55), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 40), (210, 55), (255, 255, 255), 1)
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if current_word:
                    pause_progress = min(no_hand_duration / WORD_PAUSE_SECONDS, 1.0)
                    cv2.putText(frame, f"Speaking '{current_word}' soon...",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    bar_width = int(200 * pause_progress)
                    cv2.rectangle(frame, (10, 70), (10 + bar_width, 85), (0, 255, 255), -1)
                    cv2.rectangle(frame, (10, 70), (210, 85), (255, 255, 255), 1)

            cv2.putText(frame, f"Text: {sentence}{current_word}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("ASL Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # SPACE -- manually force the current word to finish now
                finish_word()
            elif key == 8:   # BACKSPACE (key code can vary slightly by OS/keyboard)
                current_word = current_word[:-1]
            elif key == ord('c'):
                sentence = ""
                current_word = ""
            elif key == ord('s'):
                speak(sentence + current_word)
except KeyboardInterrupt:
    log.info("Stopping (Ctrl+C).")
finally:
    # Runs even on an error, not just 'q' or Ctrl+C, so the camera and
    # MediaPipe resources always get released cleanly -- important on a
    # Pi if this restarts automatically under a systemd service.
    camera.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    landmarker.close()