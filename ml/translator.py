"""
ASL Fingerspelling Translator - live camera -> letters -> words -> sentence -> speech
-------------------------------------------------------------------------------------
Pipeline (see README for the diagram):

  webcam -> MediaPipe hand landmarks -> normalize (normalize.py)
         -> letter classifier (models/letter_classifier.pkl)
         -> SignStabilizer (pipeline.py: confidence gate, majority vote,
            time-based hold, repeat lock)
         -> word (spell-checked when you drop your hand)
         -> sentence (polish_sentence: grammar glue only, never new signs)
         -> text-to-speech

WHAT IT CAN RECOGNIZE: the 24 static ASL fingerspelling letters the model
was trained on (A-Y without J and Z, which need motion). It does NOT
recognize whole-word signs, facial expressions or movement. Words and
sentences are built by fingerspelling them letter by letter.

HOW TO USE
  1. Hold a letter steady (~0.2 s) -> it is added (high beep).
  2. Drop your hand for ~0.45 s -> the word is finished and spoken.
  3. Keep your hand down for ~2.5 s (or press ENTER) -> the whole sentence
     is cleaned up, spoken, and saved to docs/translation_history.csv.
  To sign a double letter (the LL in HELLO) briefly relax the hand or
  drop it between the two letters.

KEYS (window mode only; headless mode is fully automatic)
  SPACE finish word    ENTER finish sentence    BACKSPACE delete
  p pause/resume       r replay last sentence   s speak current text
  c clear              + / - confidence threshold   h show/hide help
  q quit

All timings/thresholds live in ml/config.json.
"""

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import pickle

from mp_setup import create_landmarker, detect, draw_landmarks
from normalize import normalize_landmarks
from paths import CLASSIFIER_PATH, CONFIG_PATH, HISTORY_CSV, ML_DIR, resolve
from pipeline import SignStabilizer, polish_sentence
from runtime_config import landmarker_hand_count, resolve_runtime_config

DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "headless": False,
    "raspberry_pi": {
        "mode": "auto",
        "capture": {"width": 640, "height": 480, "frame_rate": 30},
        "touchscreen": {"fullscreen": True, "width": None, "height": None},
    },
    "confidence_threshold": 0.6,
    "letter_hold_seconds": 0.2,
    "word_pause_seconds": 0.45,
    "sentence_pause_seconds": 2.5,
    "smoothing_frames": 5,
    "spell_correct": True,
    "beep_feedback": True,
    "speak_each_word": True,
    "speech_rate": 170,
    "speech_volume": 1.0,
    "gpio_button_pin": None,
    "log_level": "INFO",
    "session_log_path": None,
    "spell_extra_words": ["asl"],
}


def _merged_config(defaults, overrides):
    """Merge the two nested Raspberry Pi sections without mutating defaults."""
    config = dict(defaults)
    config.update(overrides)
    pi = dict(defaults["raspberry_pi"])
    pi.update(overrides.get("raspberry_pi", {}))
    for section in ("capture", "touchscreen"):
        values = dict(defaults["raspberry_pi"][section])
        values.update(pi.get(section, {}))
        pi[section] = values
    config["raspberry_pi"] = pi
    return config


def validate_config(config):
    """Reject invalid runtime values before camera/display initialization."""
    pi = config["raspberry_pi"]
    if pi["mode"] not in {"auto", "enabled", "disabled"}:
        raise ValueError("raspberry_pi.mode must be auto, enabled, or disabled")
    for key in ("camera_index", "camera_width", "camera_height", "smoothing_frames",
                "speech_rate"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("confidence_threshold", "letter_hold_seconds", "word_pause_seconds",
                "sentence_pause_seconds", "speech_volume"):
        if not isinstance(config[key], (int, float)) or config[key] < 0:
            raise ValueError(f"{key} must be a non-negative number")
    if not 0 <= config["confidence_threshold"] <= 1 or not 0 <= config["speech_volume"] <= 1:
        raise ValueError("confidence_threshold and speech_volume must be between 0 and 1")
    capture = pi["capture"]
    for key in ("width", "height", "frame_rate"):
        if isinstance(capture[key], bool) or not isinstance(capture[key], int) or capture[key] <= 0:
            raise ValueError(f"raspberry_pi.capture.{key} must be a positive integer")
    touchscreen = pi["touchscreen"]
    if not isinstance(touchscreen["fullscreen"], bool):
        raise ValueError("raspberry_pi.touchscreen.fullscreen must be boolean")
    for key in ("width", "height"):
        if touchscreen[key] is not None and (isinstance(touchscreen[key], bool) or not isinstance(touchscreen[key], int) or touchscreen[key] <= 0):
            raise ValueError(f"raspberry_pi.touchscreen.{key} must be null or a positive integer")
    return config


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            raise ValueError("top level must be an object")
        return validate_config(_merged_config(DEFAULT_CONFIG, overrides))
    except FileNotFoundError:
        print(f"[config] {CONFIG_PATH} not found - using built-in defaults.")
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
        print(f"[config] Couldn't use {CONFIG_PATH} ({e}) - using built-in defaults.")
    return validate_config(_merged_config(DEFAULT_CONFIG, {}))

config = load_config()
logging.basicConfig(
    level=getattr(logging, str(config["log_level"]).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("translator")

HEADLESS = bool(config["headless"])
RUNTIME = resolve_runtime_config(config["raspberry_pi"])
WORD_PAUSE = float(config["word_pause_seconds"])
SENTENCE_PAUSE = float(config["sentence_pause_seconds"])
BEEP = bool(config["beep_feedback"])
HISTORY_PATH = resolve(config["session_log_path"]) if config["session_log_path"] else HISTORY_CSV


def fail(message):
    """Plain-language fatal error instead of a traceback."""
    print("\n" + "=" * 70 + f"\n  Can't start the translator:\n  {message}\n" + "=" * 70)
    sys.exit(1)


# --- Model --------------------------------------------------------------
if not os.path.exists(CLASSIFIER_PATH):
    fail("No trained model at models/letter_classifier.pkl.\n"
         "  Run:  python ml/train_model.py   (uses data/landmarks.csv)")
try:
    with open(CLASSIFIER_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    fail(f"models/letter_classifier.pkl could not be loaded ({e}).\n"
         "  It was probably made with another scikit-learn version - retrain it\n"
         "  with:  python ml/train_model.py")
SUPPORTED = [str(c).upper() for c in model.classes_]
log.info(f"Model supports {len(SUPPORTED)} letters: {' '.join(SUPPORTED)}")

# --- Spell correction (only for words that aren't already real words) ---
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    _spell.word_frequency.load_words(config["spell_extra_words"])
except ImportError:
    _spell = None
    if config["spell_correct"]:
        log.warning("pyspellchecker not installed - spell correction off.")


def maybe_correct(word):
    if not config["spell_correct"] or _spell is None or not word:
        return word
    lower = word.lower()
    if lower in _spell:
        return lower
    fix = _spell.correction(lower)
    # Only accept a fix one edit away: a big jump would be guessing.
    if fix and fix != lower and fix in _spell.edit_distance_1(lower):
        log.info(f"[spell] '{lower}' -> '{fix}'")
        return fix
    return lower


# --- Speech -------------------------------------------------------------
class Speaker:
    """Non-blocking TTS. Speech never stalls the camera loop and a broken
    audio setup only disables speech, it never crashes the translator."""

    def __init__(self, rate, volume):
        self.rate, self.volume = rate, volume
        self.available = True
        self._proc = None
        try:
            import pyttsx3  # noqa: F401
        except Exception as e:
            self.available = False
            log.warning(f"Text-to-speech unavailable ({e}); text will still be shown.")

    def say(self, text):
        text = text.strip()
        if not text or not self.available:
            return
        log.info(f"[speak] {text}")
        if platform.system() == "Windows":
            # pyttsx3/SAPI5 conflicts with the OpenCV window in one process.
            worker = os.path.join(ML_DIR, "speak_worker.py")
            self._proc = subprocess.Popen([sys.executable, worker, text,
                                           str(self.rate), str(self.volume)])
        else:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            log.error(f"[speak] speech failed: {e}")


speaker = Speaker(config["speech_rate"], config["speech_volume"])


def beep(freq=880, ms=90):
    if not BEEP:
        return

    def _run():
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(int(freq), int(ms))
            else:
                import math, struct, tempfile, wave
                rate = 44100
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                with wave.open(path, "w") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
                    w.writeframes(b"".join(
                        struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq * i / rate)))
                        for i in range(int(rate * ms / 1000))))
                subprocess.run(["aplay", "-q", path], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(path)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


# --- History (text only, never video/images) ----------------------------
def save_history(raw, sentence):
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        new = not os.path.exists(HISTORY_PATH)
        with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "raw_recognition", "processed_sentence"])
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), raw, sentence])
    except OSError as e:
        log.warning(f"Couldn't write history ({e})")


# --- Camera -------------------------------------------------------------
class Camera:
    def __init__(self, cfg):
        self.picam = self.cap = None
        if RUNTIME.use_picamera2:
            try:
                from picamera2 import Picamera2
                capture = cfg["raspberry_pi"]["capture"]
                self.picam = Picamera2()
                self.picam.configure(self.picam.create_video_configuration(
                    main={"size": (capture["width"], capture["height"]), "format": "BGR888"},
                    controls={"FrameDurationLimits": (int(1_000_000 / capture["frame_rate"]),) * 2}))
                self.picam.start()
            except Exception as e:
                log.warning(f"Pi camera unavailable ({e}); trying a USB webcam.")
                self.picam = None
        if self.picam is None:
            self.cap = cv2.VideoCapture(cfg["camera_index"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"No camera found at index {cfg['camera_index']}.\n"
                    "  - Is a webcam plugged in, and not in use by Zoom/Teams/another app?\n"
                    "  - On macOS allow camera access for Terminal/your IDE in\n"
                    "    System Settings > Privacy & Security > Camera.\n"
                    "  - Try \"camera_index\": 1 in ml/config.json if you have several cameras.")

    def read(self):
        if self.picam is not None:
            return True, self.picam.capture_array()
        return self.cap.read()

    def release(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


# --- Optional GPIO "clear" button ---------------------------------------
gpio_clear = threading.Event()
if config["gpio_button_pin"] is not None:
    try:
        from gpiozero import Button
        _btn = Button(config["gpio_button_pin"], pull_up=True)
        _btn.when_pressed = gpio_clear.set
    except Exception as e:
        log.warning(f"GPIO button unavailable ({e}).")


# --- Translator state ---------------------------------------------------
class Session:
    def __init__(self, display=None):
        self.words = []          # finished (spell-checked) words
        self.letters = ""        # letters of the word being signed
        self.last_sentence = ""
        self.last_raw = ""
        self.display = display

    def finish_word(self):
        if not self.letters:
            return
        word = maybe_correct(self.letters)
        self.words.append(word)
        self.letters = ""
        beep(660, 100)
        if config["speak_each_word"]:
            speaker.say(word)

    def finish_sentence(self):
        self.finish_word()
        if not self.words:
            return
        raw, sentence, notes = polish_sentence(self.words)
        for n in notes:
            log.info(f"[grammar] {n}")
        log.info(f"RAW: {raw}   ->   SENTENCE: {sentence}")
        speaker.say(sentence)
        save_history(raw, sentence)
        self.last_raw, self.last_sentence = raw, sentence
        self.words = []
        if self.display is not None:
            try:
                self.display.update_sentence(sentence)
            except Exception as e:
                log.warning(f"Raspberry Pi display update failed ({e}); continuing without it.")
                self.display = None

    def backspace(self):
        if self.letters:
            self.letters = self.letters[:-1]
        elif self.words:
            self.words.pop()

    def clear(self):
        self.words, self.letters = [], ""

    def current_text(self):
        return " ".join(self.words + ([self.letters.lower()] if self.letters else []))


def brightness_of(frame):
    small = cv2.resize(frame, (32, 24))
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()) / 255.0


# --- Drawing ------------------------------------------------------------
GREEN, YELLOW, RED, WHITE, GREY = (80, 220, 80), (0, 220, 255), (60, 60, 240), (255, 255, 255), (170, 170, 170)


def text(frame, s, pos, color=WHITE, scale=0.6, thick=1):
    cv2.putText(frame, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(frame, x, y, w, frac, color):
    cv2.rectangle(frame, (x, y), (x + int(w * max(0, min(1, frac))), y + 10), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + 10), WHITE, 1)


def draw_hud(frame, st):
    h, w = frame.shape[:2]
    # Readiness strip: never blocks, just tells you what's wrong.
    checks = [("Camera", True), ("Hand", st["hands"] > 0), ("Light", st["brightness"] > 0.25)]
    x = 10
    for name, ok in checks:
        text(frame, ("OK " if ok else "-- ") + name, (x, 22), GREEN if ok else YELLOW, 0.5)
        x += 110
    text(frame, f"{st['fps']:.0f} fps  thr {st['threshold']:.2f}", (w - 170, 22), GREY, 0.5)

    if st["paused"]:
        text(frame, "PAUSED - press p to resume", (10, 60), YELLOW, 0.8, 2)
    elif st["hands"] == 0:
        text(frame, "No hand detected - show one hand to the camera", (10, 60), GREY, 0.6)
        if st["brightness"] <= 0.25:
            text(frame, "Too dark: add light in front of you", (10, 85), YELLOW, 0.6)
        if st["pending"]:
            label, frac = st["pending"]
            text(frame, label, (10, 110), YELLOW, 0.55)
            bar(frame, 10, 118, 200, frac, YELLOW)
    else:
        ev = st["event"]
        if ev.kind == "uncertain":
            guess = f" (best guess {ev.letter.upper()} {ev.confidence:.0%})" if ev.letter else ""
            text(frame, "Uncertain - please sign again" + guess, (10, 60), RED, 0.65, 2)
        elif ev.letter:
            col = GREEN if ev.kind in ("commit", "holding") and ev.progress >= 1 else WHITE
            text(frame, f"{ev.letter.upper()}  {ev.confidence:.0%} confidence", (10, 62), col, 0.9, 2)
            bar(frame, 10, 72, 200, ev.progress, GREEN)
        if st["hands"] > 1:
            text(frame, "Two hands seen - reading the first one only", (10, 105), YELLOW, 0.5)

    # Bottom panel: raw recognition vs processed sentence.
    cv2.rectangle(frame, (0, h - 92), (w, h), (20, 20, 20), -1)
    text(frame, "RAW:  " + (st["raw"] or "-"), (10, h - 64), GREY, 0.55)
    text(frame, "TEXT: " + (st["sentence"] or "-"), (10, h - 38), WHITE, 0.7, 2)
    if st["last"]:
        text(frame, "Last: " + st["last"], (10, h - 12), GREEN, 0.5)
    if st["help"]:
        lines = ["SPACE word  ENTER sentence  BKSP delete", "p pause  r replay  s speak  c clear",
                 "+/- threshold  h help  q quit", "Supported: " + " ".join(SUPPORTED)]
        for i, line in enumerate(lines):
            text(frame, line, (10, 150 + 22 * i), GREY, 0.45)


# --- Main loop ----------------------------------------------------------
def main():
    try:
        camera = Camera(config)
    except RuntimeError as e:
        fail(str(e))
    try:
        landmarker = create_landmarker(
            num_hands=landmarker_hand_count(RUNTIME.raspberry_pi_mode))
    except Exception as e:
        camera.release()
        fail(f"The hand-detection model couldn't load ({e}).")

    stab = SignStabilizer(config["confidence_threshold"], config["letter_hold_seconds"],
                          config["smoothing_frames"])
    display = None
    if RUNTIME.use_sentence_display:
        try:
            from pi_display import PiSentenceDisplay
            display = PiSentenceDisplay(config["raspberry_pi"]["touchscreen"])
            log.info("Raspberry Pi sentence display ready.")
        except Exception as e:
            log.warning(f"Raspberry Pi display unavailable ({e}); recognition will continue.")
    s = Session(display)
    paused, show_help = False, True
    no_hand_since = None
    event = stab.hand_lost()
    frame_times, bad_reads = [], 0
    log.info("Runtime mode: %s.", RUNTIME.reason)
    log.info("Translator running. " + ("Ctrl+C to quit." if HEADLESS or not RUNTIME.show_opencv_window else "Press h for keys, q to quit."))

    try:
        while True:
            if s.display is not None:
                try:
                    s.display.pump_events()
                except Exception as e:
                    log.warning(f"Raspberry Pi display event loop failed ({e}); continuing without it.")
                    s.display = None
            ok, frame = camera.read()
            if not ok or frame is None:
                bad_reads += 1
                if bad_reads > 30:
                    log.error("The camera stopped sending frames (unplugged?). Stopping.")
                    break
                time.sleep(0.03)
                continue
            bad_reads = 0
            now = time.time()
            frame_times = (frame_times + [now])[-30:]
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
            frame = cv2.flip(frame, 1)

            hands = []
            if not paused:
                try:
                    result = detect(landmarker, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    hands = result.hand_landmarks or []
                except Exception as e:
                    log.debug(f"skipped a bad frame: {e}")

            pending = None
            if paused:
                event = stab.hand_lost()
            elif hands:
                no_hand_since = None
                try:
                    probs = model.predict_proba([normalize_landmarks(hands[0])])[0]
                    i = int(probs.argmax())
                    letter, conf = str(model.classes_[i]), float(probs[i])
                except Exception as e:
                    log.debug(f"prediction failed: {e}")
                    letter, conf = None, 0.0
                event = stab.push(letter, conf, now)
                if event.kind == "commit":
                    s.letters += event.letter.upper()
                    beep(1200, 50)
                # Pi mode intentionally skips landmark rendering. Desktop mode
                # keeps both detected hands visible while classification uses hands[0].
                if not HEADLESS and RUNTIME.show_hud:
                    for hl in hands:
                        draw_landmarks(frame, hl)
            else:
                event = stab.hand_lost()
                no_hand_since = no_hand_since or now
                gone = now - no_hand_since
                if s.letters:
                    if gone >= WORD_PAUSE:
                        s.finish_word()
                    else:
                        pending = (f"Finishing '{s.letters}'...", gone / WORD_PAUSE)
                elif s.words:
                    if gone >= SENTENCE_PAUSE:
                        s.finish_sentence()
                    else:
                        pending = ("Finishing sentence...", gone / SENTENCE_PAUSE)

            if gpio_clear.is_set():
                s.clear(); gpio_clear.clear()

            if HEADLESS or not RUNTIME.show_opencv_window:
                continue

            raw, sentence, _ = polish_sentence(s.current_text().split())
            if RUNTIME.show_hud:
                draw_hud(frame, {
                    "hands": len(hands), "brightness": brightness_of(frame), "fps": fps,
                    "threshold": stab.confidence_threshold, "paused": paused, "event": event,
                    "pending": pending, "raw": raw, "sentence": sentence,
                    "last": s.last_sentence, "help": show_help})
            cv2.imshow("ASL Fingerspelling Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(
                    "ASL Fingerspelling Translator", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == 32:
                s.finish_word()
            elif key in (13, 10):
                s.finish_sentence()
            elif key in (8, 127):
                s.backspace()
            elif key == ord("c"):
                s.clear(); stab.reset()
            elif key == ord("p"):
                paused = not paused
            elif key == ord("r"):
                speaker.say(s.last_sentence)
            elif key == ord("s"):
                speaker.say(polish_sentence(s.current_text().split())[1])
            elif key == ord("h"):
                show_help = not show_help
            elif key in (ord("+"), ord("=")):
                stab.confidence_threshold = min(0.99, stab.confidence_threshold + 0.05)
            elif key in (ord("-"), ord("_")):
                stab.confidence_threshold = max(0.1, stab.confidence_threshold - 0.05)
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        s.finish_sentence()  # don't lose a half-finished sentence
        camera.release()
        if s.display is not None:
            try:
                s.display.close()
            except Exception as e:
                log.warning(f"Raspberry Pi display close failed ({e}).")
        if not HEADLESS and RUNTIME.show_opencv_window:
            cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
