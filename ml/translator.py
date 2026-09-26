"""Live ASL fingerspelling translator.

The recognition runtime is deliberately constructed by :func:`main`, rather
than at import time.  This keeps ``Session``, sentence finalization, and the
display notification path useful in tests and other Python programs that do
not have a camera, MediaPipe, a trained model, or an audio device.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping

from paths import CLASSIFIER_PATH, CONFIG_PATH, HISTORY_CSV, ML_DIR, resolve
from pipeline import SignStabilizer, polish_sentence
from runtime_config import RuntimeConfig, resolve_runtime_config

DEFAULT_CONFIG = {
    "camera_index": 0, "camera_width": 640, "camera_height": 480,
    "headless": False,
    "raspberry_pi": {"mode": "auto", "capture": {"width": 640, "height": 480, "frame_rate": 30},
                     "allow_opencv_fallback": False, "touchscreen": {"fullscreen": True, "width": None, "height": None}},
    "confidence_threshold": 0.6, "letter_hold_seconds": 0.2, "word_pause_seconds": 0.45,
    "sentence_pause_seconds": 2.5, "smoothing_frames": 5, "spell_correct": True,
    "beep_feedback": True, "speak_each_word": True, "speech_rate": 170, "speech_volume": 1.0,
    "gpio_button_pin": None, "log_level": "INFO", "session_log_path": None, "spell_extra_words": ["asl"],
}


class TranslatorStartupError(RuntimeError):
    """A startup failure with a message a user can act on."""


class ConfigurationError(TranslatorStartupError):
    """The translator configuration could not be resolved."""


class CameraStartupError(TranslatorStartupError):
    """The selected camera backend could not be initialized."""


class ModelLoadError(TranslatorStartupError):
    """The letter classifier could not be found or read."""


class MediaPipeStartupError(TranslatorStartupError):
    """MediaPipe or its hand-landmarker asset could not be initialized."""


class DisplayStartupError(TranslatorStartupError):
    """The configured Pi sentence display could not be initialized."""


class SpeechStartupError(TranslatorStartupError):
    """Text-to-speech could not be initialized."""


def _merged_config(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(defaults); config.update(overrides)
    pi = dict(defaults["raspberry_pi"]); pi.update(overrides.get("raspberry_pi", {}))
    for section in ("capture", "touchscreen"):
        values = dict(defaults["raspberry_pi"][section]); values.update(pi.get(section, {})); pi[section] = values
    config["raspberry_pi"] = pi
    return config


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject invalid runtime values before initializing a resource."""
    pi = config["raspberry_pi"]
    if pi["mode"] not in {"auto", "enabled", "disabled"}:
        raise ValueError("raspberry_pi.mode must be auto, enabled, or disabled")
    for key in ("camera_index", "camera_width", "camera_height", "smoothing_frames", "speech_rate"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("confidence_threshold", "letter_hold_seconds", "word_pause_seconds", "sentence_pause_seconds", "speech_volume"):
        if not isinstance(config[key], (int, float)) or config[key] < 0:
            raise ValueError(f"{key} must be a non-negative number")
    if not 0 <= config["confidence_threshold"] <= 1 or not 0 <= config["speech_volume"] <= 1:
        raise ValueError("confidence_threshold and speech_volume must be between 0 and 1")
    if not isinstance(pi["allow_opencv_fallback"], bool):
        raise ValueError("raspberry_pi.allow_opencv_fallback must be boolean")
    capture = pi["capture"]
    for key in ("width", "height", "frame_rate"):
        if isinstance(capture[key], bool) or not isinstance(capture[key], int) or capture[key] <= 0:
            raise ValueError(f"raspberry_pi.capture.{key} must be a positive integer")
    touchscreen = pi["touchscreen"]
    if not isinstance(touchscreen["fullscreen"], bool):
        raise ValueError("raspberry_pi.touchscreen.fullscreen must be boolean")
    for key in ("width", "height"):
        value = touchscreen[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"raspberry_pi.touchscreen.{key} must be null or a positive integer")
    return dict(config)


def load_config(path=CONFIG_PATH) -> dict[str, Any]:
    """Load config only when an application is being started."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            overrides = json.load(file)
        if not isinstance(overrides, dict): raise ValueError("top level must be an object")
        return validate_config(_merged_config(DEFAULT_CONFIG, overrides))
    except FileNotFoundError:
        logging.getLogger("translator").warning("%s not found; using built-in defaults.", path)
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as error:
        logging.getLogger("translator").warning("Couldn't use %s (%s); using built-in defaults.", path, error)
    return validate_config(_merged_config(DEFAULT_CONFIG, {}))


def configure_logging(config: Mapping[str, Any]) -> logging.Logger:
    logging.basicConfig(level=getattr(logging, str(config["log_level"]).upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("translator")


def create_runtime_config(config: Mapping[str, Any]) -> RuntimeConfig:
    """Resolve the runtime mode before hardware resources are opened."""
    try:
        return resolve_runtime_config(config["raspberry_pi"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Invalid Raspberry Pi runtime configuration: {error}") from error


def load_model(path=CLASSIFIER_PATH):
    try:
        import pickle
    except ImportError as error:
        raise ModelLoadError("Python's pickle support is unavailable; reinstall Python and retrain the model.") from error
    if not os.path.exists(path):
        raise ModelLoadError("No trained model at models/letter_classifier.pkl. Run: python ml/train_model.py")
    try:
        with open(path, "rb") as file: return pickle.load(file)
    except Exception as error:
        raise ModelLoadError("models/letter_classifier.pkl could not be loaded. Retrain it with: "
                             f"python ml/train_model.py (underlying error: {error})") from error


def create_camera(config: Mapping[str, Any], runtime: RuntimeConfig, logger: logging.Logger):
    """Create the configured capture backend without leaking backend exceptions."""
    try:
        from camera import Camera
        return Camera(config, runtime, logger=logger)
    except Exception as error:
        raise CameraStartupError(
            "The camera could not start. Check the camera connection and configured camera index. "
            f"Underlying error: {error}") from error


def create_media_pipe(runtime: RuntimeConfig):
    try:
        from mp_setup import create_landmarker, detect, draw_landmarks
        return create_landmarker(num_hands=runtime.num_hands), detect, draw_landmarks
    except Exception as error:
        raise MediaPipeStartupError(f"The hand-detection model couldn't load: {error}") from error


def create_display(config: Mapping[str, Any], runtime: RuntimeConfig):
    if not runtime.use_sentence_display: return None
    try:
        from pi_display import PiSentenceDisplay
        return PiSentenceDisplay(config["raspberry_pi"]["touchscreen"])
    except Exception as error:
        raise DisplayStartupError("Raspberry Pi display couldn't start. Check the touchscreen/X display "
                                  f"configuration. Underlying error: {error}") from error


class Speaker:
    """Non-blocking TTS, initialized explicitly during application startup."""
    def __init__(self, rate: int, volume: float):
        try: import pyttsx3  # noqa: F401
        except Exception as error:
            raise SpeechStartupError("Text-to-speech is unavailable. Install/configure pyttsx3 and an audio backend. "
                                     f"Underlying error: {error}") from error
        self.rate, self.volume, self._proc = rate, volume, None

    def say(self, text: str) -> None:
        text = text.strip()
        if not text: return
        if platform.system() == "Windows":
            self._proc = subprocess.Popen([sys.executable, os.path.join(ML_DIR, "speak_worker.py"), text, str(self.rate), str(self.volume)])
        else: threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init(); engine.setProperty("rate", self.rate); engine.setProperty("volume", self.volume)
            engine.say(text); engine.runAndWait(); engine.stop()
        except Exception as error: logging.getLogger("translator").error("[speak] speech failed: %s", error)


def create_speaker(config: Mapping[str, Any]) -> Speaker:
    return Speaker(config["speech_rate"], config["speech_volume"])


def create_spellchecker(config: Mapping[str, Any], logger: logging.Logger):
    if not config["spell_correct"]: return None
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(); spell.word_frequency.load_words(config["spell_extra_words"]); return spell
    except ImportError:
        logger.warning("pyspellchecker not installed; spell correction is off."); return None


def save_history(raw: str, sentence: str, history_path=HISTORY_CSV, logger=None) -> None:
    try:
        os.makedirs(os.path.dirname(history_path), exist_ok=True); new = not os.path.exists(history_path)
        with open(history_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new: writer.writerow(["timestamp", "raw_recognition", "processed_sentence"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), raw, sentence])
    except OSError as error: (logger or logging.getLogger("translator")).warning("Couldn't write history (%s)", error)


class Session:
    """Importable recognition text state; hardware is supplied by the caller."""
    def __init__(self, *, display=None, speaker=None, spell=None, spell_correct=True, speak_each_word=True,
                 beep: Callable[[int, int], None] | None = None, history_path=HISTORY_CSV, logger=None):
        self.words, self.letters, self.last_sentence, self.last_raw = [], "", "", ""
        self.display, self.speaker, self.spell, self.spell_correct = display, speaker, spell, spell_correct
        self.speak_each_word, self.beep, self.history_path = speak_each_word, beep or (lambda *_: None), history_path
        self.log = logger or logging.getLogger("translator")

    def finish_word(self):
        if not self.letters: return
        word = self.letters.lower()
        if self.spell_correct and self.spell is not None and word not in self.spell:
            fix = self.spell.correction(word)
            if fix and fix != word and fix in self.spell.edit_distance_1(word): word = fix
        self.words.append(word); self.letters = ""; self.beep(660, 100)
        if self.speak_each_word and self.speaker: self.speaker.say(word)

    def finish_sentence(self):
        self.finish_word()
        if not self.words: return
        raw, sentence, notes = polish_sentence(self.words)
        for note in notes: self.log.info("[grammar] %s", note)
        if self.speaker: self.speaker.say(sentence)
        save_history(raw, sentence, self.history_path, self.log)
        self.last_raw, self.last_sentence, self.words = raw, sentence, []
        if self.display is not None:
            try: self.display.update_sentence(sentence)
            except Exception as error:
                self.log.warning("Raspberry Pi display update failed (%s); continuing without it.", error); self.display = None

    def backspace(self):
        if self.letters: self.letters = self.letters[:-1]
        elif self.words: self.words.pop()
    def clear(self): self.words, self.letters = [], ""
    def current_text(self): return " ".join(self.words + ([self.letters.lower()] if self.letters else []))


def make_beep(enabled: bool) -> Callable[[int, int], None]:
    def beep(freq=880, ms=90):
        if not enabled: return
        def run():
            try:
                if platform.system() == "Windows":
                    import winsound; winsound.Beep(int(freq), int(ms))
            except Exception: pass
        threading.Thread(target=run, daemon=True).start()
    return beep


@dataclass
class RuntimeApplication:
    config: Mapping[str, Any]; runtime: RuntimeConfig; logger: logging.Logger
    camera: Any = None; landmarker: Any = None; detect: Any = None; draw_landmarks: Any = None; display: Any = None; speaker: Any = None

    @classmethod
    def startup(cls) -> "RuntimeApplication":
        config = load_config(); logger = configure_logging(config); runtime = create_runtime_config(config)
        app = cls(config, runtime, logger)
        try:
            app.camera = create_camera(config, runtime, logger)
            app.landmarker, app.detect, app.draw_landmarks = create_media_pipe(runtime)
            app.speaker = create_speaker(config)
            app.display = create_display(config, runtime)
            app._model = load_model()
            logger.info("Model supports %d letters: %s", len(app._model.classes_),
                        " ".join(str(letter).upper() for letter in app._model.classes_))
            return app
        except Exception:
            app.close(); raise

    def close(self) -> None:
        for resource, method in ((self.display, "close"), (self.landmarker, "close"), (self.camera, "release")):
            if resource is not None:
                try: getattr(resource, method)()
                except Exception as error: self.logger.warning("Shutdown failed: %s", error)
        self.display = self.landmarker = self.camera = None

    def run(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise CameraStartupError(
                "OpenCV is unavailable. Install the ML requirements before running the translator. "
                f"Underlying error: {error}") from error
        spell = create_spellchecker(self.config, self.logger)
        history_path = resolve(self.config["session_log_path"]) if self.config["session_log_path"] else HISTORY_CSV
        session = Session(display=self.display, speaker=self.speaker, spell=spell, spell_correct=self.config["spell_correct"],
                          speak_each_word=self.config["speak_each_word"], beep=make_beep(self.config["beep_feedback"]),
                          history_path=history_path, logger=self.logger)
        stabilizer = SignStabilizer(self.config["confidence_threshold"], self.config["letter_hold_seconds"], self.config["smoothing_frames"])
        no_hand_since = None; bad_reads = 0; headless = bool(self.config["headless"])
        self.logger.info("Runtime mode: %s.", self.runtime.reason)
        try:
            while True:
                if session.display is not None:
                    try: session.display.pump_events()
                    except Exception as error: self.logger.warning("Display event loop failed (%s)", error); session.display = None
                try:
                    ok, frame = self.camera.read()
                except Exception as error:
                    raise TranslatorStartupError(
                        f"The camera failed while capturing. Check the camera connection and restart the translator. "
                        f"Underlying error: {error}") from error
                if not ok or frame is None:
                    bad_reads += 1
                    if bad_reads > 30: raise TranslatorStartupError("The camera stopped sending frames. Check its connection and restart the translator.")
                    time.sleep(.03); continue
                bad_reads = 0; now = time.time(); frame = cv2.flip(frame, 1)
                try:
                    result = self.detect(self.landmarker, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    hands = result.hand_landmarks or []
                except Exception as error:
                    raise MediaPipeStartupError(
                        f"Hand detection failed. Check the MediaPipe model and restart the translator. "
                        f"Underlying error: {error}") from error
                if hands:
                    no_hand_since = None
                    probs = self.model.predict_proba([__import__("normalize").normalize_landmarks(hands[0])])[0]
                    index = int(probs.argmax()); event = stabilizer.push(str(self.model.classes_[index]), float(probs[index]), now)
                    if event.kind == "commit": session.letters += event.letter.upper(); session.beep(1200, 50)
                else:
                    stabilizer.hand_lost(); no_hand_since = no_hand_since or now; gone = now - no_hand_since
                    if session.letters and gone >= self.config["word_pause_seconds"]: session.finish_word()
                    elif session.words and gone >= self.config["sentence_pause_seconds"]: session.finish_sentence()
                if headless or not self.runtime.show_opencv_window: continue
                cv2.imshow("ASL Fingerspelling Translator", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
                if key == 32: session.finish_word()
                elif key in (13, 10): session.finish_sentence()
                elif key in (8, 127): session.backspace()
                elif key == ord("c"): session.clear(); stabilizer.reset()
                elif key == ord("r") and self.speaker: self.speaker.say(session.last_sentence)
                elif key == ord("s") and self.speaker: self.speaker.say(polish_sentence(session.current_text().split())[1])
        finally:
            session.finish_sentence()
            if not headless and self.runtime.show_opencv_window: cv2.destroyAllWindows()

    @property
    def model(self): return self._model


def main() -> int:
    try:
        app = RuntimeApplication.startup()
        app.run()
        return 0
    except TranslatorStartupError as error:
        print(f"\nCan't start the translator:\n  {error}", file=sys.stderr); return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if "app" in locals(): app.close()


if __name__ == "__main__":
    raise SystemExit(main())
