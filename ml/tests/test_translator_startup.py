"""Startup-boundary tests for the live translator."""

from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
import translator
from runtime_config import RuntimeConfig


class _Resource:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Camera(_Resource):
    def release(self):
        self.closed = True


class TranslatorStartupTests(unittest.TestCase):
    def test_import_does_not_load_runtime_dependencies(self):
        code = "import sys; import translator; print(','.join(sys.modules))"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ML_DIR, text=True, capture_output=True, check=True
        )
        modules = set(result.stdout.strip().split(","))
        self.assertFalse({"cv2", "mediapipe", "pyttsx3", "camera", "mp_setup", "pi_display"} & modules)

    def test_camera_creation_wraps_backend_failure_in_typed_error(self):
        with patch.object(importlib, "import_module", side_effect=AssertionError("not used")):
            with patch.dict(sys.modules, {"camera": None}):
                with self.assertRaises(translator.CameraStartupError):
                    translator.create_camera({}, RuntimeConfig(False, False, 2, True, True, False, "test"),
                                             translator.logging.getLogger("test"))

    def test_startup_failure_closes_resources_already_created(self):
        camera, landmarker = _Camera(), _Resource()
        config = translator._merged_config(translator.DEFAULT_CONFIG, {})
        runtime = RuntimeConfig(False, False, 2, True, True, False, "test")
        with patch.object(translator, "load_config", return_value=config), \
             patch.object(translator, "configure_logging", return_value=translator.logging.getLogger("test")), \
             patch.object(translator, "create_runtime_config", return_value=runtime), \
             patch.object(translator, "create_camera", return_value=camera), \
             patch.object(translator, "create_media_pipe", return_value=(landmarker, object(), object())), \
             patch.object(translator, "create_speaker", return_value=object()), \
             patch.object(translator, "create_display", side_effect=translator.DisplayStartupError("no display")):
            with self.assertRaisesRegex(translator.DisplayStartupError, "no display"):
                translator.RuntimeApplication.startup()

        self.assertTrue(camera.closed)
        self.assertTrue(landmarker.closed)

    def test_session_finalization_stays_hardware_independent(self):
        display = _Resource()
        display.update_sentence = lambda sentence: setattr(display, "sentence", sentence)
        session = translator.Session(display=display, spell_correct=False, history_path="/dev/null")
        session.letters = "HELLO"
        session.finish_sentence()
        self.assertEqual(session.last_raw, "HELLO")
        self.assertEqual(display.sentence, "Hello.")


if __name__ == "__main__":
    unittest.main()
