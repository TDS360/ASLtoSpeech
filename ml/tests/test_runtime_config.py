import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import detect_raspberry_pi, resolve_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_auto_uses_device_tree_raspberry_pi_metadata(self):
        runtime = resolve_runtime_config(
            {"mode": "auto"}, system="Linux",
            read_text=lambda path: "Raspberry Pi 5 Model B Rev 1.0" if path.endswith("model") else "",
        )
        self.assertTrue(runtime.raspberry_pi_mode)
        self.assertTrue(runtime.use_picamera2)
        self.assertEqual(runtime.num_hands, 1)
        self.assertFalse(runtime.show_opencv_window)
        self.assertTrue(runtime.use_sentence_display)

    def test_auto_falls_back_to_desktop_without_pi_metadata(self):
        runtime = resolve_runtime_config({"mode": "auto"}, system="Linux", read_text=lambda _path: "")
        self.assertFalse(runtime.raspberry_pi_mode)
        self.assertFalse(runtime.use_picamera2)
        self.assertEqual(runtime.num_hands, 2)
        self.assertTrue(runtime.show_opencv_window)

    def test_explicit_modes_override_detection(self):
        read_pi = lambda _path: "Raspberry Pi"
        self.assertFalse(resolve_runtime_config(
            {"mode": "disabled"}, system="Linux", read_text=read_pi).raspberry_pi_mode)
        self.assertTrue(resolve_runtime_config(
            {"mode": "enabled"}, system="Windows", read_text=lambda _path: "").raspberry_pi_mode)

    def test_non_linux_does_not_use_metadata(self):
        self.assertFalse(detect_raspberry_pi(system="Windows", read_text=lambda _path: "Raspberry Pi"))


if __name__ == "__main__":
    unittest.main()
