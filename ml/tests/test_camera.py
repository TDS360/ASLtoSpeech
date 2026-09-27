import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import Camera, CameraError
from runtime_config import RuntimeConfig


PI_RUNTIME = RuntimeConfig(True, True, 1, False, False, True, "test")
DESKTOP_RUNTIME = RuntimeConfig(False, False, 2, True, True, False, "test")


def config(*, fallback=False):
    return {
        "camera_index": 3,
        "camera_width": 640,
        "camera_height": 480,
        "raspberry_pi": {
            "allow_opencv_fallback": fallback,
            "capture": {"width": 320, "height": 240, "frame_rate": 20},
        },
    }


class FakeFrame:
    def __init__(self, shape=(240, 320, 3)):
        self.shape = shape


class FakePicamera:
    def __init__(self, *, configure_error=None, start_error=None, frame=None):
        self.configure_error = configure_error
        self.start_error = start_error
        self.frame = frame if frame is not None else FakeFrame()
        self.video_config = None
        self.configured = None
        self.started = self.stopped = self.closed = False

    def create_video_configuration(self, **kwargs):
        self.video_config = kwargs
        return kwargs

    def configure(self, value):
        if self.configure_error:
            raise self.configure_error
        self.configured = value

    def start(self):
        if self.start_error:
            raise self.start_error
        self.started = True

    def capture_array(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.settings = []
        self.released = False

    def set(self, key, value):
        self.settings.append((key, value))

    def isOpened(self):
        return self.opened

    def read(self):
        return True, FakeFrame()

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, opened=True):
        self.capture = FakeCapture(opened)
        self.index = None

    def VideoCapture(self, index):
        self.index = index
        return self.capture


class CameraTests(unittest.TestCase):
    def test_pi_configures_bgr_resolution_frame_rate_and_small_buffer(self):
        picam = FakePicamera()
        camera = Camera(config(), PI_RUNTIME, picamera_factory=lambda: picam)

        self.assertEqual(picam.video_config["main"], {"size": (320, 240), "format": "BGR888"})
        self.assertEqual(picam.video_config["controls"], {"FrameDurationLimits": (50_000, 50_000)})
        self.assertEqual(picam.video_config["buffer_count"], 2)
        self.assertFalse(picam.video_config["queue"])
        self.assertEqual(camera.read(), (True, picam.frame))

    def test_pi_error_does_not_fallback_without_explicit_policy(self):
        with self.assertRaisesRegex(CameraError, "Picamera2/libcamera could not start"):
            Camera(config(), PI_RUNTIME, picamera_factory=lambda: FakePicamera(start_error=RuntimeError("busy")))

    def test_pi_error_uses_opencv_only_when_policy_allows_it(self):
        cv2_module = FakeCV2()
        camera = Camera(config(fallback=True), PI_RUNTIME, cv2_module=cv2_module,
                        picamera_factory=lambda: FakePicamera(configure_error=RuntimeError("bad mode")))

        self.assertIs(camera.cap, cv2_module.capture)
        self.assertEqual(cv2_module.index, 3)

    def test_invalid_pi_frame_has_actionable_error(self):
        camera = Camera(config(), PI_RUNTIME,
                        picamera_factory=lambda: FakePicamera(frame=FakeFrame((240, 320, 4))))
        with self.assertRaisesRegex(CameraError, "invalid frame"):
            camera.read()

    def test_release_is_idempotent_after_partial_startup_and_shutdown_errors(self):
        picam = FakePicamera(start_error=RuntimeError("busy"))
        with self.assertRaises(CameraError):
            Camera(config(), PI_RUNTIME, picamera_factory=lambda: picam)
        self.assertTrue(picam.closed)

        cv2_module = FakeCV2()
        camera = Camera(config(), DESKTOP_RUNTIME, cv2_module=cv2_module)
        camera.release()
        camera.stop()
        self.assertTrue(cv2_module.capture.released)


if __name__ == "__main__":
    unittest.main()
