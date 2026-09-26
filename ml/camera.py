"""Camera selection and lifecycle management for desktop and Raspberry Pi runs.

The module has no translator/model side effects so camera behaviour can be unit
-tested with fake OpenCV and Picamera2 implementations.
"""

from __future__ import annotations

import logging
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Callable, Mapping


class CameraError(RuntimeError):
    """An actionable error while selecting, starting, or reading a camera."""


class Camera:
    """Select Picamera2 in Pi mode, or OpenCV when the policy permits it."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        runtime: Any,
        *,
        cv2_module: Any | None = None,
        picamera_factory: Callable[[], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cfg = cfg
        self._runtime = runtime
        self._cv2 = cv2_module
        self._picamera_factory = picamera_factory
        self._log = logger or logging.getLogger(__name__)
        self.picam: Any | None = None
        self.cap: Any | None = None
        self._picam_started = False

        if runtime.use_picamera2:
            try:
                self._start_picamera2()
            except CameraError as error:
                self.release()
                if not self._allows_opencv_fallback():
                    raise
                self._log.warning("%s Falling back to OpenCV because "
                                  "raspberry_pi.allow_opencv_fallback is enabled.", error)

        if self.picam is None:
            self._start_opencv()

    def _allows_opencv_fallback(self) -> bool:
        return bool(self._cfg["raspberry_pi"].get("allow_opencv_fallback", False))

    def _start_picamera2(self) -> None:
        factory = self._picamera_factory
        if factory is None:
            if find_spec("picamera2") is None:
                raise CameraError(
                    "Raspberry Pi camera mode requires Picamera2, but the 'picamera2' "
                    "package is unavailable. Install it with 'sudo apt install python3-picamera2' "
                    "and use a virtual environment created with --system-site-packages. "
                    "Set raspberry_pi.allow_opencv_fallback to true only to intentionally "
                    "use a USB/OpenCV camera instead."
                )
            factory = import_module("picamera2").Picamera2

        try:
            self.picam = factory()
        except Exception as error:
            raise CameraError(
                "Picamera2 could not open the Raspberry Pi camera. Check the camera cable, "
                "enable the camera interface, and verify it with 'rpicam-hello'. "
                f"Underlying error: {error}"
            ) from error

        capture = self._cfg["raspberry_pi"]["capture"]
        size = (capture["width"], capture["height"])
        frame_duration_us = int(1_000_000 / capture["frame_rate"])
        try:
            video_config = self.picam.create_video_configuration(
                main={"size": size, "format": "BGR888"},
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
                buffer_count=2,
                queue=False,
            )
            self.picam.configure(video_config)
        except Exception as error:
            raise CameraError(
                "Picamera2/libcamera could not configure a BGR888 stream at "
                f"{size[0]}x{size[1]} ({capture['frame_rate']} fps). Check libcamera "
                "with 'rpicam-hello' and choose a supported raspberry_pi.capture size. "
                f"Underlying error: {error}"
            ) from error

        try:
            self.picam.start()
            self._picam_started = True
        except Exception as error:
            raise CameraError(
                "Picamera2/libcamera could not start capture. Ensure no other process is "
                "using the camera and verify it with 'rpicam-hello'. "
                f"Underlying error: {error}"
            ) from error

    def _start_opencv(self) -> None:
        if self._cv2 is None:
            self._cv2 = import_module("cv2")
        self.cap = self._cv2.VideoCapture(self._cfg["camera_index"])
        self.cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._cfg["camera_width"])
        self.cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._cfg["camera_height"])
        if not self.cap.isOpened():
            self.release()
            raise CameraError(
                f"No OpenCV camera found at index {self._cfg['camera_index']}.\n"
                "  - Is a webcam plugged in, and not in use by Zoom/Teams/another app?\n"
                "  - On macOS allow camera access for Terminal/your IDE in\n"
                "    System Settings > Privacy & Security > Camera.\n"
                "  - Try \"camera_index\": 1 in ml/config.json if you have several cameras."
            )

    def read(self) -> tuple[bool, Any | None]:
        if self.picam is None:
            if self.cap is None:
                return False, None
            return self.cap.read()
        try:
            frame = self.picam.capture_array()
        except Exception as error:
            raise CameraError(
                "Picamera2 failed while capturing a frame. Check the camera connection and "
                "whether another process has claimed libcamera. "
                f"Underlying error: {error}"
            ) from error
        if not self._is_bgr_frame(frame):
            shape = getattr(frame, "shape", None)
            raise CameraError(
                "Picamera2 returned an invalid frame "
                f"({shape!r}); expected a non-empty HxWx3 BGR888 image for OpenCV/MediaPipe."
            )
        return True, frame

    @staticmethod
    def _is_bgr_frame(frame: Any) -> bool:
        shape = getattr(frame, "shape", ())
        return (isinstance(shape, tuple) and len(shape) == 3 and shape[0] > 0
                and shape[1] > 0 and shape[2] == 3)

    def stop(self) -> None:
        """Stop active capture; safe to call after partial initialization."""
        self.release()

    def release(self) -> None:
        """Release camera resources once, logging shutdown failures instead of masking errors."""
        picam, cap = self.picam, self.cap
        self.picam = None
        self.cap = None
        was_started = self._picam_started
        self._picam_started = False
        if picam is not None:
            if was_started:
                try:
                    picam.stop()
                except Exception as error:
                    self._log.warning("Picamera2 stop failed during shutdown (%s).", error)
            try:
                close = getattr(picam, "close", None)
                if close is not None:
                    close()
            except Exception as error:
                self._log.warning("Picamera2 close failed during shutdown (%s).", error)
        if cap is not None:
            try:
                cap.release()
            except Exception as error:
                self._log.warning("OpenCV camera release failed during shutdown (%s).", error)
