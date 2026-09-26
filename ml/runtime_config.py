"""Small, dependency-free runtime selection for desktop and Raspberry Pi runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Callable, Mapping

_PI_METADATA_PATHS = (
    "/proc/device-tree/model",
    "/proc/device-tree/compatible",
    "/sys/firmware/devicetree/base/model",
    "/sys/firmware/devicetree/base/compatible",
)
_PI_MARKERS = ("raspberry pi", "raspberrypi", "brcm,bcm")


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved operating mode shared by capture, tracking, and presentation."""

    raspberry_pi_mode: bool
    use_picamera2: bool
    num_hands: int
    show_opencv_window: bool
    show_hud: bool
    use_sentence_display: bool
    reason: str


def detect_raspberry_pi(
    system: str | None = None,
    read_text: Callable[[str], str] | None = None,
) -> bool:
    """Return true only for Linux systems whose device-tree identifies a Pi.

    Device-tree files are supplied by Linux firmware, unlike hostnames or CPU
    architecture which are not reliable identifiers (especially in containers).
    """
    if (system or platform.system()).lower() != "linux":
        return False
    if read_text is None:
        def read_text(path: str) -> str:
            try:
                return Path(path).read_bytes().decode("utf-8", errors="ignore")
            except OSError:
                return ""

    metadata = " ".join(read_text(path).replace("\x00", " ").lower()
                         for path in _PI_METADATA_PATHS)
    return any(marker in metadata for marker in _PI_MARKERS)


def landmarker_hand_count(raspberry_pi_mode: bool) -> int:
    """Return the detector capacity appropriate for the resolved runtime mode.

    Raspberry Pi runs optimize for a single signing hand. Desktop runs continue
    detecting two hands so the translator can warn the user while classifying
    only the first detected hand.
    """
    return 1 if raspberry_pi_mode else 2


def resolve_runtime_config(
    raspberry_pi: Mapping[str, object],
    *,
    system: str | None = None,
    read_text: Callable[[str], str] | None = None,
) -> RuntimeConfig:
    """Resolve ``auto``, ``enabled``, or ``disabled`` Raspberry Pi mode.

    ``auto`` uses device-tree detection and falls back to desktop mode when no
    supported Raspberry Pi metadata is present. ``enabled`` is an explicit
    override useful for Pi-like test images; ``disabled`` always preserves the
    normal desktop webcam/OpenCV workflow.
    """
    mode = str(raspberry_pi.get("mode", "auto")).lower()
    if mode not in {"auto", "enabled", "disabled"}:
        raise ValueError("raspberry_pi.mode must be 'auto', 'enabled', or 'disabled'")

    detected = detect_raspberry_pi(system=system, read_text=read_text)
    pi_mode = mode == "enabled" or (mode == "auto" and detected)
    if mode == "enabled":
        reason = "forced by raspberry_pi.mode=enabled"
    elif mode == "disabled":
        reason = "disabled by raspberry_pi.mode=disabled"
    elif detected:
        reason = "Raspberry Pi detected from Linux device-tree metadata"
    else:
        reason = "no Raspberry Pi Linux device-tree metadata; using desktop mode"

    return RuntimeConfig(
        raspberry_pi_mode=pi_mode,
        use_picamera2=pi_mode,
        num_hands=landmarker_hand_count(pi_mode),
        show_opencv_window=not pi_mode,
        show_hud=not pi_mode,
        use_sentence_display=pi_mode,
        reason=reason,
    )
