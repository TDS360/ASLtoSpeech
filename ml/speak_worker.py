"""
Speaks one piece of text in its own process, then exits.

Used on Windows only: pyttsx3's SAPI5 engine can fail silently when it
shares a process with OpenCV's video window.
Usage: python speak_worker.py "text" [rate] [volume]
"""

import sys

try:
    import pyttsx3
except ImportError:
    sys.exit(0)

if len(sys.argv) < 2 or not sys.argv[1].strip():
    sys.exit(0)

engine = pyttsx3.init()
if len(sys.argv) > 2:
    engine.setProperty("rate", int(float(sys.argv[2])))
if len(sys.argv) > 3:
    engine.setProperty("volume", float(sys.argv[3]))
engine.say(sys.argv[1])
engine.runAndWait()
