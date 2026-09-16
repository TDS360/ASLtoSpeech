"""
Tiny standalone script that speaks one piece of text, then exits.

translator.py launches this as a completely separate program each
time it needs to talk, instead of calling pyttsx3 directly inside
the main script. Running speech in its own process avoids a known
conflict between pyttsx3's Windows speech engine and OpenCV's video
window, which can otherwise cause speech to fail silently even
though the exact same pyttsx3 code works fine on its own.
"""

import sys
import pyttsx3

if len(sys.argv) < 2:
    sys.exit(0)

text = sys.argv[1]
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()