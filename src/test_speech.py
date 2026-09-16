"""
Minimal speech test -- completely separate from the rest of the
project. If this doesn't produce sound, the problem is with
pyttsx3/Windows audio setup itself, not with translator.py.

Run with: python test_speech.py
"""

import pyttsx3

print("Available voices on this computer:")
engine = pyttsx3.init()
for voice in engine.getProperty("voices"):
    print(f"  - {voice.name}")

print("\nSpeaking now...")
engine.say("Testing one two three. Can you hear this?")
engine.runAndWait()
print("Done. If you didn't hear anything, check your speaker volume and default output device.")