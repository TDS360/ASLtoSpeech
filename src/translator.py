"""
ASL Sign Language Translator - Step 4: Live Translator (auto-speech)
-----------------------------------------------------------------
Loads your trained model (models/letter_classifier.pkl) and turns
your hand poses into SPOKEN words automatically -- no keypress
needed to hear it. This is designed with the Raspberry Pi version in
mind: a standalone device shouldn't depend on a keyboard for its
core behavior, only for optional testing/quitting.

HOW IT WORKS:
- Hold a letter steady for about half a second and it gets added to
  the word you're currently building (this "debounce" delay stops
  one held pose from being read as many copies of the same letter).
- Once a letter is added, change your hand pose (or drop your hand)
  before signing that SAME letter again, so double letters (like
  the two L's in "HELLO") don't blur into one hold.
- When you drop your hand out of frame and keep it out for about
  1.5 seconds, whatever word you were building is AUTOMATICALLY
  spoken out loud and added to the sentence. Raise your hand again
  to start the next word.

CONTROLS (all optional -- the core behavior above is automatic):
  SPACE      - manually force the current word to finish right now
  BACKSPACE  - delete the last letter of the word you're building
  c          - clear everything
  s          - manually speak everything typed so far
  q          - quit

Run this AFTER collect_data.py/process_dataset.py and train_model.py
-- it needs models/letter_classifier.pkl to already exist.
"""

import cv2
import pickle
import os
import sys
import subprocess
from normalize import normalize_landmarks
from mp_setup import create_landmarker, detect, draw_landmarks

MODEL_PATH = "../models/letter_classifier.pkl"
CONFIDENCE_THRESHOLD = 0.6   # ignore predictions the model isn't confident about
STABILITY_FRAMES = 10         # ~0.33 sec at 30 FPS -- how long a pose must hold steady
NO_HAND_THRESHOLD = 20        # ~0.65 sec at 30 FPS -- how long your hand must be gone to auto-finish a word

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "No trained model found at models/letter_classifier.pkl. "
        "Run collect_data.py or process_dataset.py, then train_model.py, "
        "before running this."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def speak(text):
    """
    Launches speak_worker.py as a separate process to say the text.
    This runs in the background (Popen doesn't wait for it to
    finish), so the camera feed keeps running while it talks.
    """
    text = text.strip()
    if not text:
        return
    print(f"[speak] launching speech for: '{text}'")
    subprocess.Popen([sys.executable, "speak_worker.py", text])


landmarker = create_landmarker(num_hands=2)
cap = cv2.VideoCapture(0)

sentence = ""                   # finished words, each followed by a space
current_word = ""               # letters built up since the last finished word
last_confirmed_letter = None    # last letter added -- blocks accidental repeats
stable_letter = None            # the letter currently being held steady
stable_count = 0                # how many consecutive frames it's been held
no_hand_counter = 0             # how many consecutive frames with NO hand at all
pause_handled = False           # true once the current pause has already triggered speech
previous_hand_present = None    # tracks changes so we can print only on transitions

print("Translator started.")
print("Sign letters to build a word. Drop your hand for ~1.5 sec to speak it automatically.")
print("Controls: SPACE=finish word now, BACKSPACE=delete, c=clear, s=speak all, q=quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detect(landmarker, rgb_frame)

    hand_present = bool(result.hand_landmarks)
    predicted_letter = None
    confidence = 0.0

    if hand_present != previous_hand_present:
        print(f"[debug] hand_present changed to: {hand_present} (current_word='{current_word}')")
        previous_hand_present = hand_present

    if hand_present:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

        # ASL fingerspelling is one-handed -- always classify the
        # first detected hand, even if a second hand is in frame.
        features = normalize_landmarks(result.hand_landmarks[0])
        probabilities = model.predict_proba([features])[0]
        best_index = probabilities.argmax()
        predicted_letter = model.classes_[best_index]
        confidence = probabilities[best_index]

        no_hand_counter = 0
        pause_handled = False
    else:
        no_hand_counter += 1

    # --- Stability / debounce logic for adding letters ---
    if predicted_letter and confidence >= CONFIDENCE_THRESHOLD:
        if predicted_letter == stable_letter:
            stable_count += 1
        else:
            stable_letter = predicted_letter
            stable_count = 1

        if stable_count == STABILITY_FRAMES:
            if predicted_letter != last_confirmed_letter:
                current_word += predicted_letter
                last_confirmed_letter = predicted_letter
    else:
        stable_letter = None
        stable_count = 0
        last_confirmed_letter = None

    # --- Automatic word-finish-and-speak when the hand pauses ---
    if (not hand_present and current_word and not pause_handled
            and no_hand_counter >= NO_HAND_THRESHOLD):
        sentence += current_word + " "
        speak(current_word)
        current_word = ""
        pause_handled = True

    # --- On-screen display ---
    if hand_present and predicted_letter:
        progress = min(stable_count, STABILITY_FRAMES) / STABILITY_FRAMES
        cv2.putText(frame, f"{predicted_letter.upper()} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        bar_width = int(200 * progress)
        cv2.rectangle(frame, (10, 40), (10 + bar_width, 55), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 40), (210, 55), (255, 255, 255), 1)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if current_word:
            pause_progress = min(no_hand_counter, NO_HAND_THRESHOLD) / NO_HAND_THRESHOLD
            cv2.putText(frame, f"Speaking '{current_word}' soon...",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            bar_width = int(200 * pause_progress)
            cv2.rectangle(frame, (10, 70), (10 + bar_width, 85), (0, 255, 255), -1)
            cv2.rectangle(frame, (10, 70), (210, 85), (255, 255, 255), 1)

    cv2.putText(frame, f"Text: {sentence}{current_word}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ASL Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACE -- manually force the current word to finish now
        if current_word:
            sentence += current_word + " "
            speak(current_word)
            current_word = ""
    elif key == 8:   # BACKSPACE (key code can vary slightly by OS/keyboard)
        current_word = current_word[:-1]
    elif key == ord('c'):
        sentence = ""
        current_word = ""
    elif key == ord('s'):
        speak(sentence + current_word)

cap.release()
cv2.destroyAllWindows()
landmarker.close()