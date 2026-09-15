"""
ASL Sign Language Translator - Step 2: Data Collection
-----------------------------------------------------------------
This script builds data/landmarks.csv from scratch, using YOUR hand
and YOUR webcam. There is no file to download -- you are creating
your own training dataset, which is exactly what you want for a
STEM fair project.

HOW TO USE:
  1. Run this script.
  2. Press a letter key (a-i, k-y -- NOT j or z, since those involve
     motion) to select which letter you're about to sign.
  3. Hold that handshape steady in front of the camera.
  4. Press SPACE to save one sample of your current hand pose.
  5. Repeat step 4 many times per letter, adjusting your hand
     slightly each time (angle, distance, position in frame) so the
     model learns to generalize, not memorize one exact pose.
  6. Aim for 100+ samples per letter. Collect extra for A, S, O and
     M, N -- these are the letters most often confused.
  7. Press 'q' to quit whenever you're done for the session. You can
     run this script again later to add more samples -- it appends
     to the same CSV instead of overwriting it.

Needs models/hand_landmarker.task downloaded first -- see mp_setup.py
for the download command if you're missing it.
"""

import cv2
import csv
import os
from normalize import normalize_landmarks
from mp_setup import create_landmarker, detect, draw_landmarks

DATA_DIR = "../data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
VALID_LETTERS = "abcdefghiklmnopqrstuvwxy"  # excludes j and z (motion-based)

os.makedirs(DATA_DIR, exist_ok=True)

# Create the CSV with a header row if it doesn't exist yet
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

landmarker = create_landmarker(num_hands=2)
cap = cv2.VideoCapture(0)
current_letter = None
sample_counts = {letter: 0 for letter in VALID_LETTERS}

# If the CSV already has data from a previous session, load real counts
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0] in sample_counts:
                sample_counts[row[0]] += 1

print("Data collection started.")
print("Press a letter key (a-i, k-y -- no j/z) to select that letter.")
print("Press SPACE to save a sample of your current hand pose.")
print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detect(landmarker, rgb_frame)

    landmarks_ready = None
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)
        # ASL fingerspelling is one-handed, so even if a second hand
        # wanders into frame, we only collect data from the first
        # hand MediaPipe detects.
        landmarks_ready = normalize_landmarks(result.hand_landmarks[0])

    # --- On-screen info ---
    letter_display = current_letter.upper() if current_letter else "-"
    count_display = sample_counts[current_letter] if current_letter else 0
    cv2.putText(frame, f"Letter: {letter_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Samples: {count_display}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if landmarks_ready is None:
        cv2.putText(frame, "No hand detected", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Data Collection - press letter key, then SPACE to save", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:  # SPACE bar
        if current_letter and landmarks_ready:
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_letter] + landmarks_ready)
            sample_counts[current_letter] += 1
    elif 0 <= key < 256 and chr(key) in VALID_LETTERS:
        current_letter = chr(key)

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\nSession complete. Samples collected per letter:")
for letter, count in sample_counts.items():
    if count > 0:
        print(f"  {letter.upper()}: {count}")
missing = [l.upper() for l, c in sample_counts.items() if c == 0]
if missing:
    print(f"\nNot started yet: {', '.join(missing)}")