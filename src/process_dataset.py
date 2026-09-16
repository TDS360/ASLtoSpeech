"""
ASL Sign Language Translator - Build landmarks.csv from a dataset
-----------------------------------------------------------------
An alternative to collect_data.py for anyone who isn't confident in
their own ASL handshapes. Instead of using your own hand, this
processes an existing labeled dataset of correctly-signed ASL
letters and builds data/landmarks.csv from that.

HOW TO USE:
1. Get a free Kaggle account, then download the "ASL Alphabet"
   dataset (search "ASL Alphabet Kaggle grassknoted" -- it's the
   most widely used one, ~87,000 images, free). It's about 1GB.
2. Unzip it. NOTE: Kaggle's zip sometimes extracts with an extra
   nested folder (asl_alphabet_train/asl_alphabet_train/A/...
   instead of asl_alphabet_train/A/...) -- this script searches the
   whole folder tree, so either layout works.
3. Change DATASET_DIR below to the full path of wherever you
   unzipped it.
4. Run this script. It prints progress constantly so you can see
   it's working, processes a sample of images per letter, and
   appends correctly-labeled rows to data/landmarks.csv.

This writes to the SAME landmarks.csv that collect_data.py builds,
so you can use this script, collect_data.py, or both together --
train_model.py doesn't care which one produced the data.

NOTE FOR YOUR PROJECT BOARD: since this uses someone else's dataset
rather than samples of your own hand, be upfront about that in your
write-up -- cite the dataset by name.
"""

import os
import glob
import csv
import cv2
import mediapipe as mp
from normalize import normalize_landmarks
from mp_setup import create_landmarker

DATASET_DIR = r"C:\Users\User\Downloads\archive"  # <-- CHANGE THIS
DATA_DIR = "../data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
VALID_LETTERS = "abcdefghiklmnopqrstuvwxy"  # excludes j and z (motion-based)
MAX_IMAGES_PER_LETTER = 300  # plenty for training; keeps runtime reasonable
PROGRESS_EVERY = 25  # print an update every N images so it never looks stalled

print("Script started.", flush=True)


def find_letter_folder(base_dir, letter):
    """
    Searches the WHOLE folder tree under base_dir for a directory
    named exactly this letter (upper or lower case). This handles
    Kaggle's extra-nested-folder quirk automatically, instead of
    assuming one fixed layout.
    """
    target_upper = letter.upper()
    target_lower = letter.lower()
    for root, dirs, _files in os.walk(base_dir):
        for d in dirs:
            if d == target_upper or d == target_lower:
                return os.path.join(root, d)
    return None


os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"DATASET_DIR does not exist: {DATASET_DIR}\n"
        "Edit the DATASET_DIR variable near the top of this script to "
        "point at wherever you unzipped the dataset."
    )

print("Loading hand landmark model (this can take a few seconds)...", flush=True)
landmarker = create_landmarker(num_hands=1)
print("Model loaded. Scanning dataset folder for letter subfolders...", flush=True)

timestamp_counter = 0
total_saved = 0

for letter in VALID_LETTERS:
    folder = find_letter_folder(DATASET_DIR, letter)
    if folder is None:
        print(f"Skipping {letter.upper()} -- no folder found anywhere under DATASET_DIR", flush=True)
        continue

    image_paths = (glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.png")))[:MAX_IMAGES_PER_LETTER]

    print(f"{letter.upper()}: found {len(image_paths)} images in {folder}", flush=True)

    saved_for_letter = 0
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        for i, path in enumerate(image_paths):
            if (i + 1) % PROGRESS_EVERY == 0:
                print(f"  ...{letter.upper()}: {i + 1}/{len(image_paths)} processed", flush=True)

            image = cv2.imread(path)
            if image is None:
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            timestamp_counter += 1
            result = landmarker.detect_for_video(mp_image, timestamp_counter)

            if result.hand_landmarks:
                features = normalize_landmarks(result.hand_landmarks[0])
                writer.writerow([letter] + features)
                saved_for_letter += 1

    total_saved += saved_for_letter
    print(f"{letter.upper()}: saved {saved_for_letter} / {len(image_paths)} images", flush=True)

landmarker.close()
print(f"\nDone. Total samples added: {total_saved}", flush=True)
if total_saved == 0:
    print("No samples were saved -- double check DATASET_DIR points at the right folder.", flush=True)