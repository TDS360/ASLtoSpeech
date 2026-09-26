"""
ASL Sign Language Translator - Optional: Data Augmentation
------------------------------------------------------------
Expands data/landmarks.csv by generating small synthetic variations
of each real sample you collected, without needing to record any
more by hand. This helps the classifier generalize to hand angles
and distances it hasn't seen an exact match for.

Run this AFTER collect_data.py (or process_dataset.py) and BEFORE
train_model.py. It reads landmarks.csv and writes a NEW file,
landmarks_augmented.csv, containing your original samples PLUS the
generated variations -- your original landmarks.csv is never
modified or overwritten.

To train on the augmented data, either:
  - change train_model.py's DATA_PATH to "../data/landmarks_augmented.csv", or
  - rename/copy landmarks_augmented.csv over landmarks.csv yourself.

HOW THE VARIATIONS WORK:
- Jitter: small random noise added to every x/y/z coordinate, to
  simulate natural hand shake / camera noise.
- Scaling: each sample is shrunk or grown slightly, to simulate the
  hand being a little closer to or farther from the camera.
(Both are applied to the already-normalized landmarks from
normalize.py, so they stay centered on the wrist correctly.)
"""

import os
import pandas as pd
import numpy as np

from paths import DATA_DIR
INPUT_PATH = os.path.join(DATA_DIR, "landmarks.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "landmarks_augmented.csv")
VARIATIONS_PER_SAMPLE = 4      # how many synthetic copies to generate per real sample
JITTER_STD = 0.01              # standard deviation of the random noise added per coordinate
SCALE_RANGE = (0.9, 1.1)       # random scale factor range applied per synthetic sample
RANDOM_SEED = 42               # keep this fixed so results are reproducible for your write-up

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"No {INPUT_PATH} found. Run collect_data.py or process_dataset.py first "
        "so there's real data to augment."
    )

rng = np.random.default_rng(RANDOM_SEED)
df = pd.read_csv(INPUT_PATH)
feature_cols = [c for c in df.columns if c != "label"]

print(f"Loaded {len(df)} original samples across {df['label'].nunique()} letters.")

augmented_rows = []
for _, row in df.iterrows():
    label = row["label"]
    features = row[feature_cols].to_numpy(dtype=float)

    for _ in range(VARIATIONS_PER_SAMPLE):
        jitter = rng.normal(0, JITTER_STD, size=features.shape)
        scale = rng.uniform(*SCALE_RANGE)
        synthetic = features * scale + jitter
        augmented_rows.append([label] + synthetic.tolist())

augmented_df = pd.DataFrame(augmented_rows, columns=["label"] + feature_cols)
combined_df = pd.concat([df, augmented_df], ignore_index=True)
combined_df.to_csv(OUTPUT_PATH, index=False)

print(f"Generated {len(augmented_df)} synthetic samples "
      f"({VARIATIONS_PER_SAMPLE} per original).")
print(f"Wrote {len(combined_df)} total samples to {OUTPUT_PATH}")
print("\nTo train on this data, point train_model.py's DATA_PATH at "
      "'../data/landmarks_augmented.csv' and re-run it.")