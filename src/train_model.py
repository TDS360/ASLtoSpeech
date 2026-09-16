"""
ASL Sign Language Translator - Step 3: Train a Letter Classifier
-------------------------------------------------------------------
Loads the data YOU collected in data/landmarks.csv, trains three
different classifiers on it, keeps whichever one scores best, and
saves it to models/letter_classifier.pkl.

Run this AFTER collect_data.py -- there's nothing to learn from
until landmarks.csv has real samples in it.
"""

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "../data/landmarks.csv"
MODEL_PATH = "../models/letter_classifier.pkl"

# --- Load your collected data ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "No data/landmarks.csv found yet. Run collect_data.py first "
        "to build your own dataset -- there's nothing to train on yet."
    )

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples across {df['label'].nunique()} letters.")

# Warn about letters with very little data -- these will hurt accuracy
counts = df["label"].value_counts()
low_data = counts[counts < 20]
if not low_data.empty:
    print("\nWarning: these letters have fewer than 20 samples, which "
          "may hurt accuracy for them specifically:")
    print(low_data)

X = df.drop(columns=["label"]).values
y = df["label"].values

# --- Split into training data (learns from this) and test data (never seen) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train a few different classifier types and compare them ---
candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

best_name = None
best_model = None
best_accuracy = 0.0

for name, model in candidates.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n--- {name}: {accuracy:.3f} accuracy ---")
    print(classification_report(y_test, predictions, zero_division=0))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_name = name
        best_model = model

# --- Save whichever model performed best ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model: {best_name} ({best_accuracy:.3f} accuracy on test data)")
print(f"Saved to {MODEL_PATH}")