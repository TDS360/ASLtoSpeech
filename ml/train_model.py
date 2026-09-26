"""
ASL Sign Language Translator - Step 3: Train a Letter Classifier
-------------------------------------------------------------------
Loads data/landmarks.csv, trains several classifiers, reports REAL
measured metrics for each (accuracy, macro precision/recall/F1, mean
inference time), keeps the best one and saves it to
models/letter_classifier.pkl. Metrics are also written to
docs/model_metrics.json so they can be quoted without re-running.

Note: the test split comes from the same signer/session as the
training data, so these numbers are an optimistic upper bound -- real
accuracy with a new signer, lighting or camera will be lower.
"""

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from paths import CLASSIFIER_PATH, DOCS_DIR, LANDMARKS_CSV


def load_dataset(path=LANDMARKS_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No dataset at {path}. Run collect_data.py or process_dataset.py first.")
    df = pd.read_csv(path)
    # collect_data.py / process_dataset.py can append a duplicate header row
    # when a CSV is merged -- drop it instead of training on a class "label".
    df = df[df["label"].astype(str).str.lower() != "label"].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    feature_cols = [c for c in df.columns if c != "label"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna()
    if len(df) != before:
        print(f"Dropped {before - len(df)} malformed rows.")
    return df[feature_cols].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=str)


def main():
    X, y = load_dataset()
    labels = sorted(set(y))
    print(f"Loaded {len(X)} samples across {len(labels)} letters: {' '.join(labels).upper()}")

    counts = pd.Series(y).value_counts()
    low = counts[counts < 20]
    if not low.empty:
        print("Warning: letters with <20 samples (expect weaker accuracy):\n", low)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1200, random_state=42),
    }

    results = {}
    best = (None, None, -1.0)
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        # Time single-sample predictions -- that's what the live loop does.
        sample = X_test[:200]
        t0 = time.perf_counter()
        for row in sample:
            model.predict_proba(row.reshape(1, -1))
        ms = (time.perf_counter() - t0) * 1000 / len(sample)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
                         "inference_ms_per_frame": ms}
        print(f"{name:14s} acc={acc:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} {ms:.2f} ms/frame")

        # Prefer accuracy; break ties with speed (it runs every frame on a Pi).
        score = acc - ms * 1e-4
        if score > best[2]:
            best = (name, model, score)

    best_name, best_model, _ = best
    cm = confusion_matrix(y_test, best_model.predict(X_test), labels=labels)
    confused = [(labels[i], labels[j], int(cm[i, j]))
                for i in range(len(labels)) for j in range(len(labels))
                if i != j and cm[i, j] > 0]
    confused.sort(key=lambda t: -t[2])
    if confused:
        print("Most confused pairs (true -> predicted, count):", confused[:5])

    os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(best_model, f)

    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, "model_metrics.json"), "w") as f:
        json.dump({"selected": best_name, "labels": labels,
                   "train_samples": len(X_train), "test_samples": len(X_test),
                   "results": results, "confusion_matrix": cm.tolist(),
                   "caveat": "Same-signer split; optimistic upper bound."}, f, indent=2)

    print(f"\nSelected: {best_name} -> {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
