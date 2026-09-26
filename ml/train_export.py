"""
ASL letter classifier: train, evaluate, and export for the browser.

This extends the original train_model.py. It does three things:

  1. Trains several candidate classifiers on data/landmarks.csv
     (the dataset collected with collect_data.py).
  2. Evaluates every candidate on a held-out test split and records
     REAL metrics: accuracy, macro precision / recall / F1, per-class
     scores, a confusion matrix, and mean inference time per sample.
  3. Exports the best browser-runnable model (a small MLP) to
     src/recognition/model/letter-model.json so the web app can run
     inference client-side with no Python server, plus the measured
     metrics to src/recognition/model/metrics.json.

Nothing here is hand-written or estimated: every number in the web
app's "Model" page comes out of this script.

Usage (from the repository root):
    python ml/train_export.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "landmarks.csv")
EXPORT_DIR = os.path.join(ROOT, "src", "recognition", "model")
MODEL_JSON = os.path.join(EXPORT_DIR, "letter-model.json")
METRICS_JSON = os.path.join(EXPORT_DIR, "metrics.json")

RANDOM_STATE = 42


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No dataset at {DATA_PATH}. Run ml/collect_data.py first."
        )
    df = pd.read_csv(DATA_PATH)
    # The original CSV was appended to repeatedly and contains a few
    # duplicated header rows; drop anything that isn't a real sample.
    df = df[df["label"] != "label"]
    df = df.dropna()
    X = df.drop(columns=["label"]).astype(float).values
    y = np.asarray(df["label"].astype(str).str.lower().tolist(), dtype=object)
    return X, y


def timed_inference(model, X_test, repeats=3):
    """Mean wall-clock time to classify ONE sample, in milliseconds."""
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X_test)
        elapsed = (time.perf_counter() - start) / len(X_test) * 1000.0
        best = elapsed if best is None else min(best, elapsed)
    return best


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "name": name,
        "model": model,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "inferenceMs": timed_inference(model, X_test),
        "predictions": predictions,
    }


def export_mlp(mlp, labels, feature_count):
    """Dump an MLP's dense layers to JSON so JS can run the forward pass."""
    layers = []
    for weights, bias in zip(mlp.coefs_, mlp.intercepts_):
        layers.append(
            {
                "weights": [[round(float(v), 6) for v in row] for row in weights],
                "bias": [round(float(v), 6) for v in bias],
            }
        )
    return {
        "format": "mlp-v1",
        "activation": mlp.activation,
        "output": "softmax",
        "featureCount": feature_count,
        "labels": list(labels),
        "layers": layers,
    }


def main():
    X, y = load_dataset()
    labels = sorted(set(y))
    print(f"Loaded {len(X)} samples across {len(labels)} classes: {', '.join(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "MLP (63-128-64)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1200,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    for name, model in candidates.items():
        result = evaluate(name, model, X_train, y_train, X_test, y_test)
        results.append(result)
        print(
            f"{name:18s} acc={result['accuracy']:.4f} "
            f"f1={result['f1']:.4f} {result['inferenceMs']:.4f} ms/sample"
        )

    # The web app runs in the browser, so the exported model must be an MLP
    # (dense matrices are trivial to evaluate in JS). We still report the
    # other candidates honestly so the comparison is visible.
    mlp_result = next(r for r in results if r["name"].startswith("MLP"))
    mlp = mlp_result["model"]

    cv = cross_val_score(
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1200, random_state=RANDOM_STATE),
        X,
        y,
        cv=5,
        n_jobs=-1,
    )
    print(f"5-fold CV (MLP): {cv.mean():.4f} +/- {cv.std():.4f}")

    predictions = mlp_result["predictions"]
    matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()
    per_class = precision_recall_fscore_support(
        y_test, predictions, labels=labels, zero_division=0
    )

    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(MODEL_JSON, "w") as f:
        json.dump(export_mlp(mlp, labels, X.shape[1]), f)

    metrics = {
        "generatedAt": time.strftime("%Y-%m-%d"),
        "dataset": {
            "samples": int(len(X)),
            "classes": labels,
            "trainSamples": int(len(X_train)),
            "testSamples": int(len(X_test)),
            "perClass": {label: int((y == label).sum()) for label in labels},
        },
        "deployedModel": "MLP (63-128-64)",
        "crossValidation": {"folds": 5, "mean": float(cv.mean()), "std": float(cv.std())},
        "candidates": [
            {
                "name": r["name"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "inferenceMs": r["inferenceMs"],
                "deployed": r["name"] == mlp_result["name"],
            }
            for r in results
        ],
        "confusionMatrix": {"labels": labels, "matrix": matrix},
        "perClass": [
            {
                "label": label,
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i, label in enumerate(labels)
        ],
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nExported model  -> {MODEL_JSON}")
    print(f"Exported metrics -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
