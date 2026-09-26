"""
Single source of truth for file locations.

Every script used to hard-code paths like "../models/...", which only
worked when you launched Python from inside the ml/ folder. These paths
are resolved relative to THIS file, so the scripts work no matter which
folder you run them from (e.g. `python ml/translator.py` from the
project root, or `python translator.py` from inside ml/).
"""

import os

ML_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ML_DIR)

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
DOCS_DIR = os.path.join(ROOT, "docs")

LANDMARKS_CSV = os.path.join(DATA_DIR, "landmarks.csv")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "letter_classifier.pkl")
CONFIG_PATH = os.path.join(ML_DIR, "config.json")
HISTORY_CSV = os.path.join(DOCS_DIR, "translation_history.csv")


def hand_landmarker_path():
    """models/hand_landmarker.task, falling back to public/models/."""
    for candidate in (
        os.path.join(MODELS_DIR, "hand_landmarker.task"),
        os.path.join(ROOT, "public", "models", "hand_landmarker.task"),
    ):
        if os.path.exists(candidate):
            return candidate
    return os.path.join(MODELS_DIR, "hand_landmarker.task")


def resolve(path):
    """Resolve a path from config.json relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(ROOT, path))
