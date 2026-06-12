import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────────

TRAIN_PATH = "../dataset/trainset.csv"
TEST_PATH = "../dataset/testset.csv"
OUTPUT_DIR = "../artifacts/models"
RESULT_PATH = os.path.join(OUTPUT_DIR, "results.txt")
ARTIFACT_PATH = os.path.join(OUTPUT_DIR, "bundle.joblib")

ARTIFACT_KEYS = ("rf_models", "lr_models", "meta_model", "scaler", "svd", "le")

# ── I/O ────────────────────────────────────────────────────────────────────────


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print("Train:", train.shape, "| Test:", test.shape)
    return train, test


def encode_labels(train, test, le=None):
    if le is None:
        le = LabelEncoder()
        train["attack_cat"] = le.fit_transform(train["attack_cat"].astype(str))
    else:
        train["attack_cat"] = le.transform(train["attack_cat"].astype(str))
    test["attack_cat"] = le.transform(test["attack_cat"].astype(str))
    return train, test, le


def build_freq_map(train):
    return {
        col: train[col].value_counts(normalize=True).to_dict()
        for col in ["proto", "service", "state"]
    }

# ── artifact persistence ───────────────────────────────────────────────────────


def save_artifacts(path, rf_models, lr_models, meta_model, scaler, svd, le):
    bundle = dict(zip(ARTIFACT_KEYS,
                      (rf_models, lr_models, meta_model, scaler, svd, le)))
    joblib.dump(bundle, path)
    print(f"Saved artifacts → {path}")


def load_artifacts(path):
    bundle = joblib.load(path)
    return tuple(bundle[k] for k in ARTIFACT_KEYS)
