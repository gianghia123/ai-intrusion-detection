import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from features import preprocess, scale_features, build_svd_features
from models import (
    build_class_weights,
    build_sampling_strategy,
    print_class_distribution,
    print_focal_weights,
    print_sampling_strategy,
    train_oof_models,
    create_meta_matrix,
    train_meta_model,
    predict_base_models
)
from evaluation import print_binary_results, print_multiclass_results

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 3

TRAIN_PATH = "../dataset/trainset.csv"
TEST_PATH = "../dataset/testset.csv"

OUTPUT_DIR = "../artifacts/models"
RESULT_PATH = os.path.join(OUTPUT_DIR, "results.txt")


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


def encode_labels(train, test):
    le = LabelEncoder()

    train["attack_cat"] = le.fit_transform(train["attack_cat"].astype(str))
    test["attack_cat"] = le.transform(test["attack_cat"].astype(str))

    return train, test, le


def build_freq_map(train):
    freq_map = {}

    for col in ["proto", "service", "state"]:
        freq_map[col] = train[col].value_counts(normalize=True).to_dict()

    return freq_map


def run_pipeline():
    train, test = load_data()
    train, test, le = encode_labels(train, test)

    freq_map = build_freq_map(train)

    X_train, y_train_raw = preprocess(train, freq_map)
    X_test, y_test = preprocess(test, freq_map)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_svd, X_test_svd, svd = build_svd_features(X_train_scaled, X_test_scaled)

    y_train_raw = np.asarray(y_train_raw)
    y_test = np.asarray(y_test)

    print(f"Features: {X_train.shape[1]}")

    counts = np.bincount(y_train_raw)
    n_classes = len(counts)
    class_names = le.classes_

    print_class_distribution(counts, class_names)

    focal_weights = build_class_weights(counts, class_names)
    print_focal_weights(focal_weights, class_names)

    strategy = build_sampling_strategy(counts, class_names)
    print_sampling_strategy(strategy, counts, class_names)

    oof_rf, oof_lr, rf_models, lr_models = train_oof_models(
        X_train_scaled,
        y_train_raw,
        n_classes,
        strategy,
        RANDOM_STATE,
        N_SPLITS
    )

    meta_train = create_meta_matrix(oof_rf, oof_lr, X_train_svd)
    print(f"\nMeta train shape: {meta_train.shape}")

    meta_model = train_meta_model(
        meta_train,
        y_train_raw,
        n_classes,
        focal_weights,
        RANDOM_STATE
    )

    print("\n⚙️  Inference...")

    rf_test, lr_test = predict_base_models(
        rf_models,
        lr_models,
        X_test_scaled
    )

    meta_test = create_meta_matrix(rf_test, lr_test, X_test_svd)
    prob_preds = meta_model.predict_proba(meta_test)

    pred = np.argmax(prob_preds, axis=1)

    print_binary_results(y_test, pred, le.classes_)
    print_multiclass_results(y_test, pred, class_names)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    original_stdout = sys.stdout

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)

        try:
            run_pipeline()
            print(f"\nSaved results to: {RESULT_PATH}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()