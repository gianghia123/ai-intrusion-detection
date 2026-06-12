import os
import sys
import warnings
import numpy as np

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
)
from utils import (
    Tee,
    OUTPUT_DIR, RESULT_PATH, ARTIFACT_PATH,
    load_data, encode_labels, build_freq_map, save_artifacts,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 3


def train():
    train_df, test_df = load_data()
    train_df, test_df, le = encode_labels(train_df, test_df)

    freq_map = build_freq_map(train_df)

    X_train, y_train_raw = preprocess(train_df, freq_map)
    X_test,  _ = preprocess(test_df,  freq_map)

    X_train_scaled, _, scaler = scale_features(X_train, X_test)
    X_train_svd,    _, svd = build_svd_features(X_train_scaled, _)

    y_train_raw = np.asarray(y_train_raw)

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
        X_train_scaled, y_train_raw, n_classes, strategy, RANDOM_STATE, N_SPLITS
    )

    meta_train = create_meta_matrix(oof_rf, oof_lr, X_train_svd)
    print(f"\nMeta train shape: {meta_train.shape}")

    meta_model = train_meta_model(
        meta_train, y_train_raw, n_classes, focal_weights, RANDOM_STATE
    )

    save_artifacts(ARTIFACT_PATH, rf_models, lr_models,
                   meta_model, scaler, svd, le)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    original_stdout = sys.stdout

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            train()
            print(f"\nResults saved to: {RESULT_PATH}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
