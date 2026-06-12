import os
import sys
import warnings
import numpy as np

from features import preprocess
from models import predict_base_models, create_meta_matrix
from evaluation import print_binary_results, print_multiclass_results
from utils import (
    Tee,
    OUTPUT_DIR, RESULT_PATH, ARTIFACT_PATH,
    load_data, encode_labels, build_freq_map, load_artifacts,
)

warnings.filterwarnings("ignore")


def evaluate():
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(
            f"No trained artifacts found at {
                ARTIFACT_PATH}. Run train.py first."
        )

    rf_models, lr_models, meta_model, scaler, svd, le = load_artifacts(
        ARTIFACT_PATH)
    print(f"Loaded artifacts from {ARTIFACT_PATH}")

    train_df, test_df = load_data()
    train_df, test_df, le = encode_labels(train_df, test_df, le=le)

    freq_map = build_freq_map(train_df)

    X_test, y_test = preprocess(test_df, freq_map)
    y_test = np.asarray(y_test)

    X_test_scaled = scaler.transform(X_test)
    X_test_svd = svd.transform(X_test_scaled)

    rf_test, lr_test = predict_base_models(rf_models, lr_models, X_test_scaled)
    meta_test = create_meta_matrix(rf_test, lr_test, X_test_svd)

    prob_preds = meta_model.predict_proba(meta_test)
    pred = np.argmax(prob_preds, axis=1)

    print_binary_results(y_test, pred, le.classes_)
    print_multiclass_results(y_test, pred, le.classes_)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    original_stdout = sys.stdout

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            evaluate()
            print(f"\nResults saved to: {RESULT_PATH}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
