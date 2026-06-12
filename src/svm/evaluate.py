import json

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    CLASS_ORDER, MODEL_PATH, SCRIPT_DIR, TEST_PATH,
    TEST_RESULT_PATH, TEST_SCORE_PATH,
    evaluate, load_xy, make_logger, predict,
)

LOG_PATH = SCRIPT_DIR / "svm_ovr_evaluate.log"
logger = make_logger("svm_ovr_evaluate", LOG_PATH)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    final_clf = artifact["model"]
    best_c = artifact["best_c"]
    logger.info(f"Loaded model from {MODEL_PATH}  (best_c={best_c})")

    X_test, y_test = load_xy(TEST_PATH, logger)

    y_test_pred, test_scores = predict(final_clf, X_test)
    test_metrics = evaluate(y_test, y_test_pred)

    logger.info(
        "\n=== TEST SET REPORT ===\n" +
        classification_report(y_test, y_test_pred,
                              labels=CLASS_ORDER, zero_division=0)
    )

    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER),
        index=[f"T:{c}" for c in CLASS_ORDER],
        columns=[f"P:{c}" for c in CLASS_ORDER],
    )
    logger.info(
        "\n=== CONFUSION MATRIX (rows=True, cols=Predicted) ===\n" +
        cm_df.to_string()
    )

    test_out = test_scores.copy()
    test_out["y_true"] = y_test
    test_out["y_pred"] = y_test_pred
    test_out.to_csv(TEST_SCORE_PATH, index=False)
    logger.info(f"Test scores saved -> {TEST_SCORE_PATH}")

    with open(TEST_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"test_metrics": test_metrics, "best_c": best_c},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"Test metrics saved -> {TEST_RESULT_PATH}")
    logger.info(f"macro_recall (test) = {test_metrics['macro_recall']:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
