import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

from utils import (
    ARTIFACTS_DIR, CLASS_ORDER, C_GRID, MODEL_PATH, N_FOLDS,
    SCRIPT_DIR, TRAIN_PATH, TRAIN_RESULT_PATH, TRAIN_SCORE_PATH, TUNING_PATH,
    evaluate, fit_multiclass, load_xy, make_logger, predict,
)

LOG_PATH = SCRIPT_DIR / "svm_ovr_train.log"
logger = make_logger("svm_ovr_train", LOG_PATH)


def cv_macro_recall(X: np.ndarray, y: np.ndarray, C: float) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = fit_multiclass(X[tr_idx], y[tr_idx], C=C)
        y_pred, _ = predict(clf, X[va_idx])
        mr = recall_score(
            y[va_idx], y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
        )
        scores.append(mr)
        logger.info(f"  fold {fold+1}/{N_FOLDS}  macro_recall={mr:.4f}")
    return float(np.mean(scores)), float(np.std(scores))


def tune(X_train: np.ndarray, y_train: np.ndarray) -> tuple[float, pd.DataFrame]:
    best_mean, best_c = -1.0, 1.0
    rows = []
    for c in C_GRID:
        logger.info(f"--- C={c} ({N_FOLDS}-fold CV) ---")
        mean_mr, std_mr = cv_macro_recall(X_train, y_train, C=c)
        logger.info(f"  C={c}  mean_macro_recall={mean_mr:.4f} ± {std_mr:.4f}")
        rows.append({"C": c, "mean_macro_recall": mean_mr,
                    "std_macro_recall": std_mr})
        if mean_mr > best_mean:
            best_mean, best_c = mean_mr, c
    logger.info(f"Best C={best_c} → mean_macro_recall={best_mean:.4f}")
    return best_c, pd.DataFrame(rows).sort_values("mean_macro_recall", ascending=False)


def main() -> None:
    X_train, y_train = load_xy(TRAIN_PATH, logger)

    logger.info(
        f"=== {N_FOLDS}-fold CV tuning C | class_weight='balanced' ===")
    best_c, tuning_df = tune(X_train, y_train)
    tuning_df.to_csv(TUNING_PATH, index=False)
    logger.info(f"Tuning results saved -> {TUNING_PATH}")

    logger.info(f"=== Training final model: C={best_c} ===")
    final_clf = fit_multiclass(X_train, y_train, C=best_c)

    y_train_pred, train_scores = predict(final_clf, X_train)
    train_metrics = evaluate(y_train, y_train_pred)

    train_out = train_scores.copy()
    train_out["y_true"] = y_train
    train_out["y_pred"] = y_train_pred
    train_out.to_csv(TRAIN_SCORE_PATH, index=False)
    logger.info(f"Train scores saved -> {TRAIN_SCORE_PATH}")

    joblib.dump(
        {"model": final_clf, "class_order": CLASS_ORDER, "best_c": best_c},
        MODEL_PATH,
    )
    logger.info(f"Model saved -> {MODEL_PATH}")

    with open(TRAIN_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"train_metrics": train_metrics, "best_c": best_c},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"Train metrics saved -> {TRAIN_RESULT_PATH}")
    logger.info(f"macro_recall (train) = {train_metrics['macro_recall']:.4f}")
    logger.info("Done! Run evaluate.py to score the held-out test set.")


if __name__ == "__main__":
    main()
