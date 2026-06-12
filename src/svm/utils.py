import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[3]

# ── Data paths ────────────────────────────────────────────────────────────────
TRAIN_PATH = SRC_DIR / "dataset" / "train.parquet"
TEST_PATH = SRC_DIR / "dataset" / "test.parquet"

# ── Artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "svm_ovr_10class.joblib"
TUNING_PATH = ARTIFACTS_DIR / "svm_ovr_tuning.csv"
TRAIN_SCORE_PATH = ARTIFACTS_DIR / "train_scores_and_predictions.csv"
TEST_SCORE_PATH = ARTIFACTS_DIR / "test_scores_and_predictions.csv"
TRAIN_RESULT_PATH = ARTIFACTS_DIR / "svm_ovr_train_results.json"
TEST_RESULT_PATH = ARTIFACTS_DIR / "svm_ovr_test_results.json"

# ── Class metadata & hyper-parameter grid ────────────────────────────────────
CLASS_ORDER = [
    "Normal", "DoS", "Fuzzers", "Backdoor", "Analysis",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

ALIASES = {
    "normal": "Normal", "dos": "DoS", "fuzzers": "Fuzzers",
    "backdoor": "Backdoor", "backdoors": "Backdoor",
    "analysis": "Analysis", "exploits": "Exploits",
    "generic": "Generic", "reconnaissance": "Reconnaissance",
    "shellcode": "Shellcode", "worms": "Worms",
}

C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_FOLDS = 5


# ── Logger factory ────────────────────────────────────────────────────────────
def make_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "[%(relativeCreated)d ms] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ── Data helpers ──────────────────────────────────────────────────────────────
def normalize_y(df: pd.DataFrame) -> np.ndarray:
    y_out = []
    for attack_cat, label in zip(
        df["attack_cat"].fillna("").astype(str), df["label"].astype(int)
    ):
        if label == 0:
            y_out.append("Normal")
        else:
            key = attack_cat.strip().lower()
            if key not in ALIASES:
                raise ValueError(f"Unknown attack_cat: '{attack_cat}'")
            y_out.append(ALIASES[key])
    return np.array(y_out)


def load_xy(path: Path, logger: logging.Logger):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)
    y = normalize_y(df)
    X = df.drop(
        columns=[c for c in ["label", "attack_cat"] if c in df.columns]
    ).astype(float)
    logger.info(f"Loaded {path.name}: X={X.shape}")
    for cls in CLASS_ORDER:
        logger.info(f"  [{cls:<18}] n={int((y == cls).sum())}")
    return X.values, y


# ── Model helpers ─────────────────────────────────────────────────────────────
def fit_multiclass(X: np.ndarray, y: np.ndarray, C: float) -> LinearSVC:
    return LinearSVC(
        C=C,
        class_weight="balanced",
        multi_class="ovr",
        max_iter=5000,
        random_state=42,
        dual="auto",
    ).fit(X, y)


def get_score_matrix(clf: LinearSVC, X: np.ndarray) -> np.ndarray:
    raw = clf.decision_function(X)
    cls2idx = {cls: i for i, cls in enumerate(clf.classes_)}
    return np.column_stack([raw[:, cls2idx[cls]] for cls in CLASS_ORDER])


def predict(clf: LinearSVC, X: np.ndarray):
    score_matrix = get_score_matrix(clf, X)
    pred_idx = np.argmax(score_matrix, axis=1)
    y_pred = np.array(CLASS_ORDER)[pred_idx]
    return y_pred, pd.DataFrame(score_matrix, columns=CLASS_ORDER)


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0, output_dict=True
    )
    return {
        "accuracy":              float(accuracy_score(y_true, y_pred)),
        "macro_recall":          float(recall_score(y_true, y_pred, labels=CLASS_ORDER, average="macro",    zero_division=0)),
        "weighted_recall":       float(recall_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "macro_precision":       float(precision_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_f1":              float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro",        zero_division=0)),
        "per_class_recall":      {cls: float(report.get(cls, {}).get("recall", 0.0)) for cls in CLASS_ORDER},
        "confusion_matrix":      confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
        "classification_report": report,
    }
