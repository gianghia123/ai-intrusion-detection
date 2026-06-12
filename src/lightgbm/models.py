import gc
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek


def print_class_distribution(counts, class_names):
    print("\nPhân phối class:")

    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {counts[i]:6d}")


def build_class_weights(counts, class_names):
    focal_weights = {}

    for i, name in enumerate(class_names):
        base_w = counts.max() / counts[i]

        if name == "Analysis":
            focal_weights[i] = np.log1p(base_w) * 7.0
        elif name == "Backdoor":
            focal_weights[i] = np.log1p(base_w) * 6.0
        elif name == "Worms":
            focal_weights[i] = np.log1p(base_w) * 5.0
        elif name == "Shellcode":
            focal_weights[i] = np.log1p(base_w) * 2.5
        elif name == "DoS":
            focal_weights[i] = np.log1p(base_w) * 2.0
        elif name == "Exploits":
            focal_weights[i] = np.log1p(base_w) * 1.5
        elif name == "Reconnaissance":
            focal_weights[i] = np.log1p(base_w) * 1.3
        elif name == "Fuzzers":
            focal_weights[i] = np.log1p(base_w) * 0.20
        elif name == "Normal":
            focal_weights[i] = np.log1p(base_w) * 1.30
        elif name == "Generic":
            focal_weights[i] = np.log1p(base_w) * 0.80
        else:
            focal_weights[i] = np.log1p(base_w) * 1.0

    return focal_weights


def print_focal_weights(focal_weights, class_names):
    print("\nFocal weights:")

    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {focal_weights[i]:.3f}")


def build_sampling_strategy(counts, class_names):
    strategy = {}

    for i, name in enumerate(class_names):
        if name == "Analysis":
            strategy[i] = 14000
        elif name == "Backdoor":
            strategy[i] = 14000
        elif name == "Worms":
            strategy[i] = 7000
        elif name == "Shellcode":
            strategy[i] = 7000

    strategy = {k: v for k, v in strategy.items() if v > counts[k]}

    return strategy


def print_sampling_strategy(strategy, counts, class_names):
    print("\nSampling strategy:")

    for k, v in strategy.items():
        print(f"  {class_names[k]:20s}: {counts[k]} → {v}")


def train_oof_models(X_train_scaled, y_train_raw, n_classes, strategy, random_state=42, n_splits=3):
    oof_rf = np.zeros((len(X_train_scaled), n_classes), dtype=np.float32)
    oof_lr = np.zeros((len(X_train_scaled), n_classes), dtype=np.float32)

    rf_models = []
    lr_models = []

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    print(f"\nPipeline RF + LR ({n_splits} folds)...")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_raw)):
        print(f"--- Fold {fold + 1}/{n_splits} ---")

        X_tr, y_tr = X_train_scaled[tr_idx], y_train_raw[tr_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train_raw[val_idx]

        smt = SMOTETomek(
            random_state=random_state,
            sampling_strategy=strategy,
            n_jobs=1
        )

        X_tr_bal, y_tr_bal = smt.fit_resample(X_tr, y_tr)

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=22,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        )

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            C=0.4,
            class_weight="balanced",
            tol=1e-3,
            random_state=random_state,
            n_jobs=-1
        )

        rf.fit(X_tr_bal, y_tr_bal)
        lr.fit(X_tr_bal, y_tr_bal)

        oof_rf[val_idx] = rf.predict_proba(X_val)
        oof_lr[val_idx] = lr.predict_proba(X_val)

        rf_models.append(rf)
        lr_models.append(lr)

        acc_rf = accuracy_score(y_val, np.argmax(oof_rf[val_idx], axis=1))
        acc_lr = accuracy_score(y_val, np.argmax(oof_lr[val_idx], axis=1))

        print(f"    RF: {acc_rf:.4f} | LR: {acc_lr:.4f}")

        del X_tr, y_tr, X_tr_bal, y_tr_bal
        gc.collect()

    print(f"\nRF OOF: {accuracy_score(
        y_train_raw, np.argmax(oof_rf, axis=1)):.4f}")
    print(f"LR OOF: {accuracy_score(
        y_train_raw, np.argmax(oof_lr, axis=1)):.4f}")

    return oof_rf, oof_lr, rf_models, lr_models


def create_meta_matrix(prob_rf, prob_lr, svd_features):
    eps = 1e-6

    diff_abs = np.abs(prob_rf - prob_lr)
    geom_mean = np.sqrt(prob_rf * prob_lr + eps)
    joint_prod = prob_rf * prob_lr
    kl_div = prob_rf * np.log((prob_rf + eps) / (prob_lr + eps))

    dot_prod = np.sum(prob_rf * prob_lr, axis=1).reshape(-1, 1)
    norm_rf = np.sqrt(np.sum(prob_rf ** 2, axis=1) + eps).reshape(-1, 1)
    norm_lr = np.sqrt(np.sum(prob_lr ** 2, axis=1) + eps).reshape(-1, 1)
    cosine = dot_prod / (norm_rf * norm_lr)

    ent_rf = -np.sum(prob_rf * np.log(prob_rf + eps), axis=1, keepdims=True)
    ent_lr = -np.sum(prob_lr * np.log(prob_lr + eps), axis=1, keepdims=True)

    s_rf = np.sort(prob_rf, axis=1)[:, ::-1]
    s_lr = np.sort(prob_lr, axis=1)[:, ::-1]

    m_rf = (s_rf[:, 0] - s_rf[:, 1]).reshape(-1, 1)
    m_lr = (s_lr[:, 0] - s_lr[:, 1]).reshape(-1, 1)

    disagree = (np.argmax(prob_rf, axis=1) != np.argmax(
        prob_lr, axis=1)).astype(np.float32).reshape(-1, 1)

    return np.hstack([
        prob_rf,
        prob_lr,
        diff_abs,
        geom_mean,
        joint_prod,
        kl_div,
        cosine,
        disagree,
        ent_rf,
        ent_lr,
        m_rf,
        m_lr,
        np.max(prob_rf, axis=1).reshape(-1, 1),
        np.max(prob_lr, axis=1).reshape(-1, 1),
        svd_features
    ]).astype(np.float32)


def train_meta_model(meta_train, y_train_raw, n_classes, focal_weights, random_state=42):
    print("\nMeta-Learner LightGBM...")

    idx_perm = np.random.RandomState(random_state).permutation(len(meta_train))
    val_split = int(len(meta_train) * 0.85)

    tr_meta_idx = idx_perm[:val_split]
    val_meta_idx = idx_perm[val_split:]

    meta_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.006,
        n_estimators=3000,
        num_leaves=70,
        max_depth=12,
        min_child_samples=20,
        max_bin=255,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=2.5,
        reg_lambda=6.0,
        class_weight=focal_weights,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1
    )

    meta_model.fit(
        meta_train[tr_meta_idx], y_train_raw[tr_meta_idx],
        eval_set=[(meta_train[val_meta_idx], y_train_raw[val_meta_idx])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=True),
            lgb.log_evaluation(period=200)
        ]
    )

    return meta_model


def predict_base_models(rf_models, lr_models, X_test_scaled):
    rf_test = np.mean([m.predict_proba(X_test_scaled)
                      for m in rf_models], axis=0)
    lr_test = np.mean([m.predict_proba(X_test_scaled)
                      for m in lr_models], axis=0)

    return rf_test, lr_test
