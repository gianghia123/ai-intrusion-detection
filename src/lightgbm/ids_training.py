import gc, os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.combine import SMOTETomek
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_SPLITS = 3

# ==========================================================
# LOAD DATA
# ==========================================================
TRAIN_PATH = "../dataset/trainset.csv"
TEST_PATH  = "../dataset/testset.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print("Train:", train.shape, "| Test:", test.shape)

# ==========================================================
# LABEL ENCODING & FREQUENCY MAP
# ==========================================================
le = LabelEncoder()
train["attack_cat"] = le.fit_transform(train["attack_cat"].astype(str))
test["attack_cat"]  = le.transform(test["attack_cat"].astype(str))

FREQ_MAP = {}
for col in ["proto", "service", "state"]:
    FREQ_MAP[col] = train[col].value_counts(normalize=True).to_dict()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def preprocess(df, freq_map):
    y = df["attack_cat"]
    X = df.drop(columns=["id", "label", "attack_cat"], errors="ignore").copy()

    # Ratios & rates
    X["bytes_ratio"] = X["sbytes"] / (X["dbytes"] + 1)
    X["pkts_ratio"]  = X["spkts"] / (X["dpkts"] + 1)
    X["byte_rate"]   = (X["sbytes"] + X["dbytes"]) / (X["dur"] + 1)
    X["pkt_rate"]    = (X["spkts"] + X["dpkts"]) / (X["dur"] + 1)

    if "swin" in X.columns and "dwin" in X.columns:
        X["win_ratio"] = X["swin"] / (X["dwin"] + 1)

    # Load asymmetry — phân biệt DoS vs Analysis/Backdoor
    X["src_load"]      = X["sbytes"] / (X["dur"] + 1e-6)
    X["dst_load"]      = X["dbytes"] / (X["dur"] + 1e-6)
    X["load_asym"]     = np.abs(X["src_load"] - X["dst_load"]) / (X["src_load"] + X["dst_load"] + 1)
    X["bpp_src"]       = X["sbytes"] / (X["spkts"] + 1)
    X["bpp_dst"]       = X["dbytes"] / (X["dpkts"] + 1)
    X["pkt_size_diff"] = np.abs(X["bpp_src"] - X["bpp_dst"])

    # Jitter & loss — phân biệt Fuzzers vs Normal
    if "sjit" in X.columns and "djit" in X.columns:
        X["jit_ratio"] = X["sjit"] / (X["djit"] + 1)
        X["jit_sum"]   = X["sjit"] + X["djit"]
    if "sloss" in X.columns and "dloss" in X.columns:
        X["loss_ratio"] = X["sloss"] / (X["dloss"] + 1)
        X["loss_sum"]   = X["sloss"] + X["dloss"]

    # SYN/ACK ratio — phân biệt DoS SYN flood
    if "synack" in X.columns and "ackdat" in X.columns:
        X["synack_ratio"] = X["synack"] / (X["ackdat"] + 1e-6)

    # Log transformation
    log_cols = ["dur", "sbytes", "dbytes", "spkts", "dpkts", "sloss", "dloss",
                "sinpkt", "dinpkt", "sjit", "djit", "response_body_len",
                "src_load", "dst_load", "bpp_src", "bpp_dst"]
    for c in log_cols:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c], 0, None))

    # Protocol flags
    X["is_tcp"]  = (X["proto"] == "tcp").astype(np.int8)
    X["is_udp"]  = (X["proto"] == "udp").astype(np.int8)
    X["is_http"] = X["service"].isin(["http","https","ssl"]).astype(np.int8)
    X["is_dns"]  = (X["service"] == "dns").astype(np.int8)
    X["is_ftp"]  = X["service"].isin(["ftp","ftp-data"]).astype(np.int8)

    # TCP flags
    if "stcpb" in X.columns and "dtcpb" in X.columns:
        X["tcp_established"] = ((X["stcpb"] != 0) & (X["dtcpb"] != 0)).astype(np.int8)
        X["tcp_one_side"]    = ((X["stcpb"] == 0) ^ (X["dtcpb"] == 0)).astype(np.int8)
        X.drop(columns=["stcpb", "dtcpb"], inplace=True)

    # Frequency encoding
    for col in ["proto", "service", "state"]:
        if col in X.columns:
            X[col] = X[col].map(freq_map[col]).fillna(0)

    X = X.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    return X, y

X_train, y_train_raw = preprocess(train, FREQ_MAP)
X_test,  y_test      = preprocess(test,  FREQ_MAP)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

y_train_raw = np.asarray(y_train_raw)
y_test      = np.asarray(y_test)

svd = TruncatedSVD(n_components=5, random_state=RANDOM_STATE)
X_train_svd = svd.fit_transform(X_train_scaled).astype(np.float32)
X_test_svd  = svd.transform(X_test_scaled).astype(np.float32)

print(f"Features: {X_train.shape[1]}")

# ==========================================================
# CLASS WEIGHTS — tập trung tăng recall từng attack
# ==========================================================
counts      = np.bincount(y_train_raw)
n_classes   = len(counts)
class_names = le.classes_

print("\nPhân phối class:")
for i, name in enumerate(class_names):
    print(f"  {name:20s}: {counts[i]:6d}")

focal_weights = {}
for i, name in enumerate(class_names):
    base_w = counts.max() / counts[i]
    # Boost mạnh các class bị nhầm nhiều nhất theo confusion matrix
    if name == 'Analysis':
        focal_weights[i] = np.log1p(base_w) * 7.0   # recall 0.178 → cần boost rất mạnh
    elif name == 'Backdoor':
        focal_weights[i] = np.log1p(base_w) * 6.0   # recall 0.430 → cần boost mạnh
    elif name == 'Worms':
        focal_weights[i] = np.log1p(base_w) * 5.0
    elif name == 'Shellcode':
        focal_weights[i] = np.log1p(base_w) * 2.5
    elif name == 'DoS':
        focal_weights[i] = np.log1p(base_w) * 2.0   # recall 0.414 → boost vừa
    elif name == 'Exploits':
        focal_weights[i] = np.log1p(base_w) * 1.5   # recall 0.602 → boost nhẹ
    elif name == 'Reconnaissance':
        focal_weights[i] = np.log1p(base_w) * 1.3
    elif name == 'Fuzzers':
        focal_weights[i] = np.log1p(base_w) * 0.20  # recall 0.321 nhưng đang nuốt Normal → kìm
    elif name == 'Normal':
        focal_weights[i] = np.log1p(base_w) * 1.30  # bị Fuzzers nuốt → tăng
    elif name == 'Generic':
        focal_weights[i] = np.log1p(base_w) * 0.80
    else:
        focal_weights[i] = np.log1p(base_w) * 1.0

print("\nFocal weights:")
for i, name in enumerate(class_names):
    print(f"  {name:20s}: {focal_weights[i]:.3f}")

# ==========================================================
# SAMPLING — tăng mạnh Analysis/Backdoor, không oversample DoS
# ==========================================================
strategy = {}
for i, name in enumerate(class_names):
    if name == 'Analysis':
        strategy[i] = 14000   # tăng từ 10080
    elif name == 'Backdoor':
        strategy[i] = 14000
    elif name == 'Worms':
        strategy[i] = 7000    # tăng từ 780
    elif name == 'Shellcode':
        strategy[i] = 7000    # tăng từ 5600

strategy = {k: v for k, v in strategy.items() if v > counts[k]}
print("\nSampling strategy:")
for k, v in strategy.items():
    print(f"  {class_names[k]:20s}: {counts[k]} → {v}")

# ==========================================================
# OOF STACKING — RF + LR
# ==========================================================
oof_rf = np.zeros((len(X_train_scaled), n_classes), dtype=np.float32)
oof_lr = np.zeros((len(X_train_scaled), n_classes), dtype=np.float32)

rf_models, lr_models = [], []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print(f"\n🚀 Pipeline RF + LR ({N_SPLITS} folds)...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_raw)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")

    X_tr, y_tr   = X_train_scaled[tr_idx], y_train_raw[tr_idx]
    X_val, y_val = X_train_scaled[val_idx], y_train_raw[val_idx]

    smt = SMOTETomek(random_state=RANDOM_STATE, sampling_strategy=strategy, n_jobs=1)
    X_tr_bal, y_tr_bal = smt.fit_resample(X_tr, y_tr)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=22,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # FIX ConvergenceWarning: tăng max_iter lên 5000
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,       # tăng từ 2000
        C=0.4,
        class_weight="balanced",
        tol=1e-3,
        random_state=RANDOM_STATE,
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

print(f"\nRF OOF: {accuracy_score(y_train_raw, np.argmax(oof_rf, axis=1)):.4f}")
print(f"LR OOF: {accuracy_score(y_train_raw, np.argmax(oof_lr, axis=1)):.4f}")

# ==========================================================
# META FEATURES
# ==========================================================
def create_meta_matrix(prob_rf, prob_lr, svd_features):
    eps = 1e-6

    diff_abs   = np.abs(prob_rf - prob_lr)
    geom_mean  = np.sqrt(prob_rf * prob_lr + eps)
    joint_prod = prob_rf * prob_lr
    kl_div     = prob_rf * np.log((prob_rf + eps) / (prob_lr + eps))

    dot_prod = np.sum(prob_rf * prob_lr, axis=1).reshape(-1, 1)
    norm_rf  = np.sqrt(np.sum(prob_rf**2, axis=1) + eps).reshape(-1, 1)
    norm_lr  = np.sqrt(np.sum(prob_lr**2, axis=1) + eps).reshape(-1, 1)
    cosine   = dot_prod / (norm_rf * norm_lr)

    ent_rf = -np.sum(prob_rf * np.log(prob_rf + eps), axis=1, keepdims=True)
    ent_lr = -np.sum(prob_lr * np.log(prob_lr + eps), axis=1, keepdims=True)

    s_rf = np.sort(prob_rf, axis=1)[:, ::-1]
    s_lr = np.sort(prob_lr, axis=1)[:, ::-1]
    m_rf = (s_rf[:, 0] - s_rf[:, 1]).reshape(-1, 1)
    m_lr = (s_lr[:, 0] - s_lr[:, 1]).reshape(-1, 1)

    disagree = (np.argmax(prob_rf, axis=1) != np.argmax(prob_lr, axis=1)).astype(np.float32).reshape(-1, 1)

    return np.hstack([
        prob_rf, prob_lr,
        diff_abs, geom_mean, joint_prod, kl_div,
        cosine, disagree,
        ent_rf, ent_lr,
        m_rf, m_lr,
        np.max(prob_rf, axis=1).reshape(-1, 1),
        np.max(prob_lr, axis=1).reshape(-1, 1),
        svd_features
    ]).astype(np.float32)

def print_binary_results(y_true_multi, y_pred_multi, class_names, normal_label="Normal"):

    normal_idx = list(class_names).index(normal_label)

    y_true_bin = (y_true_multi != normal_idx).astype(int)
    y_pred_bin = (y_pred_multi != normal_idx).astype(int)

    binary_names = ["Normal", "Attack"]

    print("\n" + "=" * 60)
    print("=== BINARY CLASSIFICATION REPORT ===")
    print("=" * 60)

    print(classification_report(
        y_true_bin,
        y_pred_bin,
        target_names=binary_names,
        digits=4
    ))

    print("\n=== BINARY CONFUSION MATRIX ===")
    print(confusion_matrix(y_true_bin, y_pred_bin))

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print("\n=== BINARY METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

meta_train = create_meta_matrix(oof_rf, oof_lr, X_train_svd)
print(f"\nMeta train shape: {meta_train.shape}")

# ==========================================================
# META LEARNER
# ==========================================================
print("\n🔥 Meta-Learner LightGBM...")

idx_perm     = np.random.RandomState(RANDOM_STATE).permutation(len(meta_train))
val_split    = int(len(meta_train) * 0.85)
tr_meta_idx  = idx_perm[:val_split]
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
    random_state=RANDOM_STATE,
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

# ==========================================================
# INFERENCE
# ==========================================================
print("\n⚙️  Inference...")
rf_test = np.mean([m.predict_proba(X_test_scaled) for m in rf_models], axis=0)
lr_test = np.mean([m.predict_proba(X_test_scaled) for m in lr_models], axis=0)

meta_test  = create_meta_matrix(rf_test, lr_test, X_test_svd)
prob_preds = meta_model.predict_proba(meta_test)

pred = np.argmax(prob_preds, axis=1)

# ==========================================================
# KẾT QUẢ
# ==========================================================
print_binary_results(y_test, pred, le.classes_)

print("\n" + "="*60)
print("=== BÁO CÁO KẾT QUẢ ===")
print("="*60)
print(classification_report(y_test, pred, target_names=le.classes_, digits=4))

print("\n=== MA TRẬN NHẦM LẪN ===")
cm = confusion_matrix(y_test, pred)
print(cm)

print("\n=== PER-CLASS RECALL ===")
for i, name in enumerate(class_names):
    mask = (y_test == i)
    if mask.sum() > 0:
        rec = (pred[mask] == i).mean()
        n   = mask.sum()
        bar = "█" * int(rec * 20)
        print(f"  {name:20s}: {rec:.4f}  {bar}  (n={n})")

print(f"\nAccuracy:  {accuracy_score(y_test, pred):.4f}")
print(f"Recall:    {recall_score(y_test, pred, average='macro'):.4f}")
print(f"Precision: {precision_score(y_test, pred, average='macro'):.4f}")
print(f"Macro F1:  {f1_score(y_test, pred, average='macro'):.4f}")