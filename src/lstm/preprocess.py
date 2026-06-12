import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.metrics import make_scorer, recall_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

DROPS = ['smean', 'dmean', 'sload', 'dload']
LOG1P = ['sbytes', 'dbytes', 'spkts', 'dpkts', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'res_bdy_len']
FLAGS = ["is_ftp_login", "is_sm_ips_ports", "is_tcp", "is_ftp", "is_http", "tcp_seq_established", "tcp_seq_one_sided", "is_zero_dur", "is_short_flow", "zero_win"]

# =====================================================================
# HÀM BỔ TRỢ: GIẢI MÃ SỐ HEX (0x...) MẠNH MẼ HƠN
# =====================================================================
def force_to_numeric(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip()
    if val_str == '' or val_str == '-':
        return 0.0
        
    if val_str.lower().startswith('0x'):
        try:
            return float(int(val_str, 16))
        except:
            return 0.0
            
    try:
        return float(val_str)
    except:
        return 0.0

def process(df: pd.DataFrame, is_train: bool, artifacts: dict) -> tuple:
    df = df.drop(columns=['id'], errors='ignore')
    
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str)
        df['attack_cat'] = df['attack_cat'].str.strip().str.capitalize()
        df['attack_cat'] = df['attack_cat'].replace({
            'Backdoors': 'Backdoor',
            '': 'Normal', 
            'None': 'Normal'
        })
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    y_label = df['label'].copy().reset_index(drop=True)
    y_cat = df['attack_cat'].copy().reset_index(drop=True)
    df = df.drop(columns=['label', 'attack_cat'], errors='ignore').reset_index(drop=True)

    if 'stcpb' in df.columns and 'dtcpb' in df.columns:
        df['tcp_seq_established'] = ((df['stcpb'] != 0) & (df['dtcpb'] != 0)).astype(int)
        df['tcp_seq_one_sided'] = ((df['stcpb'] != 0) ^ (df['dtcpb'] != 0)).astype(int)
        df = df.drop(columns=['stcpb', 'dtcpb'], errors='ignore')

    df = df.drop(columns=DROPS, errors='ignore')

    if 'rate' in df.columns:
        if is_train:
            corr, _ = spearmanr(df['rate'], (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-9))
            is_drop = bool(abs(corr) > 0.95)
            artifacts['is_drop'] = is_drop
        else:
            is_drop = artifacts.get('is_drop', False)
        if is_drop:
            df = df.drop(columns=['rate'], errors='ignore')

    df['is_tcp'] = (df['proto'] == 'tcp').astype(int)
    df['is_ftp'] = df['service'].isin(['ftp', 'ftp-data']).astype(int)
    df['is_http'] = df['service'].isin(['http', 'ssl']).astype(int)

    # -----------------------------------------------------------------
    # ÁP DỤNG MÁY NGHIỀN ÉP KIỂU SỚM CHO CÁC CỘT NGUY HIỂM NHẤT
    # -----------------------------------------------------------------
    dangerous_cols = ['sport', 'dsport', 'res_bdy_len', 'sbytes', 'dbytes', 'spkts', 'dpkts']
    for col in dangerous_cols:
        if col in df.columns:
            df[col] = df[col].apply(force_to_numeric)

    for feature in LOG1P:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
            df[feature] = np.log1p(df[feature].astype(float))
            
    df['is_zero_dur'] = (pd.to_numeric(df['dur'], errors='coerce') == 0).astype(int)
    df['is_short'] = ((pd.to_numeric(df['dur'], errors='coerce') > 0) & (pd.to_numeric(df['dur'], errors='coerce') < 0.001)).astype(int)
    df['dur'] = np.log1p(pd.to_numeric(df['dur'], errors='coerce').fillna(0).astype(float))

    math_cols = ['sjit', 'djit', 'sinpkt', 'dinpkt', 'synack', 'tcprtt', 'ackdat', 'swin', 'dwin', 'ct_srv_src', 'ct_src_ltm', 'ct_dst_src_ltm']
    for col in math_cols:
        if col in df.columns:
             df[col] = df[col].apply(force_to_numeric)

    df['bytes_ratio'] = df['sbytes'] - df['dbytes']
    df['pkts_ratio'] = df['spkts'] - df['dpkts']
    df['bytes_per_pkt_src'] = df['sbytes'] - df['spkts']
    df['bytes_per_pkt_dst'] = df['dbytes'] - df['dpkts']
    
    if 'sjit' in df.columns and 'djit' in df.columns:
        df['jitter_ratio'] = df['sjit'] - df['djit']
    if 'sinpkt' in df.columns and 'dinpkt' in df.columns:
        df['interpacket_ratio'] = df['sinpkt'] - df['dinpkt']
    if 'synack' in df.columns and 'tcprtt' in df.columns:
        df['synack_ratio'] = df['synack'] / (df['tcprtt'] + 1e-9)
        df['ack_ratio'] = df['ackdat'] / (df['tcprtt'] + 1e-9)

    if 'swin' in df.columns and 'dwin' in df.columns:
        df['zero_win'] = ((df['swin'] == 0) | (df['dwin'] == 0)).astype(int)
        df['win_asymmetry'] = np.abs(df['swin'] - df['dwin']) / (df['swin'] + df['dwin'] + 1)

    if 'ct_srv_src' in df.columns and 'ct_src_ltm' in df.columns:
        df['srv_diversity'] = 1.0 - np.clip(df['ct_srv_src'] / (df['ct_src_ltm'] + 1e-9), 0.0, 1.0)
    if 'ct_dst_src_ltm' in df.columns and 'ct_src_ltm' in df.columns:
        df['dst_concentration'] = np.clip(df['ct_dst_src_ltm'] / (df['ct_src_ltm'] + 1e-9), 0.0, 1.0)

    keep_str_cols = ['proto', 'service', 'state', 'srcip', 'dstip']
    for col in keep_str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', 'unknown')

    for col in df.columns:
        if col not in keep_str_cols:
            df[col] = df[col].apply(force_to_numeric)
    
    for col in df.columns:
        if col in keep_str_cols:
            df[col] = df[col].fillna('unknown') 
        else:
            df[col] = df[col].fillna(0.0)         

    return df, y_label, y_cat, artifacts


def select(X_train: pd.DataFrame, y_label: pd.Series, y_cat: pd.Series) -> tuple:
    print("  [Tiến trình Select] Đang chuẩn hóa nhãn và biến phân loại...")
    
    y_label = y_label.astype(int)
    le_ycat = LabelEncoder()
    y_cat_enc = pd.Series(le_ycat.fit_transform(y_cat.astype(str)), index=y_cat.index)
    
    for col in ['proto', 'service', 'state']:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].fillna('unknown').astype(str))
            
    print("  [Tiến trình Select] Ép kiểu mảng Numpy sang Float...")
    
    # -----------------------------------------------------------------
    # MÁY NGHIỀN ÉP KIỂU TUYỆT ĐỐI BƯỚC CUỐI
    # -----------------------------------------------------------------
    for col in X_train.columns:
        X_train[col] = X_train[col].apply(force_to_numeric)
            
    X_train.fillna(0.0, inplace=True)
    X_train = X_train.astype(float)
    
    features = X_train.columns.tolist()
    flag_cnt = {f: 0 for f in features}

    def flag(feat):
        flag_cnt[feat] += 1

    low_var = [col for col in features if X_train[col].var() < 0.01]
    for col in low_var: flag(col)

    cont = [c for c in features if c not in FLAGS]
    corr_matrix = X_train[cont].corr(method='spearman').abs()
    high_corr = []
    for i, c1 in enumerate(cont):
        for j, c2 in enumerate(cont):
            if j <= i: continue
            if corr_matrix.loc[c1, c2] > 0.90:
                high_corr.append((c1, c2, corr_matrix.loc[c1, c2]))

    vif_features = [c for c in cont if X_train[c].nunique() > 2]
    vif_vals = {}
    vif_arr = X_train[vif_features].values
    
    sample_idx = np.random.choice(len(vif_arr), min(10000, len(vif_arr)), replace=False)
    vif_arr_sample = vif_arr[sample_idx]
    
    for i, col in enumerate(vif_features):
        vif_vals[col] = variance_inflation_factor(vif_arr_sample, i)
    high_vif = {k: v for k, v in vif_vals.items() if v > 10}

    mi_label = pd.Series(mutual_info_classif(X_train.iloc[:20000], y_label.iloc[:20000], random_state=42), index=features)
    mi_cat = pd.Series(mutual_info_classif(X_train.iloc[:20000], y_cat_enc.iloc[:20000], random_state=42), index=features)
    thresh_label = mi_label.quantile(0.1)
    thresh_cat = mi_cat.quantile(0.1)
    low_mi = set(mi_label[mi_label < thresh_label].index) & set(mi_cat[mi_cat < thresh_cat].index)
    for f in low_mi: flag(f)

    X_arr = X_train.values
    lgb_label = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, is_unbalance=True, verbose=-1)
    lgb_label.fit(X_arr, y_label)
    gain_label = pd.Series(lgb_label.booster_.feature_importance(importance_type='gain'), index=features)

    lgb_cat = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1)
    lgb_cat.fit(X_arr, y_cat_enc) 
    gain_cat = pd.Series(lgb_cat.booster_.feature_importance(importance_type='gain'), index=features)

    thresh_glabel = gain_label.quantile(0.05)
    thresh_gcat = gain_cat.quantile(0.05)
    low_gain = set(gain_label[gain_label < thresh_glabel].index) & set(gain_cat[gain_cat < thresh_gcat].index)
    for f in low_gain: flag(f)

    binary_scorer = make_scorer(recall_score, pos_label=1, zero_division=0)
    perm = permutation_importance(lgb_label, X_arr[:10000], y_label.iloc[:10000], scoring=binary_scorer, n_repeats=3, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=features)
    low_perm = set(perm_imp[perm_imp < 0.0001].index)
    for f in low_perm: flag(f)

    idx = np.random.default_rng(42).choice(len(X_arr), min(5000, len(X_arr)), replace=False)
    explainer = shap.TreeExplainer(lgb_cat)
    shap_vals = explainer.shap_values(X_arr[idx])
    if isinstance(shap_vals, list):
        shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    else:
        shap_mean = np.abs(shap_vals).mean(axis=(0, 2))
    shap_imp = pd.Series(shap_mean, index=features)
    low_shap = set(shap_imp[shap_imp < shap_imp.quantile(0.05)].index)
    for f in low_shap: flag(f)

    d = {f for f, c in flag_cnt.items() if c >= 2}
    for c1, c2, _ in high_corr:
        i_c1 = gain_label[c1] + gain_cat[c1]
        i_c2 = gain_label[c2] + gain_cat[c2]
        l = c1 if i_c1 < i_c2 else c2
        if flag_cnt[l] >= 1:
            d.add(l)
            flag(l)
    for feat in high_vif:
        if flag_cnt[feat] >= 1:
            d.add(feat)
            flag(feat)
            
    features_after = [f for f in features if f not in d]

    print("  [Tiến trình Select] Đang chạy RFECV Tầng 1 (Binary)...")
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=min(30000, len(X_arr)), random_state=42)
    idx1, _ = next(sss1.split(X_train[features_after], y_label))
    rfecv_label = RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, is_unbalance=True, verbose=-1),
        step=2,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring=binary_scorer,
        min_features_to_select=max(10, len(features_after) // 3),
        n_jobs=-1,
    )
    rfecv_label.fit(X_train[features_after].values[idx1], y_label.values[idx1])
    selected_label = [f for f, sel in zip(features_after, rfecv_label.support_) if sel]

    print("  [Tiến trình Select] Đang chạy RFECV Tầng 2 (Multi-class)...")
    CAT_RFECV_CANDIDATES = ['proto', 'state']
    CAT_ALWAYS_KEEP      = ['ct_state_ttl']
    features_after_cat = list(dict.fromkeys(features_after + [f for f in CAT_RFECV_CANDIDATES if f not in features_after]))
    
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=min(30000, len(X_arr)), random_state=42)
    idx2, _ = next(sss2.split(X_train[features_after_cat], y_cat_enc)) 
    rfecv_cat = RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1),
        step=2,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring=make_scorer(recall_score, average='macro', zero_division=0),
        min_features_to_select=max(10, len(features_after_cat) // 3),
        n_jobs=-1,
    )
    rfecv_cat.fit(X_train[features_after_cat].values[idx2], y_cat_enc.values[idx2]) 
    selected_cat = [f for f, sel in zip(features_after_cat, rfecv_cat.support_) if sel]

    for f in CAT_ALWAYS_KEEP:
        if f not in selected_cat and f in X_train.columns:
            selected_cat.append(f)

    return selected_label, selected_cat


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, smoothing: float, is_multiclass: bool):
        self.column = column
        self.smoothing = smoothing
        self.is_multiclass = is_multiclass
        self.encoding_maps_: dict = {}
        self.global_means_: dict = {}
        self.classes_: list = []

    def _fit_single(self, col: pd.Series, target: np.ndarray, key):
        global_mean = float(target.mean())
        self.global_means_[key] = global_mean
        tmp = pd.DataFrame({'val': col.values, 't': target})
        enc_map = {}
        for val, grp in tmp.groupby('val'):
            n = len(grp)
            cat_mean = float(grp['t'].mean())
            enc_map[val] = (n * cat_mean + self.smoothing * global_mean) / (n + self.smoothing)
        self.encoding_maps_[key] = enc_map

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.column not in X.columns: return self
        col = X[self.column]
        y_arr = np.array(y)
        if self.is_multiclass:
            self.classes_ = sorted(np.unique(y_arr).tolist())
            for cl in self.classes_:
                b = (y_arr == cl).astype(float)
                self._fit_single(col, b, key=cl)
        else:
            b = y_arr.astype(float)
            self._fit_single(col, b, key='binary')
            self.classes_ = ['binary']
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns: return pd.DataFrame(index=X.index)
        col = X[self.column]
        res = {}
        for key in self.classes_:
            enc_map = self.encoding_maps_[key]
            global_mean = self.global_means_[key]
            encoded = col.map(enc_map).fillna(global_mean)
            col_name = f"proto_enc_{key}" if self.is_multiclass else "proto_enc"
            res[col_name] = encoded.values
        return pd.DataFrame(res, index=X.index)


class Pipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.proto_enc_label_: SmoothedTargetEncoder | None = None
        self.proto_enc_cat_: SmoothedTargetEncoder | None = None
        self.ohe_: OneHotEncoder | None = None
        self.scaler_: RobustScaler | None = None
        self.ohe_cols_: list = []
        self.cont_cols_: list = []
        self.output_cols_: list = []

    def fit(self, X: pd.DataFrame, y_label: pd.Series, y_cat: pd.Series):
        if 'proto' in X.columns:
            self.proto_enc_label_ = SmoothedTargetEncoder('proto', smoothing=10, is_multiclass=False)
            self.proto_enc_label_.fit(X[['proto']], y_label)
            self.proto_enc_cat_ = SmoothedTargetEncoder('proto', smoothing=10, is_multiclass=True)
            self.proto_enc_cat_.fit(X[['proto']], y_cat)
            
        self.ohe_cols_ = [c for c in ['service', 'state'] if c in X.columns]
        if self.ohe_cols_:
            self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.ohe_.fit(X[self.ohe_cols_].astype(str))
            
        X_encoded = self._apply_encodings(X)
        ohe_names = (self.ohe_.get_feature_names_out(self.ohe_cols_).tolist() if self.ohe_ is not None else [])
        non_scale_cols = set(FLAGS) | set(ohe_names)
        self.cont_cols_ = [c for c in X_encoded.columns if c not in non_scale_cols]
        
        for col in self.cont_cols_:
            X_encoded[col] = X_encoded[col].apply(force_to_numeric)
        
        self.scaler_ = RobustScaler()
        if self.cont_cols_:
            self.scaler_.fit(X_encoded[self.cont_cols_])
            
        self.output_cols_ = X_encoded.columns.tolist()
        return self

    def _apply_encodings(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'proto' in X.columns and self.proto_enc_label_ is not None:
            proto_label = self.proto_enc_label_.transform(X[['proto']])
            proto_cat = self.proto_enc_cat_.transform(X[['proto']])
            X = pd.concat([X.drop(columns=['proto']), proto_label, proto_cat], axis=1)
            
        if self.ohe_cols_ and self.ohe_ is not None:
            ohe_names = self.ohe_.get_feature_names_out(self.ohe_cols_).tolist()
            ohe_df = pd.DataFrame(self.ohe_.transform(X[self.ohe_cols_].astype(str)), columns=ohe_names, index=X.index)
            X = pd.concat([X.drop(columns=self.ohe_cols_), ohe_df], axis=1)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = self._apply_encodings(X)
        
        if self.cont_cols_:
            for col in self.cont_cols_:
                X_encoded[col] = X_encoded[col].apply(force_to_numeric)
            X_encoded[self.cont_cols_] = self.scaler_.transform(X_encoded[self.cont_cols_])
        return X_encoded