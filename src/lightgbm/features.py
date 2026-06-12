import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD


def preprocess(df, freq_map):
    y = df["attack_cat"]
    X = df.drop(columns=["id", "label", "attack_cat"], errors="ignore").copy()

    X["bytes_ratio"] = X["sbytes"] / (X["dbytes"] + 1)
    X["pkts_ratio"] = X["spkts"] / (X["dpkts"] + 1)
    X["byte_rate"] = (X["sbytes"] + X["dbytes"]) / (X["dur"] + 1)
    X["pkt_rate"] = (X["spkts"] + X["dpkts"]) / (X["dur"] + 1)

    if "swin" in X.columns and "dwin" in X.columns:
        X["win_ratio"] = X["swin"] / (X["dwin"] + 1)

    X["src_load"] = X["sbytes"] / (X["dur"] + 1e-6)
    X["dst_load"] = X["dbytes"] / (X["dur"] + 1e-6)
    X["load_asym"] = np.abs(X["src_load"] - X["dst_load"]) / (X["src_load"] + X["dst_load"] + 1)
    X["bpp_src"] = X["sbytes"] / (X["spkts"] + 1)
    X["bpp_dst"] = X["dbytes"] / (X["dpkts"] + 1)
    X["pkt_size_diff"] = np.abs(X["bpp_src"] - X["bpp_dst"])

    if "sjit" in X.columns and "djit" in X.columns:
        X["jit_ratio"] = X["sjit"] / (X["djit"] + 1)
        X["jit_sum"] = X["sjit"] + X["djit"]

    if "sloss" in X.columns and "dloss" in X.columns:
        X["loss_ratio"] = X["sloss"] / (X["dloss"] + 1)
        X["loss_sum"] = X["sloss"] + X["dloss"]

    if "synack" in X.columns and "ackdat" in X.columns:
        X["synack_ratio"] = X["synack"] / (X["ackdat"] + 1e-6)

    log_cols = [
        "dur", "sbytes", "dbytes", "spkts", "dpkts", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit", "response_body_len",
        "src_load", "dst_load", "bpp_src", "bpp_dst"
    ]

    for c in log_cols:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c], 0, None))

    X["is_tcp"] = (X["proto"] == "tcp").astype(np.int8)
    X["is_udp"] = (X["proto"] == "udp").astype(np.int8)
    X["is_http"] = X["service"].isin(["http", "https", "ssl"]).astype(np.int8)
    X["is_dns"] = (X["service"] == "dns").astype(np.int8)
    X["is_ftp"] = X["service"].isin(["ftp", "ftp-data"]).astype(np.int8)

    if "stcpb" in X.columns and "dtcpb" in X.columns:
        X["tcp_established"] = ((X["stcpb"] != 0) & (X["dtcpb"] != 0)).astype(np.int8)
        X["tcp_one_side"] = ((X["stcpb"] == 0) ^ (X["dtcpb"] == 0)).astype(np.int8)
        X.drop(columns=["stcpb", "dtcpb"], inplace=True)

    for col in ["proto", "service", "state"]:
        if col in X.columns:
            X[col] = X[col].map(freq_map[col]).fillna(0)

    X = X.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)

    return X, y


def scale_features(X_train, X_test):
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return X_train_scaled, X_test_scaled, scaler


def build_svd_features(X_train_scaled, X_test_scaled, n_components=5, random_state=42):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    X_train_svd = svd.fit_transform(X_train_scaled).astype(np.float32)
    X_test_svd = svd.transform(X_test_scaled).astype(np.float32)

    return X_train_svd, X_test_svd, svd