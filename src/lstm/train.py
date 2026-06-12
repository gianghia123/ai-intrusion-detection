# =====================================================================
# FILE: train.py (Hoặc main.py - Phiên bản tích hợp Chọn lọc Đặc trưng nâng cao)
# =====================================================================
import os
import gc
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import config
# Import các hàm nâng cao bạn vừa chuyển sang file preprocess.py
from preprocess import process, select, Pipeline

# =====================================================================
# 1. HÀM TẠO SEQUENCE (CỬA SỔ TRƯỢT 3D CHO LSTM)
# =====================================================================
def create_sequences(df, feature_cols, le_multi, window_size):
    print(f"--> Đang nhóm luồng mạng (Groupby) và trượt cửa sổ cỡ {window_size}...")
    X_seq, y_bin_seq, y_mul_seq = [], [], []
    
    # Gom nhóm theo cặp IP nguồn và IP đích để đảm bảo tính liên tục của chuỗi thời gian
    grouped = df.groupby(['srcip', 'dstip'])
    
    for _, group in grouped:
        if len(group) >= window_size:
            vals = group[feature_cols].values
            lb_bin = group['label'].values
            lb_mul = le_multi.transform(group['attack_cat'].values)
            
            # Trượt cửa sổ dọc theo các gói tin trong nhóm
            for i in range(len(group) - window_size + 1):
                X_seq.append(vals[i:i + window_size])
                # Nhãn của chuỗi sẽ là nhãn của gói tin CUỐI CÙNG trong cửa sổ đó
                y_bin_seq.append(lb_bin[i + window_size - 1])
                y_mul_seq.append(lb_mul[i + window_size - 1])
                
    return (np.array(X_seq, dtype=np.float32), 
            np.array(y_bin_seq, dtype=np.float32), 
            np.array(y_mul_seq, dtype=np.int32))

# =====================================================================
# 2. LUỒNG CHẠY HUẤN LUYỆN CHÍNH (MAIN PIPELINE)
# =====================================================================
def main():
    print("Đang tải dữ liệu gốc (File 1, 2, 3, 4)...")
    df_list = []
    for i in range(1, 5):
        file_path = os.path.join(config.DATA_DIR, f"UNSW-NB15_{i}.csv")
        if os.path.exists(file_path):
            print(f"  [READ] Đang nạp: {file_path}")
            df_list.append(pd.read_csv(file_path, header=None, names=config.UNSW_COLUMNS, low_memory=False))
        else:
            print(f"  [CẢNH BÁO] Không tìm thấy file {file_path}. Đang bỏ qua...")
            
    if not df_list:
        print("LỖI KHẨN CẤP: Không tìm thấy dữ liệu thô tại thư mục chỉ định trong config.py")
        return
        
    df = pd.concat(df_list)
    
    print("Đang sắp xếp toàn bộ dữ liệu theo trục thời gian...")
    df.sort_values(by='stime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"\n--- PHÂN CHIA DỮ LIỆU ({config.TRAIN_RATIO*100}% TRAIN / {(1-config.TRAIN_RATIO)*100}% TEST) ---")
    split_idx = int(len(df) * config.TRAIN_RATIO)
    df_train_raw = df.iloc[:split_idx].copy()
    df_test_raw = df.iloc[split_idx:].copy()
    del df; gc.collect() 

    # -----------------------------------------------------------------
    # BƯỚC A: CHẠY HÀM PROCESS() MỚI (Trích xuất nhãn và Feature Engineering)
    # -----------------------------------------------------------------
    print("\n[GIAI ĐOẠN 1] Áp dụng Feature Engineering nâng cao...")
    artifacts = {}
    X_train_raw, y_train_bin, y_train_cat, artifacts = process(df_train_raw, is_train=True, artifacts=artifacts)
    X_test_raw, y_test_bin, y_test_cat, _ = process(df_test_raw, is_train=False, artifacts=artifacts)
    del df_train_raw, df_test_raw; gc.collect()

    # -----------------------------------------------------------------
    # BƯỚC B: TÁCH BIỆT METADATA TRƯỚC KHI CHỌN LỌC ĐẶC TRƯNG
    # -----------------------------------------------------------------
    meta_cols = ['srcip', 'dstip', 'stime', 'ltime']
    
    # Cất giấu các cột chuỗi chữ/thời gian để không làm hàm select() bị crash
    df_meta_train = X_train_raw[meta_cols].copy()
    df_meta_test = X_test_raw[meta_cols].copy()
    
    X_train_for_select = X_train_raw.drop(columns=meta_cols, errors='ignore')

    # -----------------------------------------------------------------
    # BƯỚC C: CHẠY HÀM SELECT() NÂNG CAO (MI, VIF, LightGBM, RFECV...)
    # -----------------------------------------------------------------
    print("\n[GIAI ĐOẠN 2] Đang kích hoạt bộ lọc chọn đặc trưng tinh hoa (RFECV, SHAP, VIF)...")
    selected_label, selected_cat = select(X_train_for_select, y_train_bin, y_train_cat)
    
    # Lấy hợp tập (Union) của các đặc trưng tinh hoa nhất được chọn cho cả 2 tầng
    union_features = list(dict.fromkeys(selected_label + selected_cat))
    print(f"-> Số lượng đặc trưng tinh hoa được giữ lại: {len(union_features)} cột.")

    # -----------------------------------------------------------------
    # BƯỚC D: HUẤN LUYỆN PIPELINE (Target Encoding cho proto/service/state & RobustScale)
    # -----------------------------------------------------------------
    print("\n[GIAI ĐOẠN 3] Đang mã hóa cấu trúc mã nguồn Pipeline...")
    pl = Pipeline()
    pl.fit(X_train_raw[union_features], y_train_bin, y_train_cat)
    
    X_train_encoded = pl.transform(X_train_raw[union_features])
    X_test_encoded = pl.transform(X_test_raw[union_features])
    del X_train_raw, X_test_raw; gc.collect()

    # -----------------------------------------------------------------
    # BƯỚC E: KHÔI PHỤC IP ĐỂ CHẠY CỬA SỔ TRƯỢT (SLIDING WINDOWS)
    # -----------------------------------------------------------------
    print("\n[GIAI ĐOẠN 4] Khôi phục cấu trúc dòng chảy mạng để trượt cửa sổ thời gian...")
    
    # Đồng bộ Label Encoder cho mục tiêu Multi-class
    le_multi = LabelEncoder()
    le_multi.fit(y_train_cat.unique().tolist() + y_test_cat.unique().tolist())
    
    # Khôi phục IP và Nhãn vào DataFrame để Groupby
    X_train_encoded['srcip'] = df_meta_train['srcip'].values
    X_train_encoded['dstip'] = df_meta_train['dstip'].values
    X_train_encoded['label'] = y_train_bin.values
    X_train_encoded['attack_cat'] = y_train_cat.values

    X_test_encoded['srcip'] = df_meta_test['srcip'].values
    X_test_encoded['dstip'] = df_meta_test['dstip'].values
    X_test_encoded['label'] = y_test_bin.values
    X_test_encoded['attack_cat'] = y_test_cat.values

    # Xác định danh sách các cột số thực tế sẽ đưa vào LSTM
    final_feature_cols = [c for c in X_train_encoded.columns if c not in ['srcip', 'dstip', 'label', 'attack_cat']]

    print("Đang tạo chuỗi dữ liệu 3D cho tập Train và Test...")
    X_train, y_train_bin_3d, y_train_mul_3d = create_sequences(X_train_encoded, final_feature_cols, le_multi, config.WINDOW_SIZE)
    X_test, y_test_bin_3d, y_test_mul_3d = create_sequences(X_test_encoded, final_feature_cols, le_multi, config.WINDOW_SIZE)
    
    # Lưu đối tượng lưu trữ thông tin kiểm thử
    with open("transformers.pkl", "wb") as f:
        pickle.dump({
            'le_multi': le_multi,
            'feature_cols': final_feature_cols,
            'normal_idx': le_multi.transform(['Normal'])[0]
        }, f)
        
    # Lưu tập test 3D xuống ổ đĩa để predict.py gọi lại
    np.save("X_test_3d.npy", X_test)
    np.save("y_test_bin_3d.npy", y_test_bin_3d)
    np.save("y_test_mul_3d.npy", y_test_mul_3d)
    
    del X_test, X_test_encoded, df_meta_train, df_meta_test; gc.collect()

    # =================================================================
    # HUẤN LUYỆN TẦNG 1: BINARY MODEL (DỮ LIỆU ĐÃ LỌC TINH HOA)
    # =================================================================
    print("\n" + "="*50)
    print(" BƯỚC 1: HUẤN LUYỆN TẦNG 1 (BINARY) TRÊN ĐỮ LIỆU GỐC TINH HOA")
    print("="*50)
    from models import build_tier1_binary_model, build_tier2_multiclass_model
    
    model_tier1 = build_tier1_binary_model(config.WINDOW_SIZE, len(final_feature_cols))
    model_tier1.fit(X_train, y_train_bin_3d, validation_split=0.1, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    model_tier1.save(config.MODEL_TIER1_PATH)

    # =================================================================
    # HUẤN LUYỆN TẦNG 2: MULTI-CLASS MODEL + SMOTE CÁCH LY
    # =================================================================
    print("\n" + "="*50)
    print(" BƯỚC 2: CHUẨN BỊ & HUẤN LUYỆN TẦNG 2 (MULTI-CLASS)")
    print("="*50)
    
    attack_indices_train = np.where(y_train_bin_3d == 1)[0]
    X_train_attacks = X_train[attack_indices_train]
    y_train_mul_attacks = y_train_mul_3d[attack_indices_train]
    del X_train, y_train_bin_3d; gc.collect()
    
    print("Đang kích hoạt SMOTE cân bằng nội bộ các nhóm tấn công...")
    samples, window, features = X_train_attacks.shape
    X_flat_attacks = X_train_attacks.reshape((samples, window * features)).astype(np.float32)
    
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_res_flat_attacks, y_res_mul_attacks = smote.fit_resample(X_flat_attacks, y_train_mul_attacks)
    X_train_res_attacks = X_res_flat_attacks.reshape((X_res_flat_attacks.shape[0], window, features))
    
    print(f"-> Kích thước tập dữ liệu Tầng 2 sau khi tăng cường SMOTE: {X_train_res_attacks.shape[0]} mẫu.")
    del X_train_attacks, X_flat_attacks; gc.collect()

    model_tier2 = build_tier2_multiclass_model(config.WINDOW_SIZE, len(final_feature_cols), len(le_multi.classes_))
    model_tier2.fit(X_train_res_attacks, y_res_mul_attacks, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=0.1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    model_tier2.save(config.MODEL_TIER2_PATH)
    
    print("\n🎉 [THÀNH CÔNG] Đã hoàn tất huấn luyện hệ thống IDS kết hợp Chọn lọc Đặc trưng nâng cao!")

if __name__ == "__main__":
    main()