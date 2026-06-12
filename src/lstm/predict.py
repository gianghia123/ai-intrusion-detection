# =====================================================================
# FILE: predict.py (Phiên bản nâng cấp - Đọc trực tiếp mảng 3D .npy)
# =====================================================================
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

import config

def main():
    # 1. Kiểm tra sự tồn tại của các file mảng 3D và mô hình đã huấn luyện
    required_files = [
        "transformers.pkl", 
        "X_test_3d.npy", 
        "y_test_bin_3d.npy", 
        "y_test_mul_3d.npy", 
        config.MODEL_TIER1_PATH, 
        config.MODEL_TIER2_PATH
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ LỖI: Thiếu các file bắt buộc sau: {missing_files}")
        return
        
    print("Đang nạp bộ mã hóa nhãn toàn cục từ transformers.pkl...")
    with open("transformers.pkl", "rb") as f:
        transformers = pickle.load(f)
        
    le_multi = transformers['le_multi']
    normal_idx = transformers['normal_idx']
    
    print("Đang nạp mảng chuỗi thời gian kiểm thử tinh hoa 3D (.npy)...")
    X_test = np.load("X_test_3d.npy")
    y_test_bin = np.load("y_test_bin_3d.npy")
    y_test_mul = np.load("y_test_mul_3d.npy")
    
    print("Đang nạp các mô hình mạng Deep Learning tuần tự h5/keras...")
    model_tier1 = tf.keras.models.load_model(config.MODEL_TIER1_PATH)
    model_tier2 = tf.keras.models.load_model(config.MODEL_TIER2_PATH)
    
    print("\n" + "="*50)
    print(" BƯỚC 3: ĐÁNH GIÁ PIPELINE KIẾN TRÚC 2 TẦNG (TESTING)")
    print("="*50)
    
    # 2. Tầng 1 thực hiện phán xét toàn bộ tập dữ liệu Test
    print("Tầng 1 (Người gác cổng) đang tiến hành rà soát lưu lượng luồng mạng...")
    preds_tier1 = model_tier1.predict(X_test, batch_size=config.BATCH_SIZE)
    y_pred_tier1 = (preds_tier1 > 0.5).astype(int).flatten()
    
    print("\n[BÁO CÁO TẦNG 1: NGƯỜI GÁC CỔNG (NORMAL VS ATTACK)]")
    print(classification_report(y_test_bin, y_pred_tier1, target_names=['Normal', 'Attack'], zero_division=0))
    
    # 3. Tầng 2 đón nhận và xử lý chuyên sâu các luồng bị nghi ngờ
    print("\nĐang chuyển các luồng nghi ngờ độc hại xuống Tầng 2 để điều tra chuyên sâu...")
    final_predictions = np.full(shape=(len(X_test),), fill_value=normal_idx)
    attack_indices_test = np.where(y_pred_tier1 == 1)[0]
    
    if len(attack_indices_test) > 0:
        X_test_suspects = X_test[attack_indices_test]
        preds_tier2 = model_tier2.predict(X_test_suspects, batch_size=config.BATCH_SIZE)
        y_pred_tier2 = np.argmax(preds_tier2, axis=1)
        # Điền nhãn phân loại mã độc chi tiết vào bảng kết quả chung
        final_predictions[attack_indices_test] = y_pred_tier2

    print("\n[BÁO CÁO PIPELINE CUỐI CÙNG: PHÂN LOẠI CỤ THỂ 10 LỚP]")
    target_names = le_multi.classes_ 
    labels_idx = np.arange(len(target_names)) 
    print(classification_report(y_test_mul, final_predictions, labels=labels_idx, target_names=target_names, zero_division=0))

    # 4. Đóng gói dữ liệu và xuất báo cáo kết quả chi tiết dạng bảng biểu CSV
    print("\nĐang tiến hành đóng gói dữ liệu và kết xuất file báo cáo CSV...")
    final_labels_text = le_multi.inverse_transform(final_predictions)
    true_labels_text = le_multi.inverse_transform(y_test_mul)
    
    results_df = pd.DataFrame({
        'Nhan_Thuc_Te': true_labels_text,
        'Du_Doan_Tang_1': ['Attack' if x == 1 else 'Normal' for x in y_pred_tier1],
        'Du_Doan_Cuoi_Cung': final_labels_text,
        'Du_Doan_Dung': true_labels_text == final_labels_text 
    })
    
    results_df.to_csv(config.OUTPUT_REPORT_PATH, index=False, encoding='utf-8')
    print(f"✅ XUẤT FILE KẾT QUẢ BÁO CÁO THÀNH CÔNG TẠI: {os.path.abspath(config.OUTPUT_REPORT_PATH)}")

if __name__ == "__main__":
    main()