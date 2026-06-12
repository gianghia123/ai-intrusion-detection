HỆ THỐNG PHÁT HIỆN XÂM NHẬP (IDS) KIẾN TRÚC 2 TẦNG SỬ DỤNG MẠNG STACKED LSTM
Dataset: UNSW-NB15

--- CẤU TRÚC THƯ MỤC DỰ ÁN ---
1. config.py     : Chứa toàn bộ tham số siêu cấu hình (Batch size, Epochs, Cửa sổ trượt, Đường dẫn file).
2. preprocess.py : Chứa các hàm Feature Engineering tối ưu, kiểm tra Variance và tạo mảng Sliding Window 3D.
3. models.py     : Định nghĩa độc lập 2 mô hình Sequential Keras mạng Stacked LSTM (Tầng 1 & Tầng 2).
4. train.py      : Thực hiện Pipeline huấn luyện tuần tự. Áp dụng SMOTE cục bộ cách ly tại Tầng 2.
5. predict.py    : Thực hiện Blind Test đánh giá mô hình Pipeline thực tế và xuất file báo cáo kết quả CSV.

--- HƯỚNG DẪN CHẠY HỆ THỐNG ---
Bước 1: Mở file `config.py` và cập nhật đường dẫn thư mục chứa 4 file CSV dữ liệu thô (DATA_DIR).
Bước 2: Chạy script huấn luyện hệ thống bằng lệnh:
        python train.py
Bước 3: Sau khi quá trình huấn luyện hoàn tất (sinh ra các file .keras và transformers.pkl), chạy kiểm thử:
        python predict.py
Bước 4: Mở file `Ket_Qua_Du_Doan_IDS.csv` vừa sinh ra để xem chi tiết kết quả phân tích từng luồng mạng.
