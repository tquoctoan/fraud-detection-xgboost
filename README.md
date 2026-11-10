# Fraud Detection with XGBoost

Dự án phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection) sử dụng mô hình **XGBoost**, tối ưu siêu tham số bằng **Optuna**, xử lý dữ liệu mất cân bằng (imbalanced data) và giải thích mô hình bằng **SHAP**.

---

## 📁 Cấu trúc thư mục

```
fraud-detection-xgboost/
├── README.md                    # Tài liệu hướng dẫn dự án
├── requirements.txt             # Thư viện phụ thuộc
├── .gitignore                   # Cấu hình bỏ qua file nhạy cảm/dữ liệu nặng
│
├── data/
│   ├── raw/                     # Dữ liệu gốc (creditcard.csv)
│   └── processed/               # train.parquet, test.parquet sau tiền xử lý
│
├── src/                         # Module hàm dùng chung
│   ├── __init__.py
│   ├── config.py                # Cấu hình SEED, đường dẫn, tham số
│   ├── data.py                  # Hàm đọc, làm sạch, chia và lưu dữ liệu
│   ├── pipeline.py              # Xây dựng Pipeline tiền xử lý & mô hình
│   ├── tune.py                  # Tối ưu siêu tham số Optuna theo PR-AUC
│   ├── evaluate.py              # Đánh giá chỉ số, PR curve, chọn ngưỡng tối ưu
│   ├── compare.py               # So sánh các cách xử lý mất cân bằng
│   └── explain.py               # Giải thích mô hình bằng SHAP
│
├── scripts/                     # Scripts chạy theo thứ tự pipeline
│   ├── 01_preprocess.py         # Tiền xử lý dữ liệu
│   ├── 02_eda_overview.py       # EDA tổng quan (phân bố nhãn, Amount, Time)
│   ├── 03_eda_features.py       # EDA chi tiết các biến V1-V28
│   ├── 04_baseline_xgb.py       # Huấn luyện mô hình cơ sở XGBoost
│   ├── 05_imbalance_compare.py  # So sánh các phương pháp xử lý mất cân bằng
│   ├── 06_tune_xgb.py           # Tinh chỉnh siêu tham số
│   ├── 07_evaluate.py           # Đánh giá mô hình cuối & chọn ngưỡng
│   ├── 08_explain.py            # Giải thích mô hình với SHAP
│   └── run_all.py               # Script chạy toàn bộ quy trình từ 01 đến 08
│
├── results/                     # Kết quả đầu ra (metrics, params, predictions, models)
├── figures/                     # Đồ thị, biểu đồ xuất ra phục vụ báo cáo
├── notebooks/                   # Jupyter Notebook báo cáo tổng kết
├── report/                      # Nội dung các chương báo cáo
└── scratch/                     # Thử nghiệm cá nhân, không commit
```

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Cài đặt môi trường

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt các thư viện
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Chạy lệnh sau từ thư mục gốc để tải dữ liệu (tự bỏ qua nếu CSV đã có):

```bash
python -c "from src.data import fetch_raw; fetch_raw()"
```

`fetch_raw()` dùng `sklearn.datasets.fetch_openml` với `target_column="Time"`. Mặc định sklearn bỏ `Time` vì OpenML đánh dấu đây là row identifier; đặt `Time` làm target thì `frame` trả về đủ 31 cột. Hàm ép `Class` về `int` rồi lưu CSV.

Kết quả đúng: 284.807 dòng × 31 cột, 492 gian lận, 1.081 dòng trùng; sau làm sạch còn 283.726 dòng / 473 gian lận.

### 3. Thực thi

Chạy toàn bộ pipeline tự động:
```bash
python scripts/run_all.py
```
Hoặc chạy từng bước độc lập từ thư mục gốc của dự án:
```bash
python scripts/01_preprocess.py
python scripts/02_eda_overview.py
python scripts/03_eda_features.py
python scripts/04_baseline_xgb.py
python scripts/05_imbalance_compare.py
python scripts/06_tune_xgb.py
python scripts/07_evaluate.py
python scripts/08_explain.py
```
