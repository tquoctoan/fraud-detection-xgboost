# Hướng dẫn chạy dự án Fraud Detection with XGBoost

> ⚠️ **Lưu ý:** Luôn kéo code mới nhất về trước khi chạy: `git pull origin main`

---

## 1. Cài đặt môi trường

Mở terminal tại thư mục gốc của dự án (`fraud-detection-xgboost`):

```bash
# 1. Tạo môi trường ảo (chỉ cần làm lần đầu)
python -m venv .venv

# 2. Kích hoạt môi trường ảo:
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Trên Linux / macOS:
source .venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 2. Cách chạy mã nguồn

Luôn chạy lệnh từ **thư mục gốc** của dự án:

### Cách 1: Chạy tự động toàn bộ quy trình
```bash
python scripts/run_all.py
```
*(Tùy chọn: có thể dùng `--skip 05 06` để bỏ qua các bước chạy lâu nếu đã có kết quả).*

---

### Cách 2: Chạy tuần tự từng bước theo thứ tự

| Bước | Lệnh chạy | Mô tả kết quả |
|---|---|---|
| **01** | `python scripts/01_preprocess.py` | Tải dữ liệu, lọc trùng lặp, chia tập train/test (80/20) và lưu vào `data/processed/` |
| **02** | `python scripts/02_eda_overview.py` | Phân tích tổng quan (phân bố nhãn, Amount, Time), xuất ảnh vào `figures/` |
| **03** | `python scripts/03_eda_features.py` | Phân tích chi tiết các biến ẩn danh V1–V28 và ma trận tương quan |
| **04** | `python scripts/04_baseline_xgb.py` | Huấn luyện mô hình XGBoost cơ sở (chưa xử lý mất cân bằng) |
| **05** | `python scripts/05_imbalance_compare.py` | So sánh 4 phương pháp cân bằng: `none`, `weight`, `smote`, `under` |
| **06** | `python scripts/06_tune_xgb.py` | Tinh chỉnh siêu tham số XGBoost bằng Optuna (tìm kiếm theo PR-AUC) |
| **07** | `python scripts/07_evaluate.py` | Huấn luyện mô hình tốt nhất, chọn ngưỡng tối ưu và đánh giá tập Test |
| **08** | `python scripts/08_explain.py` | Giải thích mô hình bằng SHAP và phân tích các trường hợp dự đoán lỗi |
