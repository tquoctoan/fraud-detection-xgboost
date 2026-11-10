# 📅 KẾ HOẠCH CHI TIẾT 3 TUẦN (2 TUẦN CHÍNH + 1 TUẦN BONUS)
### Dự án: Phát hiện gian lận thẻ tín dụng với XGBoost (P1 — P5)

> **Mô hình làm việc:** 5 thành viên làm việc song song theo lớp (Layered Workflow).  
> **Tổng thời gian:** 21 ngày (Tuần 1: Xây dựng nền móng & Baseline; Tuần 2: Tối ưu, Đánh giá & Báo cáo; Tuần 3: Mở rộng Bonus, Slide & Diễn tập).

---

## 🧭 Ma trận phân công trách nhiệm (RACI)

| Hạng mục công việc | P1 (Data) | P2 (Core/CV) | P3 (Imbalance) | P4 (Tuning) | P5 (Explain/Lead) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dữ liệu & Tiền xử lý** | **Chính (R)** | Hỗ trợ | Phối hợp | Phối hợp | EDA đặc trưng |
| **Kiến trúc Pipeline & CV** | Phối hợp | **Chính (R)** | Kế thừa | Kế thừa | Kế thừa |
| **So sánh mất cân bằng** | — | Hỗ trợ | **Chính (R)** | Kế thừa | — |
| **Tối ưu siêu tham số Optuna** | — | Hỗ trợ | Phối hợp | **Chính (R)** | — |
| **Đánh giá Test & Chọn ngưỡng** | — | **Chính (R)** | — | Phối hợp | Phân tích lỗi |
| **SHAP & Giải thích mô hình** | — | Hỗ trợ | — | — | **Chính (R)** |
| **Script tích hợp `run_all.py`** | — | — | Hỗ trợ | — | **Chính (R)** |
| **Mở rộng Bonus (Tuần 3)** | Drift Check | Model Compare | Cost Matrix | Cost Matrix | Dashboard Notebook |
| **Slide thuyết trình & Tổng duyệt** | Hỗ trợ | Hỗ trợ | **Chính Slide** | Hỗ trợ | **Chính Slide** |

*(Ghi chú: **R** = Responsible - Người chịu trách nhiệm chính thực hiện).*

---

## 🗓️ TUẦN 1: NỀN MÓNG DỮ LIỆU, PIPELINE & MÔ HÌNH CƠ SỞ
> **Mục tiêu tuần 1:** Dữ liệu sạch 100%, khung Pipeline & Cross-Validation hoạt động, hoàn thành mô hình Baseline và xác định phương pháp xử lý mất cân bằng tốt nhất.

### Ngày 1 (Thứ Tư, 11/11): Đặt nền móng & Bản khung (Skeleton Day)
* **08:30 – 11:30**:
  * **P2**: Hoàn thiện `src/config.py` (khai báo `SEED = 42`, đường dẫn `RESULTS`, `FIGURES`, danh sách 30 cột đặc trưng).
  * **P1**: Viết `src/data.py` gồm hàm `make_fake_data()` và nạp cờ `FRAUD_FAKE=1` để sinh dữ liệu giả lập chuẩn schema.
  * **P2**: Đẩy khung hàm `build_pipeline()` và `run_cv()` chạy thử được trong `src/pipeline.py` & `src/evaluate.py`.
* **13:30 – 17:30**:
  * **P3, P4, P5**: Kích hoạt biến môi trường `$env:FRAUD_FAKE=1`, tiến hành viết code module của mình độc lập mà không cần chờ dữ liệu thật.

### Ngày 2 (Thứ Năm, 12/11): Dữ liệu sạch & Chia tập huấn luyện (P1 & P5)
* **09:00 – 15:00**:
  * **P1**: Tải dữ liệu OpenML ID 1597 (đủ 31 cột có `Time`), xử lý ép kiểu `int` cho `Class`.
  * **P1**: Thực hiện xóa đúng 1.081 dòng trùng lặp trên 31 cột (còn lại 283.726 dòng / 473 fraud).
  * **P1**: Chia Stratified 80/20, xuất `train.parquet` và `test.parquet`.
* **17:00 [MILESTONE IP1 - DỮ LIỆU THẬT SẴN SÀNG]**:
  * **P1**: Xuất file `results/metrics/preprocess_summary.csv`. Cả nhóm chính thức bỏ cờ `FRAUD_FAKE` để chuyển sang dữ liệu thật.
  * **P5**: Bắt đầu nhận dữ liệu sạch để code `scripts/03_eda_features.py`.

### Ngày 3 (Thứ Sáu, 13/11): EDA Tổng quan & Chạy Baseline (P1 & P2)
* **09:00 – 17:00**:
  * **P1**: Chạy `scripts/02_eda_overview.py`, xuất các hình phân bố nhãn, Amount thang log, Time theo lớp vào `figures/02_*.png`. Viết nháp **Chương 4**.
  * **P2**: Chạy `scripts/04_baseline_xgb.py` (XGBoost mặc định, `imbalance="none"`, 5-Fold CV).
  * **P2**: Xuất `results/metrics/baseline.csv` và `results/predictions/xgb_baseline_oof.csv`. Viết nháp **Chương 3** & **Mục 6.1 – 6.2**.

### Ngày 4 (Thứ Bảy, 14/11): Thực nghiệm xử lý mất cân bằng (P3)
* **09:00 – 18:00**:
  * **P3**: Hoàn thiện `src/compare.py` và chạy `scripts/05_imbalance_compare.py`.
  * **P3**: So sánh 4 biến thể trên tập train (5-Fold CV): `none`, `weight`, `smote`, `under`.
  * **P3**: Thu thập chỉ số PR-AUC, F1, Recall và thời gian chạy huấn luyện.
* **21:00 [MILESTONE IP2 - CHỐT PHƯƠNG PHÁP CÂN BẰNG]**:
  * **P3**: Xuất file `results/params/chosen_imbalance.json` (ghi rõ lý do chọn biến thể tốt nhất) và vẽ `figures/05_pr_curves.png`. Viết nháp **Mục 6.3**.

### Ngày 5 – 6 (Chủ Nhật, 15/11 – Thứ Hai, 16/11): Chuẩn bị Tuning & EDA Đặc trưng (P4 & P5)
* **P4**: Nhận `chosen_imbalance.json`, hoàn thiện `src/tune.py` với Optuna `TPESampler`. Chạy thử nghiệm 5 trials để kiểm tra tính ổn định.
* **P5**: Hoàn thành `scripts/03_eda_features.py`, xuất biểu đồ ma trận tương quan Heatmap và Top biến tương quan cao nhất với gian lận (`figures/03_*.png`). Viết nháp **Mục 5.4 – 5.6**.

### Ngày 7 (Thứ Ba, 17/11): Họp Review Tuần 1 & Nghiệm thu Sprint 1
* **20:00 – 21:30 [HỌP ONLINE TEAM]**:
  * Rà soát tiến độ: Kiểm tra kết quả các script `01`, `02`, `03`, `04`, `05`.
  * Đo đạc thời gian 1 vòng tuning của P4 để lên lịch chạy chính thức qua đêm.

---

## 🗓️ TUẦN 2: TỐI ƯU HÓA, SHAP, ĐÁNH GIÁ TẬP TEST & BÁO CÁO CHÍNH
> **Mục tiêu tuần 2:** Tìm được bộ siêu tham số tốt nhất, hoàn thành đánh giá mô hình cuối trên tập test, phân tích lỗi bằng SHAP, hoàn thiện báo cáo phiên bản 1.0.

### Ngày 8 – 9 (Thứ Tư, 18/11 – Thứ Năm, 19/11): Tuning Optuna chính thức (P4)
* **P4**: Thực thi `scripts/06_tune_xgb.py` với 50 trials trên 5-Fold CV (tối ưu hóa chỉ số PR-AUC trung bình).
* **18:00 (19/11) [MILESTONE IP3 - SIÊU THAM SỐ TỐI ƯU]**:
  * **P4**: Xuất `results/params/best_params.json`, bảng lịch sử `tuning_trials.csv` và biểu đồ `figures/06_optimization_history.png`.
  * **P4**: Hoàn thiện nội dung viết **Mục 6.4**.

### Ngày 10 (Thứ Sáu, 20/11): Huấn luyện mô hình cuối & Đánh giá Test (P2)
* **09:00 – 17:00**:
  * **P2**: Nạp `best_params.json`, huấn luyện mô hình XGBoost cuối cùng trên toàn bộ tập Train.
  * **P2**: Tìm ngưỡng phân loại tối ưu (Threshold Tuning) trên dự đoán Out-of-fold.
  * **P2**: **Mở tập Test đánh giá duy nhất 1 lần** thông qua `scripts/07_evaluate.py`.
* **21:00 [MILESTONE IP4 - MÔ HÌNH CUỐI & ĐÁNH GIÁ TEST]**:
  * **P2**: Lưu model `results/models/xgb_final.joblib`, ngưỡng `threshold.json`, chỉ số `final_test.csv` và biểu đồ Confusion Matrix (`07_confusion_matrix.png`).
  * **P2**: Hoàn thiện nội dung viết **Mục 6.5 – 6.6**.

### Ngày 11 – 12 (Thứ Bảy, 21/11 – Chủ Nhật, 22/11): SHAP & Phân tích lỗi (P5)
* **P5**: Nạp `xgb_final.joblib` và `threshold.json`, thực thi `scripts/08_explain.py`.
* **P5**: Dùng `shap.TreeExplainer` trên mẫu con 5.000 giao dịch (giữ 100% ca gian lận), xuất biểu đồ Beeswarm `figures/08_shap_summary.png` và Bar plot `08_shap_bar.png`.
* **P5**: Phân tích chi tiết các ca bỏ sót (False Negatives) và báo động nhầm (False Positives), xuất bảng `results/metrics/errors.csv`. Viết **Mục 6.7**.

### Ngày 13 (Thứ Hai, 23/11): Tích hợp toàn diện `run_all.py` (P5 & Cả nhóm)
* **P5**: Hoàn thiện script điều phối `scripts/run_all.py` (chạy tự động từ `01` đến `08`, dừng ngay khi có lỗi, hỗ trợ cờ `--skip`).
* **Cả nhóm**: Chạy kiểm thử toàn bộ quy trình từ đầu đến cuối trên máy độc lập để đảm bảo tính tái lập (reproducibility).

### Ngày 14 (Thứ Ba, 24/11): Ghép Báo cáo v1.0 & Nghiệm thu Sprint 2
* **Cả nhóm**: Ghép toàn bộ nội dung từ `report/ch1_...` đến `ch6_...`, viết lời mở đầu và kết luận.
* Xuất file báo cáo sơ bộ v1.0, sẵn sàng bước vào tuần nâng cao Bonus.

---

## 🗓️ TUẦN 3 (TUẦN BONUS): MỞ RỘNG NÂNG CAO, PHÂN TÍCH KINH TẾ & DIỄN TẬP BẢO VỆ
> **Mục tiêu tuần 3:** Tạo điểm nhấn khác biệt cho đồ án (đạt điểm xuất sắc), hoàn thiện Slide thuyết trình chuyên nghiệp và diễn tập phản biện.

### Ngày 15 – 16 (Thứ Tư, 25/11 – Thứ Năm, 26/11): Triển khai các Module Bonus (`scripts/bonus_*.py`)
Cả nhóm phối hợp thực hiện 3 phân tích nâng cao:
1. **Bonus 1: Phân tích ma trận chi phí tài chính (`scripts/bonus_cost_matrix.py`) [P3 & P4]**:
   * Xây dựng hàm tổn thất kinh tế:
     $$\text{Chi phí} = \text{FN} \times \text{Giá trị giao dịch mất trắng} + \text{FP} \times \text{Chi phí CSKH/Khóa thẻ nhầm (\$10)}$$
   * Tìm ngưỡng phân loại giúp ngân hàng **giảm thiểu thiệt hại tài chính tối đa** thay vì chỉ dựa vào F1-score toán học.
2. **Bonus 2: Benchmark đối sánh đa mô hình (`scripts/bonus_benchmark.py`) [P2 & P4]**:
   * Huấn luyện đối sánh nhanh giữa XGBoost với **LightGBM** và **Random Forest** trên cùng tập train/test để chứng minh XGBoost là sự lựa chọn tối ưu.
3. **Bonus 3: Kiểm định trôi dạt dữ liệu (`scripts/bonus_drift_check.py`) [P1 & P5]**:
   * Kiểm tra tính ổn định của mô hình theo thời gian (chia train/test theo mốc Time thay vì ngẫu nhiên) để đánh giá khả năng chống suy giảm hiệu năng trong thực tế.

### Ngày 17 – 18 (Thứ Sáu, 27/11 – Thứ Bảy, 28/11): Xây dựng Jupyter Notebook & Slide bảo vệ
* **P5 & P1**: Tạo file `notebooks/final_report.ipynb` tổng hợp câu chuyện dự án trực quan (kết nối dữ liệu, bảng chỉ số và biểu đồ đẹp mắt, đã xóa output nặng trước khi nộp).
* **P3 & P2**: Thiết kế bộ Slide thuyết trình chuyên nghiệp tại `slides/Bao_Cao_Bao_Ve.pptx` (18–20 slides theo cấu trúc: Đặt vấn đề ➔ Thách thức mất cân bằng ➔ Giải pháp kỹ thuật ➔ Kết quả thực nghiệm ➔ Giải thích SHAP ➔ Giá trị kinh tế ➔ Kết luận).

### Ngày 19 (Chủ Nhật, 29/11): Tổng duyệt Báo cáo chính thức (vFinal)
* Soát lỗi chính tả toàn văn, định dạng chuẩn canh lề, đánh số bảng biểu, căn chỉnh công thức KaTeX/LaTeX.
* Bổ sung đầy đủ trích dẫn tài liệu tham khảo theo chuẩn IEEE/APA trong `report/tai_lieu_tham_khao.md`.
* Kết xuất file Báo cáo tổng thể PDF chất lượng cao.

### Ngày 20 (Thứ Hai, 30/11): Diễn tập thuyết trình (Rehearsal) & Hỏi đáp phản biện
* **19:30 – 21:30 [BUỔI DIỄN TẬP TOÀN ĐỘI]**:
  * Phân vai thuyết trình theo thế mạnh của từng người (mỗi người 3 phút).
  * Diễn tập trả lời trước các câu hỏi "bẫy" thường gặp của giảng viên phản biện:
    * *Tại sao chọn chỉ số PR-AUC thay vì ROC-AUC hay Accuracy?*
    * *Làm sao đảm bảo SMOTE không bị rò rỉ dữ liệu (data leakage)?*
    * *Vì sao giá trị scale_pos_weight lại xấp xỉ 598.8 mà không phải 577?*
    * *SHAP giúp ích gì cho ngân hàng trong việc tuân thủ pháp lý khi khóa thẻ?*

### Ngày 21 (Thứ Ba, 01/12): Đóng gói dự án & Nộp bài
* Kiểm tra lần cuối: `git status` hoàn toàn sạch, không có file thừa.
* Tạo Git Tag đóng băng phiên bản: `git tag -a v1.0.0 -m "Nghiệm thu đồ án cuối kỳ"`
* Đẩy toàn bộ lên GitHub: `git push origin main --tags`
* Nộp báo cáo, slide và đường dẫn repository cho giảng viên.
