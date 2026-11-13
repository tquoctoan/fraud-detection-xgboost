# Chương 6: Xây dựng và Đánh giá mô hình

## 6.1 Thiết lập thử nghiệm và mô hình cơ sở Baseline
Nhằm đảm bảo tính khách quan và khoa học, toàn bộ quy trình thực nghiệm được chuẩn hóa như sau:
- **Tập dữ liệu:** Sử dụng bộ dữ liệu chuẩn đã làm sạch gồm 283,726 giao dịch (sau khi loại bỏ 1,081 dòng trùng lặp hoàn toàn).
- **Phân chia dữ liệu:** Tách tập Train (80% ~ 226,980 dòng, 378 ca gian lận) và tập Test (20% ~ 56,746 dòng, 95 ca gian lận) theo phương pháp phân tầng (`Stratified train_test_split`, `random_state=42`) để bảo toàn chính xác tỷ lệ 0.167% gian lận.
- **Quy tắc ngăn chặn rò rỉ dữ liệu (No Data Leakage):**
  - Mọi tham số và bộ biến đổi (Sampler, Scaler) chỉ được áp dụng và học trên tập Train.
  - Sử dụng `imblearn.pipeline.Pipeline` kết hợp Stratified 5-Fold Cross-Validation, đảm bảo các kỹ thuật Resampling (như SMOTE, Undersampling) chỉ thực hiện bên trong từng fold huấn luyện, không lan sang fold kiểm chứng.
  - Tập Test được giữ nguyên vẹn 100% về phân phối gốc, chỉ được đánh giá một lần duy nhất.
- **Mô hình cơ sở (Baseline):** Sử dụng thuật toán XGBoost với cấu hình tham số cơ sở: `tree_method='hist'`, `eval_metric='aucpr'`, `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`. Mô hình Baseline không áp dụng bất kỳ kỹ thuật cân bằng dữ liệu nào (`scale_pos_weight=1.0`).

---

## 6.2 Kết quả Baseline XGBoost
Kết quả đánh giá mô hình Baseline qua Stratified 5-Fold Cross-Validation và trên tập Test độc lập (ngưỡng phân loại mặc định 0.5) được ghi nhận chi tiết:

| Tập dữ liệu | PR-AUC | Accuracy | Precision | Recall | F1-score | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|---|
| **5-Fold CV** | 0.8541 ± 0.0312 | 0.9996 ± 0.0001 | 0.9447 ± 0.0263 | 0.7910 ± 0.0467 | 0.8598 ± 0.0261 | 0.9298 ± 0.0131 | 0.9996 ± 0.0001 |
| **Test Set** | 0.8120 | 0.9995 | 0.9459 | 0.7368 | 0.8284 | 0.9141 | 0.9995 |

### Ma trận nhầm lẫn của Baseline trên tập Test (Tổng số 95 ca gian lận, 56,651 ca bình thường):
- **True Positive (TP):** 70 ca (bắt được 73.68% số vụ gian lận).
- **False Negative (FN):** 25 ca (bỏ sót 26.32% số vụ gian lận).
- **False Positive (FP):** 4 ca (báo động giả).
- **True Negative (TN):** 56,647 ca.

### Phân tích hạn chế của Baseline:
Mặc dù mô hình Baseline đạt Accuracy lên tới 99.95% và Precision cao (94.59%), nhưng chỉ số Recall chỉ dừng lại ở mức 73.68%, bỏ lọt tới 25 vụ gian lận trong tổng số 95 vụ. Trong nghiệp vụ ngân hàng, mỗi vụ gian lận bị bỏ lọt đồng nghĩa với tổn thất tài chính trực tiếp và suy giảm uy tín tổ chức. Điều này minh chứng cho sự cần thiết phải can thiệp bằng các kỹ thuật xử lý dữ liệu mất cân bằng.

---

## 6.3 So sánh các phương pháp xử lý mất cân bằng (Nhiệm vụ P3)
Chúng tôi tiến hành triển khai và so sánh đồng bộ 4 phương pháp trên cùng một nền tảng thực nghiệm:
1. **Baseline:** Không can thiệp mất cân bằng (`imbalance='none'`).
2. **Class Weight:** Gán trọng số phạt `scale_pos_weight = 599.48` cho lớp gian lận (`imbalance='weight'`).
3. **SMOTE:** Sinh mẫu nhân tạo cho lớp thiểu số qua k-NN (`k=5`, `imbalance='smote'`).
4. **Random Undersampling:** Giảm ngẫu nhiên số lượng mẫu lớp bình thường về tỷ lệ 1:1 (`imbalance='under'`).

### 6.3.1 Bảng so sánh kết quả thực nghiệm tổng hợp trên tập Test độc lập

| Phương pháp | Accuracy | Precision | Recall | F1-score | Macro F1 | Weighted F1 | PR-AUC | TP | FP | FN (Bỏ sót) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Gốc)** | 0.9995 | 0.9459 | 0.7368 | 0.8284 | 0.9141 | 0.9995 | 0.8120 | 70 | 4 | 25 |
| **Class Weight** | **0.9996** | **0.9740** | 0.7895 | **0.8721** | **0.9359** | **0.9996** | **0.8225** | 75 | **2** | 20 |
| **SMOTE** | 0.9995 | 0.8736 | 0.8000 | 0.8352 | 0.9175 | 0.9995 | 0.8212 | 76 | 11 | 19 |
| **Undersampling** | 0.9762 | 0.0584 | **0.8737** | 0.1094 | 0.5487 | 0.9865 | 0.7024 | **83** | 1,339 | **12** |

### 6.3.2 Bảng so sánh kết quả 5-Fold Stratified Cross-Validation (Trung bình ± Độ lệch chuẩn)

| Phương pháp | PR-AUC (CV) | F1-score (CV) | Recall (CV) | Precision (CV) | Thời gian Fit / fold |
|---|---|---|---|---|---|
| **Baseline** | 0.8541 ± 0.0312 | 0.8598 ± 0.0261 | 0.7910 ± 0.0467 | 0.9447 ± 0.0263 | 2.29s |
| **Class Weight** | 0.8576 ± 0.0319 | **0.8688 ± 0.0205** | 0.8253 ± 0.0406 | 0.9195 ± 0.0306 | 2.32s |
| **SMOTE** | **0.8585 ± 0.0346** | 0.8418 ± 0.0275 | 0.8359 ± 0.0425 | 0.8510 ± 0.0496 | 2.93s |
| **Undersampling** | 0.7407 ± 0.0638 | 0.0924 ± 0.0096 | **0.9021 ± 0.0339** | 0.0487 ± 0.0052 | **0.16s** |

### 6.3.3 Phân tích chuyên sâu và sự đánh đổi (Trade-off Analysis)

#### 1. Sự đánh đổi giữa Precision và Recall
- **Random Undersampling:** Đạt Recall cao nhất (0.8737), phát hiện được 83/95 ca gian lận. Tuy nhiên, việc loại bỏ ngẫu nhiên 99.83% dữ liệu lớp bình thường đã dẫn tới **mất mát thông tin nghiêm trọng (Severe Information Loss)**. Precision sụp đổ xuống mức thảm họa **5.84%** với **1,339 ca báo động giả (FP = 1,339)**. Trong thực tế, hệ thống này sẽ làm tê liệt bộ phận vận hành và gây gián đoạn dịch vụ nghiêm trọng cho khách hàng hợp pháp.
- **SMOTE:** Tăng Recall lên 0.8000 (bắt 76 ca), nhưng việc nội suy điểm dữ liệu trong không gian nhiều chiều tạo ra các mẫu nhân tạo xâm lấn vào vùng biên giới của lớp bình thường, làm giảm Precision từ 0.9459 xuống 0.8736 (FP tăng từ 4 lên 11). Đồng thời, thời gian huấn luyện tăng gần 30% do kích thước tập Train phình to gấp đôi.
- **Class Weight:** Đạt sự cân bằng lý tưởng nhất. Bằng cách phạt nặng hàm mất mát khi đoán sai lớp dương, mô hình cải thiện Recall từ 0.7368 lên **0.7895** (+5.27%), đồng thời nâng Precision lên mức kỷ lục **0.9740** (chỉ có 2 ca báo sai trong gần 57,000 giao dịch). F1-score đạt **0.8721** (cao nhất trong cả 4 phương pháp).

#### 2. So sánh giữa SMOTE và Class Weight
Nhiều nghiên cứu học thuật thường mặc định SMOTE sẽ vượt trội hơn các phương pháp đơn giản. Tuy nhiên, trên dữ liệu giao dịch thực tế kết hợp với thuật toán GBDT (XGBoost), **Class Weight hoàn toàn chiến thắng SMOTE**:
- Về F1-score: Class Weight (0.8721) cao hơn SMOTE (0.8352).
- Về độ tin cậy cảnh báo: Class Weight chỉ có 2 ca báo động giả, trong khi SMOTE gây ra 11 ca.
- Về tài nguyên tính toán: Class Weight không tốn thêm bộ nhớ RAM, thời gian fit nhanh hơn đáng kể so với SMOTE.

### 6.3.4 Quyết định lựa chọn phương pháp cho các bước tiếp theo
Dựa trên quy tắc thống kê học máy đã thống nhất trong quy định dự án:
- Mức PR-AUC giữa Class Weight (0.8576) và SMOTE (0.8585) là tương đương nhau (chênh lệch chỉ 0.0009, nhỏ hơn nhiều so với độ lệch chuẩn 0.0346).
- Theo nguyên lý Parsimony (ưu tiên giải pháp đơn giản và ổn định), phương pháp **Class Weight (`scale_pos_weight = 599.48`)** được chính thức lựa chọn để bàn giao cho thành viên P4 thực hiện tối ưu hóa siêu tham số bằng Optuna.
- Quyết định này đã được ghi nhận tại `results/params/chosen_imbalance.json`.

---

## 6.4 Tối ưu hóa siêu tham số bằng Optuna
*(Phụ trách bởi P4 - Hyperparameter Tuning)*

## 6.5 Đánh giá mô hình tối ưu trên tập Test
*(Phụ trách bởi P2 - Core Evaluation)*

## 6.6 Tinh chỉnh ngưỡng phân loại (Threshold Tuning)
*(Phụ trách bởi P2 - Core Evaluation)*

## 6.7 Giải thích mô hình với SHAP và Phân tích lỗi (Error Analysis)
*(Phụ trách bởi P5 - Model Explainability & Integration)*
