# Chương 2: Cơ sở lý thuyết

## 2.1 Tổng quan về bài toán dữ liệu mất cân bằng (Imbalanced Data)
Trong bài toán học có giám sát (Supervised Learning), hiện tượng mất cân bằng dữ liệu (Class Imbalance) xảy ra khi số lượng quan sát thuộc một lớp (lớp đa số - Majority Class / Negative) áp đảo hoàn toàn số lượng quan sát thuộc lớp còn lại (lớp thiểu số - Minority Class / Positive). Trong bài toán phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection), tỷ lệ giao dịch gian lận thường chỉ chiếm dưới 0.2% tổng số giao dịch.

### 2.1.1 Nguyên nhân mô hình bị thiên lệch (Inductive Bias)
Hầu hết các thuật toán học máy cổ điển được thiết kế dựa trên nguyên lý cực tiểu hóa rủi ro kinh nghiệm (Empirical Risk Minimization - ERM) với giả định rằng chi phí cho mọi loại sai lầm là như nhau:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} l(y_i, f(x_i; \theta))$$

Khi lớp đa số chiếm tới 99.83%, hàm mục tiêu sẽ bị chi phối gần như hoàn toàn bởi việc giảm thiểu sai số trên lớp này. Mô hình có xu hướng đẩy ranh giới quyết định (Decision Boundary) lấn sâu vào không gian của lớp thiểu số, dẫn đến hiện tượng mô hình rất ngần ngại dự đoán nhãn 1 (gian lận), gây ra tỷ lệ bỏ sót tội phạm (False Negative) cực kỳ nguy hiểm.

### 2.1.2 Nghịch lý độ chính xác (Accuracy Paradox)
Chỉ số Accuracy được định nghĩa là:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
Nếu một mô hình "ngây thơ" gán nhãn toàn bộ giao dịch là bình thường ($y=0$), mô hình vẫn đạt Accuracy lên tới 99.83%. Do đó, trong dữ liệu mất cân bằng, Accuracy hoàn toàn mất đi ý nghĩa đánh giá và cần được thay thế bằng các thước đo chuyên dụng như Precision, Recall, F1-score, Macro F1, và diện tích dưới đường cong Precision-Recall (PR-AUC).

---

## 2.2 Các kỹ thuật xử lý dữ liệu mất cân bằng

### 2.2.1 Kỹ thuật gán trọng số lớp (Class Weight / Cost-Sensitive Learning)
Thay vì biến đổi số lượng mẫu, phương pháp này can thiệp trực tiếp vào hàm mất mát của thuật toán. Mỗi lớp được gán một trọng số phạt tương ứng với mức độ quan trọng:
$$\mathcal{L}_{CS}(\theta) = \sum_{i: y_i=0} l(0, f(x_i)) + w \cdot \sum_{i: y_i=1} l(1, f(x_i))$$
Trong thuật toán XGBoost, tham số `scale_pos_weight` kiểm soát trọng số này, thường được thiết lập theo tỷ lệ nghịch với tần suất xuất hiện của lớp:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$
**Ưu điểm:** Giữ nguyên vẹn tập dữ liệu gốc, không sinh dữ liệu nhân tạo hay làm mất thông tin, chi phí tính toán tối thiểu.

### 2.2.2 Kỹ thuật tổng hợp mẫu thiểu số SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE là kỹ thuật tăng cường dữ liệu ở cấp độ không gian đặc trưng (Feature Space). Thuật toán không sao chép mẫu trùng lặp mà tạo ra các điểm tổng hợp bằng cách nội suy tuyến tính giữa mẫu thiểu số $x_i$ và một trong $k$ láng giềng gần nhất $x_{zi}$ cùng lớp:
$$x_{\text{new}} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \in [0, 1]$$
**Ưu điểm:** Mở rộng vùng quyết định của lớp thiểu số, giảm nguy cơ Overfitting so với Random Oversampling.  
**Nhược điểm:** Dễ sinh ra các mẫu nhiễu (noise) nằm sâu trong ranh giới của lớp đa số nếu dữ liệu gốc có các điểm ngoại lai (outliers), đồng thời làm tăng kích thước dữ liệu và thời gian huấn luyện.

### 2.2.3 Kỹ thuật giảm mẫu ngẫu nhiên (Random Undersampling)
Phương pháp này cân bằng dữ liệu bằng cách loại bỏ ngẫu nhiên các quan sát thuộc lớp đa số cho tới khi tỷ lệ giữa hai lớp đạt mức mong muốn (thường là 1:1).  
**Ưu điểm:** Giảm đáng kể dung lượng bộ nhớ và tăng tốc độ huấn luyện mô hình.  
**Nhược điểm (Rủi ro mất thông tin - Information Loss):** Việc xóa bỏ phần lớn dữ liệu lớp đa số làm mất đi tính đa dạng và các đặc trưng nhận diện giao dịch hợp pháp, khiến mô hình tạo ra rất nhiều báo động giả (False Positive) khi hoạt động trong môi trường thực tế.

---

## 2.3 Tổng quan thuật toán XGBoost (Extreme Gradient Boosting)
XGBoost là một thuật toán thuộc họ cây tăng cường độ dốc (Gradient Boosted Decision Trees - GBDT), tối ưu hóa hàm mục tiêu có chứa thành phần chính quy hóa (Regularization):
$$\mathcal{L}^{(t)} = \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
Trong đó $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ và $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ lần lượt là đạo hàm bậc một và bậc hai của hàm mất mát. XGBoost nổi bật với khả năng xử lý dữ liệu dạng bảng (Tabular Data) với tốc độ vượt trội nhờ cấu trúc cây xấp xỉ histogram (`tree_method='hist'`) và khả năng kiểm soát Overfitting hiệu quả.

---

## 2.4 Phương pháp tối ưu siêu tham số Bayes (Optuna)
*(Phụ trách bởi P4 - Hyperparameter Tuning)*

## 2.5 Phương pháp giải thích mô hình bằng SHAP (TreeExplainer)
*(Phụ trách bởi P5 - Model Explainability & Integration)*
