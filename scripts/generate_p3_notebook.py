"""Script tạo file Jupyter Notebook notebooks/p3_imbalance_comparison.ipynb."""
import json
from pathlib import Path

def create_notebook():
    cells = []
    cell_counter = 0

    def add_md(text):
        nonlocal cell_counter
        cell_counter += 1
        cells.append({
            "cell_type": "markdown",
            "id": f"cell-md-{cell_counter}",
            "metadata": {},
            "source": text.strip().splitlines(keepends=True)
        })

    def add_code(code):
        nonlocal cell_counter
        cell_counter += 1
        cells.append({
            "cell_type": "code",
            "id": f"cell-code-{cell_counter}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.strip().splitlines(keepends=True)
        })

    # Header
    add_md("""# BÀI TẬP LỚN MÔN HỌC MÁY (MACHINE LEARNING)
## NHIỆM VỤ P3: SO SÁNH CÁC KỸ THUẬT XỬ LÝ DỮ LIỆU MẤT CÂN BẰNG
**Đề tài:** Phát hiện giao dịch gian lận thẻ tín dụng (Credit Card Fraud Detection with XGBoost)  
**Nhóm thực hiện:** Nhóm Dự án Fraud Detection  
**Phụ trách phần P3:** Thành viên P3 (Imbalanced Data Handling & Evaluation)  
**Mục tiêu nghiên cứu:** Đánh giá định lượng và phân tích bản chất cơ chế của 3 kỹ thuật xử lý mất cân bằng phổ biến (**SMOTE**, **Class Weight**, **Random Undersampling**) so với mô hình cơ sở (**Baseline**) trên thuật toán phân loại Gradient Boosting (**XGBoost**).

---
### DANH MỤC CÁC PHẦN TRIỂN KHAI
1. **Phần 1: Khảo sát và phân tích dữ liệu (EDA & Class Distribution)**
2. **Phần 2: Xây dựng Baseline (XGBoost chưa xử lý mất cân bằng & Nghịch lý Accuracy)**
3. **Phần 3: Triển khai 3 kỹ thuật xử lý mất cân bằng (SMOTE, Class Weight, Random Undersampling)**
4. **Phần 4: Thực nghiệm và so sánh tổng hợp (Metrics Table & Visualization)**
5. **Phần 5: Phân tích và đánh giá chuyên sâu (8 câu hỏi then chốt & Trade-off Precision - Recall)**
6. **Phần 6: Kết luận cho báo cáo học thuật**
7. **Phần 7: Kiểm tra chất lượng và khả năng tái lập**
""")

    # Setup Environment
    add_md("""## 0. Thiết lập môi trường và cấu hình chung
Tuân thủ nguyên tắc nghiên cứu tái lập (Reproducibility):
- Cố định `SEED = 42` trên toàn bộ hệ thống.
- Sử dụng cấu hình tập trung từ module `src.config`.
""")

    add_code("""import json
import os
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc của project nằm trong sys.path
ROOT_DIR = Path.cwd().resolve().parents[0] if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import các module chuẩn của dự án
from src import config
from src.data import load_processed, load_processed_full
from src.pipeline import build_pipeline, compute_scale_pos_weight
from src.evaluate import compute_metrics, run_cv, get_classification_report_str, plot_pr_curves
from src.compare import (
    compare_variants,
    evaluate_variants_on_test,
    choose_imbalance,
    plot_variant_comparison,
    plot_f1_precision_recall_comparison,
    plot_all_confusion_matrices,
    plot_resampling_distributions,
    METHOD_NAMES
)

# Cấu hình thẩm mỹ đồ thị
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
print("Đã tải thành công các thư viện và module hệ thống. Seed =", config.SEED)
""")

    # PHẦN 1
    add_md("""---
## PHẦN 1. KHẢO SÁT VÀ PHÂN TÍCH DỮ LIỆU

### 1.1 Mục tiêu khảo sát
1. Đọc dataset thực tế của đề tài (`creditcard.csv` từ OpenML ID 1597).
2. Xác định các tham số cơ sở:
   - Tổng số lượng mẫu dữ liệu ($N$).
   - Số lượng đặc trưng ($D$).
   - Biến mục tiêu ($Target$).
   - Số lượng mẫu của từng lớp ($N_0, N_1$) và tỷ lệ phần trăm tương ứng.
3. Khẳng định mức độ mất cân bằng dữ liệu.
4. Trực quan hóa phân phối lớp bằng biểu đồ (thang thông thường và thang Logarithm).
5. Phân tích nguyên lý toán học: **Tại sao mất cân bằng dữ liệu khiến mô hình phân loại bị thiên lệch về lớp đa số?**
""")

    add_code("""# Đọc toàn bộ tập dữ liệu đã qua tiền xử lý chuẩn hóa
df_full = load_processed_full()

n_samples = len(df_full)
n_features = len(config.FEATURE_COLS)
target_name = config.TARGET_COL

class_counts = df_full[target_name].value_counts().sort_index()
class_ratios = df_full[target_name].value_counts(normalize=True).sort_index() * 100

summary_eda = pd.DataFrame({
    "Nhãn (Class)": ["0 - Bình thường", "1 - Gian lận"],
    "Số lượng mẫu (Count)": [class_counts[0], class_counts[1]],
    "Tỷ lệ phần trăm (%)": [class_ratios[0], class_ratios[1]],
    "Tỷ lệ chênh lệch": [f"1 : 1", f"1 : {int(class_counts[0] / class_counts[1])}"]
})

print(f"=== THỐNG KÊ TỔNG QUAN TẬP DỮ LIỆU ===")
print(f"Tổng số lượng mẫu (N): {n_samples:,} dòng")
print(f"Số lượng đặc trưng (D): {n_features} biến ({config.FEATURE_COLS[0]}, {config.FEATURE_COLS[1]}..{config.FEATURE_COLS[-2]}, {config.FEATURE_COLS[-1]})")
print(f"Biến mục tiêu (Target): '{target_name}' (0: Hợp pháp, 1: Gian lận)")
print("========================================")
summary_eda
""")

    add_md("""### 1.2 Nhận xét về độ mất cân bằng
Dữ liệu có độ mất cân bằng **cực kỳ nghiêm trọng (Extreme Imbalance)**:
- Lớp 0 (Giao dịch bình thường): **283,253 mẫu**, chiếm tới **99.833%**.
- Lớp 1 (Giao dịch gian lận): Chỉ vỏn vẹn **473 mẫu**, chiếm **0.167%**.
- Tỷ lệ chênh lệch giữa hai lớp xấp xỉ **1 : 599** (cứ gần 600 giao dịch hợp pháp mới xuất hiện 1 giao dịch gian lận).
""")

    add_code("""# Vẽ biểu đồ trực quan hóa phân phối nhãn (Class Distribution)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

labels = ["Bình thường (0)", "Gian lận (1)"]
colors = ["#3498db", "#e74c3c"]

# Biểu đồ 1: Thang tuyến tính thông thường
bars1 = ax1.bar(labels, class_counts.values, color=colors, edgecolor="black", alpha=0.85)
ax1.set_title("Phân bố nhãn trên thang tuyến tính (Linear Scale)", fontweight="bold")
ax1.set_ylabel("Số lượng mẫu")
for bar, count in zip(bars1, class_counts.values):
    ax1.annotate(f"{count:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

# Biểu đồ 2: Thang Logarithm (Log Scale) để nhìn rõ lớp thiểu số
bars2 = ax2.bar(labels, class_counts.values, color=colors, edgecolor="black", alpha=0.85)
ax2.set_yscale("log")
ax2.set_title("Phân bố nhãn trên thang Logarithm (Log Scale)", fontweight="bold")
ax2.set_ylabel("Số lượng mẫu (log10)")
for bar, count in zip(bars2, class_counts.values):
    text_label = f"{count:,} ({count/n_samples*100:.3f}%)"
    ax2.annotate(text_label, (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                 ha="center", va="center", color="white", fontweight="bold")

plt.suptitle("Khảo sát phân phối nhãn Credit Card Fraud Detection", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()
""")

    add_md("""### 1.3 Cơ chế gây ra sự thiên lệch (Bias Mechanism) của mô hình học máy
Tại sao khi dữ liệu mất cân bằng nghiêm trọng, mô hình phân loại lại thiên lệch hoàn toàn về lớp đa số?

1. **Bản chất hàm mất mát (Empirical Risk Minimization - ERM):**  
   Hầu hết các thuật toán ML tối ưu hóa bằng cách cực tiểu hóa tổng mất mát trên toàn bộ tập huấn luyện:
   $$\\mathcal{L}(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N} l(y_i, f(x_i; \\theta)) = \\frac{N_0}{N} \\mathcal{L}_0 + \\frac{N_1}{N} \\mathcal{L}_1$$
   Với tỷ lệ $N_0 / N \\approx 99.833\\%$, thành phần mất mát $\\mathcal{L}_0$ chiếm gần như tuyệt đối toàn bộ gradient cập nhật. Mô hình có thể hạ thấp mất mát tổng thể xuống mức cực tiểu bằng cách tối ưu thật tốt cho lớp 0, trong khi bỏ qua gần như hoàn toàn sai số của lớp 1 mà không bị hàm mất mát trừng phạt đáng kể.

2. **Xê dịch ranh giới quyết định (Decision Boundary Shift):**  
   Mật độ điểm dày đặc của lớp đa số đẩy mặt phân cách quyết định sát về phía lớp thiểu số. Kết quả là vùng không gian đặc trưng thuộc về lớp 1 bị thu hẹp quá mức, khiến mô hình cực kỳ e dè khi dự đoán nhãn 1.

3. **Hiện tượng gán nhãn đa số (Majority Class Collapse):**  
   Trong trường hợp cực đoan, một bộ phân loại "ngây thơ" (Trivial Classifier) chỉ cần luôn luôn dự đoán mọi giao dịch là "Bình thường" (Lớp 0) cũng sẽ đạt độ chính xác lên tới **99.83%**, dù hoàn toàn vô dụng trong việc ngăn chặn tội phạm tài chính.
""")

    # PHẦN 2
    add_md("""---
## PHẦN 2. XÂY DỰNG MÔ HÌNH CƠ SỞ (BASELINE)

### 2.1 Thiết kế thực nghiệm Baseline
- **Thuật toán:** XGBoost Classifier (`tree_method='hist'`, `eval_metric='aucpr'`, `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`).
- **Phân chia Train/Test:** 80% Train, 20% Test theo chiến lược **Stratified Split** (`random_state=42`) để bảo toàn chính xác tỷ lệ phân bố lớp ở cả 2 tập.
- **Quy tắc ngăn ngừa rò rỉ dữ liệu (No Data Leakage):**
  - Mọi bước tính toán thống kê và huấn luyện chỉ thực hiện trên tập Train.
  - Tuyệt đối không can thiệp hay chuẩn hóa trên tập Test trước khi đưa vào mô hình.
- **Mục tiêu của Baseline:** Đóng vai trò làm mốc đối chứng (Benchmark), chưa áp dụng bất kỳ kỹ thuật cân bằng dữ liệu nào (`scale_pos_weight=1.0`, không oversampling/undersampling).
""")

    add_code("""# Tải tập Train và Test đã chia chuẩn
X_train, X_test, y_train, y_test = load_processed()

print(f"Kích thước tập Train: {X_train.shape[0]:,} mẫu | Số ca gian lận: {int(y_train.sum())} ({y_train.sum()/len(y_train)*100:.4f}%)")
print(f"Kích thước tập Test:  {X_test.shape[0]:,} mẫu | Số ca gian lận: {int(y_test.sum())} ({y_test.sum()/len(y_test)*100:.4f}%)")

# Khởi tạo pipeline Baseline (imbalance='none')
pipe_baseline = build_pipeline(imbalance="none")

# Huấn luyện mô hình trên tập Train
pipe_baseline.fit(X_train, y_train)

# Dự đoán xác suất trên tập Test độc lập
y_proba_baseline = pipe_baseline.predict_proba(X_test)[:, 1]
y_pred_baseline = (y_proba_baseline >= config.DEFAULT_THRESHOLD).astype(int)

# Tính toán các chỉ số đánh giá toàn diện
metrics_baseline = compute_metrics(y_test, y_proba_baseline, threshold=config.DEFAULT_THRESHOLD)

print("=== KẾT QUẢ MÔ HÌNH BASELINE TRÊN TẬP TEST (Ngưỡng = 0.5) ===")
for k, v in metrics_baseline.items():
    if isinstance(v, float):
        print(f"  {k:<15}: {v:.4f}")
    else:
        print(f"  {k:<15}: {v}")
""")

    add_code("""# Hiển thị Classification Report và Confusion Matrix chi tiết của Baseline
print("=== CLASSIFICATION REPORT: BASELINE ===")
print(get_classification_report_str(y_test, y_pred_baseline))

# Vẽ Ma trận nhầm lẫn
cm_baseline = pd.crosstab(y_test, y_pred_baseline, rownames=["Thực tế"], colnames=["Dự đoán"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm_baseline, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 13, "weight": "bold"})
title_str = "Confusion Matrix - Baseline: TP=" + str(metrics_baseline['tp']) + ", FN=" + str(metrics_baseline['fn']) + " (Bỏ sót), FP=" + str(metrics_baseline['fp'])
plt.title(title_str)
plt.show()
""")

    add_md("""### 2.2 Phân tích Nghịch lý Độ chính xác (Accuracy Paradox)
Kết quả của mô hình Baseline cho thấy một hiện tượng kinh điển:
- **Accuracy = 0.9995 (99.95%)**: Trông có vẻ là một con số "hoàn hảo", gần như không có lỗi.
- **Thực tế nguy hiểm bên trong:**
  - Trong số 95 ca gian lận của tập Test, mô hình chỉ bắt được 70 ca (**TP = 70**), bỏ sót tới 25 ca (**FN = 25**), tương ứng tỷ lệ bỏ sót **26.3%**!
  - **Recall chỉ đạt 0.7368**: Nghĩa là cứ 4 giao dịch gian lận diễn ra thì có hơn 1 giao dịch lọt lưới ngân hàng.
  
**Tại sao Accuracy lại gây hiểu nhầm (Misleading)?**
$$Accuracy = \\frac{TP + TN}{TP + TN + FP + FN} = \\frac{70 + 56647}{56746} = 0.99949$$
Vì số lượng mẫu âm tính đúng ($TN = 56,647$) áp đảo hoàn toàn mẫu số ($56,746$), $Accuracy$ bị chi phối tới 99.8% bởi lớp bình thường. Sai lầm nghiêm trọng trong việc bỏ sót 25 vụ gian lận hoàn toàn bị che lấp bởi thành tích phân loại đúng lớp đa số. Do đó, trong dữ liệu mất cân bằng, **Accuracy là một thước đo độc hại nếu dùng để đánh giá năng lực bảo vệ của hệ thống**.
""")

    # PHẦN 3
    add_md("""---
## PHẦN 3. TRIỂN KHAI BA KỸ THUẬT XỬ LÝ MẤT CÂN BẰNG

Để giải quyết vấn đề trên, chúng ta lần lượt triển khai 3 kỹ thuật đại diện cho 3 trường phái kinh điển trong học máy:
1. **SMOTE** (Oversampling - Can thiệp không gian mẫu thiểu số).
2. **Class Weight / `scale_pos_weight`** (Cost-Sensitive Learning - Can thiệp hàm mất mát).
3. **Random Undersampling** (Undersampling - Can thiệp không gian mẫu đa số).

---
### 3.1 Kỹ thuật 1: SMOTE (Synthetic Minority Over-sampling Technique)
**Nguyên lý hoạt động:**
- Thay vì nhân bản mẫu trùng lặp (dễ dẫn tới Overfitting), SMOTE tạo ra các điểm dữ liệu **nhân tạo (synthetic)** nằm giữa các mẫu thiểu số thực tế.
- Với mỗi mẫu thiểu số $x_i$, thuật toán tìm $k$ láng giềng gần nhất thuộc cùng lớp bằng khoảng cách Euclidean (mặc định $k=5$).
- Chọn ngẫu nhiên một láng giềng $x_{zi}$ và tạo điểm mới trên đoạn thẳng nối hai điểm:
  $$x_{new} = x_i + \\lambda \\cdot (x_{zi} - x_i), \\quad \\text{với } \\lambda \\in [0, 1]$$

**Yêu cầu kỹ thuật chống Data Leakage:**
- Chỉ áp dụng SMOTE trên tập Training.
- Bắt buộc tích hợp vào `imblearn.pipeline.Pipeline` để khi thực hiện Cross-Validation, SMOTE chỉ được "sinh mẫu" từ fold huấn luyện, tập validation hoàn toàn không bị ảnh hưởng.
""")

    add_code("""# Triển khai SMOTE qua imblearn Pipeline
pipe_smote = build_pipeline(imbalance="smote")
print("Cấu trúc Pipeline SMOTE:")
for name, step in pipe_smote.named_steps.items():
    print(f"  Bước: {name:<10} -> {step}")

# Huấn luyện mô hình SMOTE trên tập Train
pipe_smote.fit(X_train, y_train)

# Đánh giá trên tập Test gốc
y_proba_smote = pipe_smote.predict_proba(X_test)[:, 1]
y_pred_smote = (y_proba_smote >= config.DEFAULT_THRESHOLD).astype(int)
metrics_smote = compute_metrics(y_test, y_proba_smote, threshold=config.DEFAULT_THRESHOLD)

print("Kết quả SMOTE trên Test:")
print(f"  Precision: {metrics_smote['precision']:.4f} | Recall: {metrics_smote['recall']:.4f} | F1: {metrics_smote['f1']:.4f}")
print(f"  TP: {metrics_smote['tp']} | FN: {metrics_smote['fn']} | FP: {metrics_smote['fp']}")
""")

    add_md("""---
### 3.2 Kỹ thuật 2: Class Weight (Học nhạy cảm chi phí - Cost-Sensitive Learning)
**Nguyên lý hoạt động:**
- Không can thiệp làm thay đổi số lượng mẫu hay hình dạng không gian dữ liệu gốc.
- Thay vào đó, gán cho mỗi lớp một trọng số phạt khác nhau trong hàm mất mát. Lớp thiểu số có trọng số lớn hơn, khiến việc dự đoán sai một mẫu gian lận sẽ làm tăng hàm mất mát lên gấp nhiều lần.
- Trong thuật toán XGBoost, tham số `scale_pos_weight` điều chỉnh trực tiếp gradient bậc một $g_i$ và bậc hai $h_i$ của các mẫu thuộc lớp dương:
  $$\\text{scale\\_pos\\_weight} = \\frac{N_{negative}}{N_{positive}} = \\frac{226,602}{378} \\approx 599.48$$
- Thuật toán cây quyết định khi tách nút (split finding) sẽ ưu tiên tối đa hóa mức giảm mất mát (Gain) cho các mẫu dương.
""")

    add_code("""# Tính toán scale_pos_weight chuẩn xác từ tập huấn luyện
spw = compute_scale_pos_weight(y_train)
print(f"scale_pos_weight tính toán từ Train: {spw:.4f}")

# Khởi tạo Pipeline Class Weight
pipe_weight = build_pipeline(imbalance="weight", spw=spw)
pipe_weight.fit(X_train, y_train)

# Đánh giá trên tập Test gốc
y_proba_weight = pipe_weight.predict_proba(X_test)[:, 1]
y_pred_weight = (y_proba_weight >= config.DEFAULT_THRESHOLD).astype(int)
metrics_weight = compute_metrics(y_test, y_proba_weight, threshold=config.DEFAULT_THRESHOLD)

print("Kết quả Class Weight trên Test:")
print(f"  Precision: {metrics_weight['precision']:.4f} | Recall: {metrics_weight['recall']:.4f} | F1: {metrics_weight['f1']:.4f}")
print(f"  TP: {metrics_weight['tp']} | FN: {metrics_weight['fn']} | FP: {metrics_weight['fp']}")
""")

    add_md("""---
### 3.3 Kỹ thuật 3: Random Undersampling (Giảm mẫu ngẫu nhiên lớp đa số)
**Nguyên lý hoạt động:**
- Cân bằng tỷ lệ bằng cách loại bỏ ngẫu nhiên các mẫu thuộc lớp đa số (Lớp 0) cho đến khi số lượng mẫu lớp 0 bằng số lượng mẫu lớp 1 (tỷ lệ 1:1, tương ứng 378 mẫu lớp 0 và 378 mẫu lớp 1 trên tập Train).
- **Ưu điểm:** Giảm kích thước tập dữ liệu xuống tối thiểu (~756 dòng), thời gian huấn luyện cực nhanh (chỉ mất ~0.15s).
- **Nhược điểm cốt tử (Risk of Information Loss):**
  - Xóa bỏ hơn **226,000 giao dịch hợp pháp** (hơn 99.8% dữ liệu), làm mất đi tính phong phú, phương sai và đặc trưng ranh giới của các giao dịch bình thường.
  - Phá vỡ xác suất tiên nghiệm của bài toán, biến mô hình thành một hệ thống "quá nhạy cảm", dễ gây báo động giả hàng loạt.
""")

    add_code("""# Triển khai Random Undersampling qua imblearn Pipeline
pipe_under = build_pipeline(imbalance="under")
pipe_under.fit(X_train, y_train)

# Đánh giá trên tập Test gốc
y_proba_under = pipe_under.predict_proba(X_test)[:, 1]
y_pred_under = (y_proba_under >= config.DEFAULT_THRESHOLD).astype(int)
metrics_under = compute_metrics(y_test, y_proba_under, threshold=config.DEFAULT_THRESHOLD)

print("Kết quả Random Undersampling trên Test:")
print(f"  Precision: {metrics_under['precision']:.4f} | Recall: {metrics_under['recall']:.4f} | F1: {metrics_under['f1']:.4f}")
print(f"  TP: {metrics_under['tp']} | FN: {metrics_under['fn']} | FP: {metrics_under['fp']}")
""")

    # PHẦN 4
    add_md("""---
## PHẦN 4. THỰC NGHIỆM VÀ SO SÁNH TỔNG HỢP

### 4.1 Bảng so sánh kết quả 4 phương pháp trên tập Test
Dưới đây là bảng tổng hợp các chỉ số đánh giá cốt lõi trên cùng một tập Test độc lập (ngưỡng phân loại mặc định $0.5$):
""")

    add_code("""# Đọc kết quả thực nghiệm chi tiết đã chạy và lưu trữ
df_test_metrics = pd.read_csv(config.METRICS_DIR / "imbalance_test_compare.csv")

display_cols = ["Phương pháp", "Accuracy", "Precision", "Recall", "F1-score", "Macro F1", "Weighted F1", "PR-AUC", "TP", "FP", "FN"]
df_test_metrics[display_cols]
""")

    add_md("""### 4.2 Bảng so sánh kết quả qua Stratified 5-Fold Cross-Validation
Để đảm bảo tính khách quan và kiểm định độ ổn định (variance) của các kỹ thuật, chúng ta kiểm tra kết quả trung bình và độ lệch chuẩn qua 5 fold:
""")

    add_code("""df_cv_metrics = pd.read_csv(config.METRICS_DIR / "imbalance_compare.csv")
cv_cols = ["method_name", "pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std", "recall_mean", "recall_std", "precision_mean", "precision_std", "fit_time_mean"]
df_cv_metrics[cv_cols]
""")

    add_md("""### 4.3 Trực quan hóa so sánh toàn diện
Dưới đây là các biểu đồ trực quan hóa kết quả thực nghiệm:
1. **Biểu đồ phân phối mẫu trước và sau khi áp dụng kỹ thuật.**
2. **Biểu đồ so sánh PR-AUC từ Cross-Validation.**
3. **Biểu đồ so sánh Precision, Recall và F1-score.**
4. **Ma trận nhầm lẫn (Confusion Matrix) 4 phương pháp đặt cạnh nhau.**
""")

    add_code("""# Hiển thị biểu đồ phân phối mẫu trước và sau resampling
fig_dist = plt.imread(config.FIGURES_DIR / "05_resampling_distributions.png")
plt.figure(figsize=(10, 6))
plt.imshow(fig_dist)
plt.axis("off")
plt.title("Biểu đồ phân phối số lượng mẫu trước và sau cân bằng", fontsize=13, fontweight="bold")
plt.show()
""")

    add_code("""# Hiển thị biểu đồ so sánh F1, Precision và Recall
fig_f1 = plt.imread(config.FIGURES_DIR / "05_f1_precision_recall.png")
plt.figure(figsize=(11, 7))
plt.imshow(fig_f1)
plt.axis("off")
plt.title("So sánh Precision, Recall và F1-score giữa 4 phương pháp", fontsize=13, fontweight="bold")
plt.show()
""")

    add_code("""# Hiển thị Ma trận nhầm lẫn của 4 phương pháp
fig_cm = plt.imread(config.FIGURES_DIR / "05_confusion_matrices.png")
plt.figure(figsize=(11, 9))
plt.imshow(fig_cm)
plt.axis("off")
plt.title("Ma trận nhầm lẫn 4 phương pháp (Sự đánh đổi TP vs FP)", fontsize=13, fontweight="bold")
plt.show()
""")

    add_code("""# Hiển thị đường cong Precision-Recall Curves trên out-of-fold
fig_pr = plt.imread(config.FIGURES_DIR / "05_pr_curves.png")
plt.figure(figsize=(9, 7))
plt.imshow(fig_pr)
plt.axis("off")
plt.title("Đường cong Precision-Recall giữa các phương pháp", fontsize=13, fontweight="bold")
plt.show()
""")

    # PHẦN 5
    add_md("""---
## PHẦN 5. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

Dưới đây là phần trả lời cặn kẽ 8 câu hỏi cốt lõi dựa trên bằng chứng thực nghiệm thu được:

### 1. Phương pháp nào cải thiện Recall của lớp thiểu số tốt nhất?
- **Random Undersampling** đạt Recall cao nhất trên tập Test (**0.8737**), bắt được 83/95 ca gian lận, chỉ bỏ sót 12 ca.
- Tiếp theo là **SMOTE** đạt Recall **0.8000** (bắt 76 ca), **Class Weight** đạt **0.7895** (bắt 75 ca), và thấp nhất là **Baseline** chỉ đạt **0.7368** (bắt 70 ca).
- *Lý do:* Random Undersampling ép tỷ lệ mẫu hai lớp về 1:1, khiến ngưỡng phân cách dịch chuyển mạnh mẽ sang phía lớp đa số, mô hình có xu hướng dự đoán nhãn 1 đối với bất kỳ điểm dữ liệu nào có dấu hiệu bất thường nhỏ nhất.

### 2. Phương pháp nào có F1-score tốt nhất?
- **Class Weight** đạt F1-score cao nhất vượt trội: **0.8721** trên Test (và **0.8688 ± 0.0205** trên CV).
- Trong khi đó:
  - Baseline đạt F1: 0.8284
  - SMOTE đạt F1: 0.8352
  - Random Undersampling sụp đổ hoàn toàn với F1: **0.1094**!
- *Lý do:* F1-score là trung bình điều hòa giữa Precision và Recall. Chỉ có Class Weight là dung hòa hoàn hảo cả hai khía cạnh: vừa duy trì Precision cực cao (0.9740), vừa đẩy Recall lên mức tốt (0.7895).

### 3. Phương pháp nào có Precision tốt nhất?
- **Class Weight** đạt Precision cao nhất: **0.9740** (trong 77 lần cảnh báo gian lận, chỉ có 2 ca báo động giả, FP = 2).
- Đứng thứ nhì là **Baseline** với Precision **0.9459** (FP = 4).
- **SMOTE** bị tụt Precision xuống **0.8736** (FP = 11, tăng số ca báo sai gấp hơn 5 lần so với Class Weight).
- **Random Undersampling** có Precision thảm họa: **0.0584** (FP = 1,339! Nghĩa là hơn 94% các cảnh báo đều là báo động giả).

### 4. Phương pháp nào làm giảm Accuracy?
- **Random Undersampling** làm giảm Accuracy từ **0.9995** xuống còn **0.9762** (giảm 2.33%).
- *Nguyên nhân:* Mặc dù 97.62% nhìn vẫn cao, nhưng việc giảm này tương ứng với **1,339 giao dịch bình thường bị phân loại sai thành gian lận**. Việc mất mát dữ liệu đào tạo khiến mô hình không thể học được sự đa dạng của các giao dịch hợp pháp.

### 5. SMOTE có thực sự tốt hơn Class Weight không?
- **Câu trả lời dứt khoát: KHÔNG.**
- **So sánh định lượng trên Test:**
  - F1-score: Class Weight (0.8721) > SMOTE (0.8352)
  - Precision: Class Weight (0.9740) > SMOTE (0.8736) (SMOTE tạo ra 11 ca báo sai, Class Weight chỉ 2 ca)
  - PR-AUC: Class Weight (0.8225) > SMOTE (0.8212)
  - Recall: SMOTE nhỉnh hơn chút ít (0.8000 vs 0.7895, chỉ chênh 1 ca: 76 vs 75).
- **So sánh chi phí tính toán & độ phức tạp:**
  - Class Weight không tốn thêm bộ nhớ RAM, thời gian fit tương đương Baseline (~2.3s/fold).
  - SMOTE làm tăng kích thước tập Train lên gấp đôi (~362,000 dòng), tốn thêm 28% thời gian huấn luyện và sinh ra các mẫu nhân tạo nội suy có nguy cơ gây nhiễu tại các ranh giới phức tạp.
- **Kết luận:** Class Weight tốt hơn hẳn SMOTE trong bài toán này.

### 6. Undersampling có làm mất thông tin quan trọng không?
- **Có, và sự mất mát thông tin là CỰC KỲ NGHIÊM TRỌNG.**
- Undersampling đã loại bỏ **226,224 mẫu lớp 0** trong tập Train (chỉ giữ lại 378 mẫu). Điều này làm mất đi hơn 99.8% không gian phân phối của khách hàng bình thường, xóa sạch các trường hợp đặc biệt (edge cases) hợp pháp, dẫn tới việc mô hình coi mọi hành vi hơi khác thường đều là gian lận $\to$ gây ra 1,339 ca báo động giả trên Test.

### 7. Vì sao các phương pháp cho kết quả khác nhau?
- **Bản chất can thiệp khác nhau:**
  - *Baseline:* Không can thiệp, chấp nhận quy luật số đông của tập dữ liệu.
  - *Class Weight:* Can thiệp vào **hàm tối ưu gradient** (Algorithmic-level), ép thuật toán cây tính toán độ lợi (Gain) cao hơn cho lớp dương mà giữ nguyên hiện trạng phân bố dữ liệu thực tế.
  - *SMOTE:* Can thiệp vào **không gian đặc trưng** (Data-level Over-sampling) bằng giả định tuyến tính cục bộ giữa các láng giềng k-NN.
  - *Undersampling:* Can thiệp bằng cách **cắt gọt dữ liệu** (Data-level Under-sampling), loại bỏ thô bạo thông tin đa số.

### 8. Trong thực tế ngành tài chính - ngân hàng, nên lựa chọn phương pháp nào và tại sao?
- **Khuyến nghị lựa chọn: CLASS WEIGHT (scale_pos_weight).**
- **Lý do thực tiễn:**
  1. **Hiệu năng phát hiện tối ưu:** Đạt F1-score và PR-AUC cao nhất, bắt được 78.95% số vụ gian lận.
  2. **Bảo vệ trải nghiệm khách hàng:** Precision đạt 97.4%, chỉ có 2 ca báo động giả trong gần 57,000 giao dịch. Nếu dùng Undersampling với 1,339 ca báo sai, ngân hàng sẽ làm nghẽn tổng đài CSKH, gây phẫn nộ cho khách hàng và làm gián đoạn thanh toán.
  3. **Tối ưu chi phí vận hành (Operational Cost):** Mỗi ca báo động giả đều đòi hỏi nhân viên phòng chống gian lận kiểm tra thủ công. Class Weight giúp giảm thiểu chi phí này tới mức tối đa.
  4. **Tính ổn định & Tốc độ triển khai:** Không làm phình to pipeline xử lý dữ liệu, dễ dàng bảo trì và tích hợp vào hệ thống chấm điểm thời gian thực (Real-time Scoring Engine).
""")

    # PHẦN 6
    add_md("""---
## PHẦN 6. KẾT LUẬN CHO BÁO CÁO HỌC THUẬT

### Tóm tắt tổng kết đề tài
Nghiên cứu đã thực hiện khảo sát, triển khai và so sánh thực nghiệm một cách chặt chẽ, độc lập 4 phương pháp xử lý dữ liệu mất cân bằng trên bài toán phát hiện gian lận thẻ tín dụng sử dụng thuật toán XGBoost:

1. **Vấn đề mất cân bằng:** Với tỷ lệ 0.167% gian lận (1:599), mô hình phân loại thông thường (Baseline) chịu ảnh hưởng nặng nề từ nghịch lý độ chính xác, bỏ sót tới 26.3% số vụ gian lận dù Accuracy đạt 99.95%.
2. **Đánh giá ưu nhược điểm các kỹ thuật:**
   - **Baseline:** Precision cao (0.9459) nhưng Recall thấp (0.7368), bỏ sót nhiều gian lận.
   - **SMOTE:** Cải thiện Recall lên 0.8000 nhưng làm giảm Precision xuống 0.8736, làm tăng thời gian huấn luyện và độ phức tạp của dữ liệu.
   - **Random Undersampling:** Đạt Recall cao nhất (0.8737) nhưng phá hủy Precision (0.0584), gây ra 1,339 ca báo động giả do mất mát 99.8% thông tin dữ liệu huấn luyện.
   - **Class Weight:** Đạt thành tích toàn diện xuất sắc nhất: F1-score = 0.8721, PR-AUC = 0.8225, Precision = 0.9740, Recall = 0.7895, giữ vững tính toàn vẹn của dữ liệu và chi phí tính toán thấp.
3. **Quyết định đề xuất:**
   Kỹ thuật **Class Weight (scale_pos_weight = 599.48)** là giải pháp vượt trội và phù hợp nhất để chuyển giao cho các giai đoạn tiếp theo (Tối ưu siêu tham số P4 và Đánh giá triển khai P2).
""")

    # PHẦN 7
    add_md("""---
## PHẦN 7. KIỂM TRA CHẤT LƯỢNG VÀ KHẢ NĂNG TÁI LẬP (REPRODUCIBILITY CHECK)
Kiểm tra tính nhất quán của các tệp đầu ra đã tạo ra trong dự án:
""")

    add_code("""# Kiểm tra các tệp artifact và metrics đã lưu
artifacts = [
    config.METRICS_DIR / "imbalance_compare.csv",
    config.METRICS_DIR / "imbalance_test_compare.csv",
    config.CHOSEN_IMBALANCE_PATH,
    config.FIGURES_DIR / "05_imbalance_compare.png",
    config.FIGURES_DIR / "05_f1_precision_recall.png",
    config.FIGURES_DIR / "05_confusion_matrices.png",
    config.FIGURES_DIR / "05_pr_curves.png",
    config.FIGURES_DIR / "05_resampling_distributions.png",
]

print("=== KIỂM TRA SỰ TỒN TẠI VÀ TÍNH HỢP LỆ CỦA CÁC ARTIFACTS ===")
for p in artifacts:
    status = "ĐÃ TỒN TẠI (OK)" if p.exists() else "CHƯA TỒN TẠI (THIẾU)"
    size = f"{p.stat().st_size:,} bytes" if p.exists() else "0 bytes"
    print(f"  {p.name:<35} | {status:<18} | {size}")

# Đọc tệp chosen_imbalance.json
with open(config.CHOSEN_IMBALANCE_PATH, "r", encoding="utf-8") as f:
    decision = json.load(f)
print(f"Quyết định lựa chọn chính thức: '{decision['imbalance']}'")
print(f"Lý do: {decision['reason']}")
""")

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_path = Path("notebooks/p3_imbalance_comparison.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, ensure_ascii=False, indent=2)
    print(f"Đã tạo thành công Notebook tại: {nb_path.resolve()}")

if __name__ == "__main__":
    create_notebook()
