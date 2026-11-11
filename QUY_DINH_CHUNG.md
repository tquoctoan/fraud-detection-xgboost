# QUY ĐỊNH CHUNG: Fraud Detection with XGBoost

Tài liệu này trả lời 4 câu hỏi của mỗi thành viên:

1. **Mình được sửa file nào?** (Phần A)
2. **Mỗi ngày mình làm gì, lấy file của ai, giao file gì cho ai?** (Phần B)
3. **Ai phải giao gì cho ai, hạn ngày mấy, nếu chậm thì sao?** (Phần C)
4. **Mỗi file `.py` của mình viết như thế nào?** (Phần D)

Phụ lục: chữ ký hàm và lược đồ file (dùng để tra cứu khi code).

| Mục | Nội dung |
|---|---|
| [Phần A](#phần-a-luật-chung) | Luật chung, bảng file, cách chạy |
| [Phần B](#phần-b-lộ-trình-từng-người-theo-ngày) | Lộ trình P1 đến P5 theo ngày |
| [Phần C](#phần-c-bảng-bàn-giao) | Bảng bàn giao và xử lý khi chậm |
| [Phần D](#phần-d-hướng-dẫn-làm-từng-file) | Hướng dẫn từng file `.py` |
| [Phụ lục](#phụ-lục-chữ-ký-hàm-và-lược-đồ-file) | Chữ ký hàm, lược đồ file |

---

# PHẦN A. LUẬT CHUNG

## A.1 Bảy luật

1. **Chỉ sửa file của mình** (xem bảng A.2). Cần đổi file người khác thì nhắn chủ file hoặc mở Issue, ghi rõ hàm nào, đầu vào và đầu ra mong muốn.
2. **`src/` chứa hàm, `scripts/` chứa lệnh chạy.** File trong `src/` chỉ định nghĩa hàm và hằng số, import vào không được tự chạy gì. Script chỉ gọi hàm, không viết logic.
   *Ngoại lệ:* `02_eda_overview.py` và `03_eda_features.py` được viết thẳng lệnh vẽ biểu đồ trong `main()`, vì chỉ là vẽ hình, không có logic dùng chung.
3. **Luôn chạy từ thư mục gốc của repo**: `python scripts/xx.py`.
4. **Seed và đường dẫn lấy từ `src/config.py`.** Không tự gõ `42`, không tự viết đường dẫn.
5. **Không dùng tập test để chọn bất cứ thứ gì** (tham số, ngưỡng, đặc trưng, mô hình). Chỉ `07_evaluate.py` của P2 được đánh giá trên test, đúng một lần.
6. **Chưa có file của người trước thì đừng ngồi chờ.** Viết báo cáo phần lý thuyết, viết nháp code theo chữ ký hàm ở Phụ lục. Có file là chạy thử ngay.
7. **Bàn giao đúng hạn là ưu tiên số 1.** Ai sắp trễ phải báo cả nhóm **trước hạn**. Làm xong thì mở Pull Request, người review duyệt rồi mới merge vào `main`.

## A.2 Mỗi file làm gì, của ai

| File | Nhiệm vụ (một câu) | Chủ |
|---|---|---|
| `src/config.py` | Hằng số, seed, đường dẫn, tham số mặc định. Không có logic | P2 |
| `src/data.py` | Tải, làm sạch, chia, lưu và đọc dữ liệu | P1 |
| `src/pipeline.py` | Dựng pipeline: scaler tùy chọn, cân bằng tùy chọn, XGBoost | P2 |
| `src/evaluate.py` | Cross-validation, chỉ số, chọn ngưỡng, vẽ biểu đồ đánh giá | P2 |
| `src/compare.py` | So sánh 4 cách xử lý mất cân bằng và chọn cách tốt nhất | P3 |
| `src/tune.py` | Tìm tham số XGBoost tốt nhất bằng Optuna | P4 |
| `src/explain.py` | SHAP và phân tích ca đoán sai | P5 |
| `scripts/01_preprocess.py` | Tải, làm sạch, chia, lưu `parquet` | P1 |
| `scripts/02_eda_overview.py` | EDA tổng quan: nhãn, `Amount`, `Time` | P1 |
| `scripts/03_eda_features.py` | EDA V1-V28, tương quan, xu hướng | P5 |
| `scripts/04_baseline_xgb.py` | Mô hình XGBoost cơ sở | P2 |
| `scripts/05_imbalance_compare.py` | Chạy so sánh `none/weight/smote/under` | P3 |
| `scripts/06_tune_xgb.py` | Chạy tuning, lưu `best_params.json` | P4 |
| `scripts/07_evaluate.py` | Mô hình cuối, chọn ngưỡng, đánh giá test một lần | P2 |
| `scripts/08_explain.py` | SHAP và phân tích lỗi trên test | P5 |
| `scripts/run_all.py` | Chạy tuần tự `01` đến `08` | P5 |
| `notebooks/final_report.ipynb` | Ghép cuối, chỉ làm khi `run_all.py` chạy ổn | P5 |

## A.3 Cách chạy và đặt tên

- Cài đặt môi trường và chạy từng bước: xem `README.md`.
- Tên file đầu ra theo mẫu: hình `NN_mo_ta.png` (NN là số của script), chỉ số `ten.csv`, dự đoán `xgb_<biến thể>_oof.csv`. Tên chính xác của từng file nằm ở Phần B và Phụ lục.
- Biến thể cân bằng chỉ có 4 giá trị: `none`, `weight`, `smote`, `under`.

## A.4 Git

- Mỗi người một nhánh: `feat/p1-data`, `feat/p2-core`, ...
- Mở Pull Request vào `main`, người review (theo vòng P1→P2→P3→P4→P5→P1) duyệt.
- Không commit: `data/`, `results/models/`, `scratch/`, `.venv/`.
- Commit message ngắn, rõ: `P3: thêm so sánh SMOTE vs scale_pos_weight`.

## A.5 Tự kiểm tra trước khi mở Pull Request

- [ ] Script chạy từ thư mục gốc, không lỗi
- [ ] Không sửa file của người khác
- [ ] Không có đường dẫn cứng, không có `plt.show()`
- [ ] Mọi thứ ngẫu nhiên dùng `config.SEED`
- [ ] Không dùng tập test để chọn gì (trừ `07` của P2)
- [ ] File đầu ra đúng tên, đúng thư mục


---

# PHẦN B. LỘ TRÌNH TỪNG NGƯỜI THEO NGÀY

Phạm vi: **tuần 1 (ngày 1-7)** là code và thực nghiệm. Tuần 2-3 (viết báo cáo, kiểm tra chéo, bonus) xem [`PHAN_CONG.md`](PHAN_CONG.md) và [`HUONG_DAN_VIET_BAO_CAO.md`](HUONG_DAN_VIET_BAO_CAO.md).

> Chữ **đậm** trong cột "Output" là **hạn bàn giao**: người khác đang chờ file đó.

## B.1 P1: Dữ liệu (đường găng số 1, không phụ thuộc ai)

**File của P1:** `src/data.py`, `scripts/01_preprocess.py`, `scripts/02_eda_overview.py`

| Ngày | Làm gì (file) | Input | Output |
|---|---|---|---|
| **1** | Viết `data.py` (`fetch_raw`, `load_raw`, `clean`, `split_data`, `summarize_preprocess`, `save_processed`, `load_processed`, `load_processed_full`) và chạy thật `01_preprocess.py` | OpenML ID 1597 | `data/processed/train.parquet` (~226.980 dòng, 378 gian lận), `test.parquet` (~56.746 dòng, 95 gian lận), `results/metrics/preprocess_summary.csv`. **Hạn: cuối ngày 1, giao P2, P3, P4, P5** |
| 2-3 | Viết `02_eda_overview.py` | `load_processed_full()` | `figures/02_label_distribution.png`, `02_amount_hist.png`, `02_amount_box.png`, `02_time_dist.png`; `results/metrics/eda_describe.csv`, `eda_label_counts.csv` |
| 4-6 | Viết báo cáo Chương 1, Chương 4, mục 5.1-5.3 | `preprocess_summary.csv`, các hình `02_*` | `report/ch1_*`, `ch4_*`, `ch5_*` (phần 1) |
| 7 | Chạy lại `01` và `02` từ đầu, mở Pull Request | | Mốc 1 |

Từ ngày 2 P1 nhẹ việc nhất. Nếu P2 hoặc P4 chậm thì P1 hỗ trợ (không nhận gói mới).

## B.2 P2: Lõi mô hình

**File của P2:** `src/config.py` (đã có), `src/pipeline.py`, `src/evaluate.py`, `scripts/04_baseline_xgb.py`, `scripts/07_evaluate.py`

| Ngày | Làm gì (file) | Input | Output |
|---|---|---|---|
| **1** | Viết `pipeline.py` (`compute_scale_pos_weight`, `build_xgb`, `build_pipeline`, `get_clf`, `transform_features`) và `evaluate.py` (`run_cv`, `compute_metrics`, `find_threshold`, hàm vẽ, `save_metrics`). Khi có `parquet` thì test | `train.parquet` (P1, cuối ngày 1) | **Hạn: sáng ngày 2**: `pipeline.py` và `evaluate.py` bản chạy được (bắt buộc có `build_pipeline` và `run_cv`), giao P3, P4, P5 |
| **2** | Viết `04_baseline_xgb.py` và chạy | `train.parquet` | `results/metrics/baseline.csv`, `results/predictions/xgb_baseline_oof.csv`, `figures/04_baseline_pr_curve.png`, `results/models/xgb_baseline.joblib`. **Hạn: cuối ngày 2, giao P5 file `.joblib`** |
| 3-4 | Hoàn thiện hàm vẽ và chọn ngưỡng; viết `07_evaluate.py`; viết Chương 3 | | |
| 5 | Thử logic `07` bằng dự đoán out-of-fold (chạy với `--no-test`, **chưa đánh giá test**); viết 6.1-6.2 | `xgb_baseline_oof.csv` | |
| **6** | Chạy `07_evaluate.py` thật, **một lần duy nhất** | `best_params.json` (P4), `train/test.parquet` | `results/models/xgb_final.joblib`, `results/params/threshold.json`, `results/predictions/xgb_final_test.csv`, `results/metrics/final_test.csv`, `figures/07_pr_curve.png`, `07_confusion_matrix.png`. **Hạn: cuối ngày 6, giao P5** |
| 7 | Viết 6.5-6.6, mốc 1 | | |

## B.3 P3: So sánh cân bằng (phụ thuộc P2)

**File của P3:** `src/compare.py`, `scripts/05_imbalance_compare.py`

| Ngày | Làm gì (file) | Input | Output |
|---|---|---|---|
| 1 | Viết báo cáo Chương 2.1-2.3 (lý thuyết, không cần số liệu). Viết nháp `compare.py` và `05` theo chữ ký hàm ở Phụ lục (chưa chạy) | | |
| 2 | Khi có code của P2 và `parquet`: chạy thử `05` với 1 biến thể, sửa lỗi | `build_pipeline`, `run_cv` (P2), `train.parquet` (P1) | |
| **3** | Chạy `05` đủ 4 biến thể `none/weight/smote/under` | `train.parquet` | `results/metrics/imbalance_compare.csv`, `results/predictions/xgb_<biến thể>_oof.csv`, `figures/05_imbalance_compare.png`, `05_pr_curves.png`, `results/params/chosen_imbalance.json` (có trường `reason`). **Hạn: cuối ngày 3, giao P4** |
| 4-6 | Viết báo cáo mục 6.3, dàn ý Kết luận | `imbalance_compare.csv` | `report/ch6_*` (6.3) |
| 7 | Mốc 1 | | |

## B.4 P4: Tuning (phụ thuộc P2, P3)

**File của P4:** `src/tune.py`, `scripts/06_tune_xgb.py`

| Ngày | Làm gì (file) | Input | Output |
|---|---|---|---|
| 1 | Viết báo cáo Chương 2.4-2.6, bắt đầu gom tài liệu tham khảo. Viết nháp `tune.py` và `06` theo chữ ký ở Phụ lục (chưa chạy) | | |
| 2 | Khi có code của P2 và `parquet`: chạy thử 3-5 vòng, **đo thời gian một vòng** để quyết định số vòng (30-50) và có giảm số fold khi tuning không | `build_pipeline`, `run_cv` (P2), `train.parquet` (P1) | số vòng quyết định |
| 3 | Hoàn thiện `tune.py`, tài liệu tham khảo. Nhận `chosen_imbalance.json` cuối ngày | `chosen_imbalance.json` (P3) | |
| **4-5** | Chạy tuning chính thức | `train.parquet`, `chosen_imbalance.json` | `results/params/best_params.json`, `results/metrics/tuning_trials.csv`, `tuning_compare.csv`, `figures/06_optimization_history.png`. **Hạn: cuối ngày 5, giao P2** |
| 6-7 | Viết báo cáo 6.4 và tài liệu tham khảo; mốc 1 | | |

## B.5 P5: Phân tích và tích hợp

**File của P5:** `src/explain.py`, `scripts/03_eda_features.py`, `scripts/08_explain.py`, `scripts/run_all.py`, `notebooks/final_report.ipynb`

| Ngày | Làm gì (file) | Input | Output |
|---|---|---|---|
| 1 | Viết nháp `explain.py` và khung `run_all.py` theo chữ ký ở Phụ lục (chưa chạy) | | |
| 2 | Chạy thật `03_eda_features.py` | `load_processed_full()` (P1) | `figures/03_correlation_matrix.png`, `03_top_corr_features.png`, `03_feature_dist_top.png`, `03_fraud_vs_time.png`, `03_fraud_vs_amount.png`; `results/metrics/eda_corr_with_class.csv` |
| 3 | Thử `explain.py` và `08` trên **mô hình baseline** với ngưỡng 0,5; hoàn thiện `run_all.py` | `xgb_baseline.joblib` (P2, cuối ngày 2) | code `08` đã chạy được |
| 4-5 | Viết báo cáo 5.4-5.6; chạy thử `run_all.py` đến bước đã có | | |
| **6** | Chạy `08_explain.py` thật | `xgb_final.joblib`, `threshold.json` (P2), `test.parquet` | `figures/08_shap_summary.png`, `08_shap_bar.png`; `results/metrics/errors.csv`, `errors_summary.csv` |
| 7 | Chạy `run_all.py` trọn từ đầu đến cuối (dùng `--skip 05 06` để khỏi chạy lại phần lâu); mốc 1 | | Mốc 1 |
| Tuần 2 | Viết 6.7; ghép `final_report.ipynb` sau khi `run_all.py` ổn | | |

---

# PHẦN C. BẢNG BÀN GIAO

| Ai giao | Giao gì | Cho ai | Hạn |
|---|---|---|---|
| P1 | `train.parquet`, `test.parquet` | P2, P3, P4, P5 | cuối ngày 1 |
| P2 | `pipeline.py`, `evaluate.py` (bản chạy được) | P3, P4, P5 | sáng ngày 2 |
| P2 | `xgb_baseline.joblib` | P5 | cuối ngày 2 |
| P3 | `chosen_imbalance.json` | P4 | cuối ngày 3 |
| P4 | `best_params.json` | P2 | cuối ngày 5 |
| P2 | `xgb_final.joblib`, `threshold.json` | P5 | cuối ngày 6 |
| P5 | `run_all.py` chạy trọn | cả nhóm | ngày 7 |

## Nếu ai đó bị chậm

| Tình huống | Cách xử lý |
|---|---|
| P1 chưa giao `parquet` | Đây là đường găng. Mọi người chuyển sang viết báo cáo. **Không ai tự tải và chia riêng**, vì kết quả sẽ khác nhau |
| P2 trễ `pipeline.py` quá nửa ngày | P1 (đã xong sớm) hỗ trợ viết phần vẽ biểu đồ trong `evaluate.py` |
| P3 chưa có `chosen_imbalance.json` | P4 bắt đầu tuning với `weight`; nếu P3 chọn biến thể khác thì chạy lại, ghi rõ trong báo cáo |
| Tuning chạy quá lâu | P4 giảm số vòng hoặc số fold khi tuning, ghi rõ vào báo cáo |
| P4 trễ `best_params.json` | P2 chạy `07 --no-test` (tạo mô hình và ngưỡng, **không chạm test**) để P5 có mô hình thử, rồi chạy `07` đầy đủ khi có tham số |

---

# PHẦN D. HƯỚNG DẪN LÀM TỪNG FILE

Mỗi file có cùng một khuôn:

- **Làm gì**: mục đích của file
- **Các bước**: làm theo thứ tự
- **Code gợi ý**: chỉ những đoạn khó (đã chạy thử, có thể dùng gần như nguyên văn)
- **Lỗi hay gặp**
- **Xong khi**: lệnh kiểm tra và kết quả đúng

Các import dùng chung ở đầu mỗi file `src/`:

```python
from __future__ import annotations
import logging
from src import config
log = logging.getLogger(__name__)
```

Mỗi script có 3 dòng đầu để import được `src/` và cấu hình log (xem mẫu ở D.9).

---

## D.1 `src/data.py` (P1)

**Làm gì:** mọi thứ liên quan đến đọc, làm sạch, chia và lưu dữ liệu. Không scale, không SMOTE, không vẽ, không huấn luyện.

**Các bước:**

1. `fetch_raw()`: nếu `config.RAW_CSV` đã có thì bỏ qua. Nếu chưa: `fetch_openml` với `target_column="Time"` (để giữ đủ 31 cột), ép `Class` về `int`, lưu CSV.
2. `load_raw()`: đọc `config.RAW_CSV`.
3. `clean(df)`: kiểm tra thiếu, **xóa trùng trên cả 31 cột**, `reset_index`, log số dòng và số ca gian lận trước/sau, cảnh báo nếu lệch so với `config.EXPECTED_*`.
4. `load_and_clean()` = `load_raw()` rồi `clean()`.
5. `split_data(df)`: chia stratified, trả `X_train, X_test, y_train, y_test`.
6. `summarize_preprocess(...)`: trả dict để ghi `preprocess_summary.csv`.
7. `save_processed(...)` và `load_processed()`: ghi/đọc `parquet`.
8. `load_processed_full()`: nối train và test lại (dùng cho EDA mô tả).

**Code gợi ý:**

```python
def fetch_raw():
    if config.RAW_CSV.exists():
        log.info("Đã có %s, bỏ qua bước tải", config.RAW_CSV)
        return config.RAW_CSV
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=config.OPENML_ID, as_frame=True,
                         target_column=config.OPENML_FETCH_TARGET)
    df = bunch.frame
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(str).astype(int)
    df = df[config.FEATURE_COLS + [config.TARGET_COL]]   # đúng thứ tự cột
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_CSV, index=False)
    return config.RAW_CSV


def clean(df):
    n0, f0 = len(df), int(df[config.TARGET_COL].sum())
    log.info("Giá trị thiếu: %d", int(df.isna().sum().sum()))
    df = df.drop_duplicates().reset_index(drop=True)   # KHÔNG truyền subset: xét cả 31 cột
    log.info("Trước: %d dòng, %d gian lận | Sau: %d dòng, %d gian lận",
             n0, f0, len(df), int(df[config.TARGET_COL].sum()))
    return df


def split_data(df):
    from sklearn.model_selection import train_test_split
    X, y = df[config.FEATURE_COLS], df[config.TARGET_COL]
    return train_test_split(X, y, test_size=config.TEST_SIZE,
                            stratify=y, random_state=config.SEED)
    # thứ tự trả về: X_train, X_test, y_train, y_test
```

Khi lưu `parquet`: ghép đặc trưng và nhãn thành một bảng (`pd.concat([X, y], axis=1)`), dùng `to_parquet(path, index=False)` (cần `pyarrow`). Khi đọc: tách lại `X = df[config.FEATURE_COLS]`, `y = df[config.TARGET_COL].astype(int)`.

**Lỗi hay gặp:**
- Xóa trùng thiếu cột `Time` thì **xóa nhầm 9.144 dòng**. Dùng `drop_duplicates()` mặc định (không truyền `subset`) trên DataFrame đủ 31 cột.
- `Class` đọc về dạng chuỗi hoặc `category`: luôn ép `astype(str).astype(int)`.
- Chia xong mới xóa trùng: sai thứ tự, gây rò rỉ giữa train và test. **Xóa trùng trước, chia sau.**
- Quên `reset_index(drop=True)`: về sau `X.iloc[...]` và nhãn lệch nhau.

**Xong khi:**

```bash
python -c "from src import data; df = data.load_and_clean(); print(df.shape, int(df['Class'].sum()))"
# đúng: (283726, 31) 473
```

---

## D.2 `scripts/01_preprocess.py` (P1)

**Làm gì:** chạy toàn bộ tiền xử lý một lần, ghi `parquet` và bảng tóm tắt.

**Các bước trong `main()`:**

1. `config.ensure_dirs()`
2. `fetch_raw()` (tự bỏ qua nếu đã có CSV)
3. `df_raw = load_raw()`, `df = clean(df_raw)`
4. `X_train, X_test, y_train, y_test = split_data(df)`
5. `save_processed(X_train, X_test, y_train, y_test)`
6. `summary = summarize_preprocess(df_raw, df, y_train, y_test)`, ghi bằng pandas:
   `pd.DataFrame([summary]).to_csv(config.METRICS_DIR / "preprocess_summary.csv", index=False)`
   (P1 tự ghi bằng pandas, **không** import `evaluate.py` của P2, vì P2 chưa có vào ngày 1.)
7. Log số ca gian lận ở train và test.

**Các cột của `preprocess_summary.csv`:** `n_raw, n_dup, n_clean, fraud_raw, fraud_clean, train_n, train_fraud, test_n, test_fraud`.

**Lỗi hay gặp:** để lẫn bước scale/SMOTE vào đây (cấm); quên `ensure_dirs()` nên không ghi được file.

**Xong khi:**

```bash
python scripts/01_preprocess.py
```
Kết quả đúng: 284.807 dòng gốc, 1.081 dòng trùng, 283.726 dòng sau làm sạch (473 gian lận); train ~226.980 dòng (378 gian lận), test ~56.746 dòng (95 gian lận); có `train.parquet`, `test.parquet`, `preprocess_summary.csv`.

---

## D.3 `scripts/02_eda_overview.py` (P1)

**Làm gì:** thống kê và biểu đồ tổng quan (phục vụ mục 5.1-5.3). Dùng toàn bộ dữ liệu đã làm sạch, **chỉ để mô tả**.

**Các bước:**

1. `df = load_processed_full()`
2. `df.describe().T` lưu `eda_describe.csv`.
3. Đếm nhãn và tỷ lệ, lưu `eda_label_counts.csv` (cột: `class, count, ratio`).
4. Vẽ 4 hình (lưu vào `config.FIGURES_DIR`):

| Hình | Cách vẽ |
|---|---|
| `02_label_distribution.png` | Biểu đồ cột hai lớp, trục y thang log (`plt.yscale("log")`), ghi số trên cột |
| `02_amount_hist.png` | Histogram của `np.log1p(Amount)`, ghi rõ trục "log(1 + Amount)" |
| `02_amount_box.png` | Boxplot `np.log1p(Amount)` theo `Class` |
| `02_time_dist.png` | `Time / 3600` (giờ), histogram theo lớp, `stat="density"`, `common_norm=False` |

Mỗi hình: có tiêu đề, nhãn trục, `plt.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")`, rồi `plt.close()`.

**Lỗi hay gặp:** vẽ `Amount` thang thường nên mọi thứ dồn vào sát 0; histogram `Time` của hai lớp không dùng `common_norm=False` nên lớp gian lận gần như phẳng; dùng `plt.show()`.

**Xong khi:** `python scripts/02_eda_overview.py` ra đủ 4 hình và 2 file CSV; mở từng hình xem trục và tiêu đề có đủ.

---

## D.4 `src/pipeline.py` (P2)

**Làm gì:** dựng pipeline XGBoost, đóng gói scaler, bộ cân bằng và mô hình.

**Các bước:**

1. `compute_scale_pos_weight(y)`: (số lớp 0) / (số lớp 1), **tính bằng code**.
2. `build_xgb(params=None, scale_pos_weight=1.0)`: gộp `XGB_FIXED_PARAMS`, `XGB_BASE_PARAMS`, rồi `params` (tham số truyền vào ghi đè).
3. `build_pipeline(imbalance, scale, params, spw)`: ghép các bước theo thứ tự `scaler` (nếu `scale=True`), `sampler` (nếu `smote`/`under`), `clf`.
4. `get_clf(pipe)` và `transform_features(pipe, X)`: để P5 lấy mô hình và đặc trưng cho SHAP.

**Code gợi ý:**

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline            # KHÔNG dùng sklearn.pipeline.Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


def compute_scale_pos_weight(y) -> float:
    return float((y == 0).sum() / (y == 1).sum())


def build_xgb(params=None, scale_pos_weight=1.0):
    p = {**config.XGB_FIXED_PARAMS, **config.XGB_BASE_PARAMS, **(params or {})}
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **p)


def build_pipeline(imbalance="weight", scale=False, params=None, spw=None):
    if imbalance not in config.IMBALANCE_CHOICES:
        raise ValueError(f"imbalance phải thuộc {config.IMBALANCE_CHOICES}")
    if imbalance == "weight" and spw is None:
        raise ValueError("imbalance='weight' cần spw=compute_scale_pos_weight(y_train)")
    steps = []
    if scale:   # cần scikit-learn >= 1.2 cho set_output
        steps.append(("scaler", ColumnTransformer(
            [("s", RobustScaler(), config.SCALE_COLS)],
            remainder="passthrough", verbose_feature_names_out=False,
        ).set_output(transform="pandas")))
    if imbalance == "smote":
        steps.append(("sampler", SMOTE(
            sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
            k_neighbors=config.SMOTE_K_NEIGHBORS, random_state=config.SEED)))
    elif imbalance == "under":
        steps.append(("sampler", RandomUnderSampler(
            sampling_strategy=config.UNDER_SAMPLING_STRATEGY, random_state=config.SEED)))
    spw_used = spw if imbalance == "weight" else 1.0   # smote/under/none luôn là 1
    steps.append(("clf", build_xgb(params, spw_used)))
    return Pipeline(steps)


def get_clf(pipe):
    return pipe.named_steps["clf"]


def transform_features(pipe, X):
    Xt = X
    for name, step in pipe.steps[:-1]:
        if name == "sampler":      # sampler chỉ chạy lúc fit, bỏ qua khi biến đổi
            continue
        Xt = step.transform(Xt)
    return Xt
```

**Lỗi hay gặp:**
- Dùng `sklearn.pipeline.Pipeline`: không nhận bước sampler.
- Gõ cứng `scale_pos_weight=577`: sai, giá trị đúng phải tính từ `y_train` (xấp xỉ 599,5).
- Dùng cả SMOTE lẫn `scale_pos_weight` (cân bằng hai lần). Code trên đã ép về 1 cho `smote`, `under`, `none`.
- Truyền `imbalance="weight"` mà quên `spw`: code sẽ báo lỗi ngay, không âm thầm chạy sai.

**Xong khi:** (chạy khi đã có `parquet` của P1)

```bash
python -c "from src.data import load_processed; from src.pipeline import *; X,_,y,_ = load_processed(); p = build_pipeline('weight', spw=compute_scale_pos_weight(y)); p.fit(X, y); print(p.predict_proba(X.head(3))[:,1])"
```
Chạy được, in ra 3 xác suất trong khoảng 0 đến 1. Thử thêm `imbalance` là `none`, `smote`, `under` và `scale=True`.

---

## D.5 `src/evaluate.py` (P2)

**Làm gì:** cross-validation, chỉ số, chọn ngưỡng, vẽ biểu đồ đánh giá. Đây là file **cả nhóm dùng**, nên hàm `run_cv` phải xong sớm.

**Các bước:**

1. `compute_metrics(y_true, y_proba, threshold)`: PR-AUC, Precision, Recall, F1 của lớp gian lận, và `tn, fp, fn, tp`.
2. `run_cv(pipeline, X, y, n_splits)`: vòng lặp `StratifiedKFold`, mỗi fold `clone` pipeline, fit, dự đoán, ghi xác suất out-of-fold. Trả `(summary, oof_proba)`.
3. `find_threshold(y_true, y_proba, strategy, target_recall)`.
4. `plot_pr_curve`, `plot_pr_curves`, `plot_confusion_matrix`.
5. `save_metrics(obj, name)`: ghi `results/metrics/<name>.csv`.

**Code gợi ý:**

```python
import time
import matplotlib
matplotlib.use("Agg")                       # chạy được không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold


def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"pr_auc": float(average_precision_score(y_true, y_proba)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run_cv(pipeline, X, y, n_splits=config.N_FOLDS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=config.CV_SHUFFLE,
                          random_state=config.SEED)
    oof = np.zeros(len(y))
    folds = {m: [] for m in config.CV_METRICS}
    fit_times = []
    for i, (tr, va) in enumerate(skf.split(X, y)):
        model = clone(pipeline)                       # mỗi fold một bản mới
        t0 = time.time()
        model.fit(X.iloc[tr], y.iloc[tr])
        fit_times.append(time.time() - t0)
        proba = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        m = compute_metrics(y.iloc[va], proba, config.DEFAULT_THRESHOLD)
        for k in config.CV_METRICS:
            folds[k].append(m[k])
        log.info("Fold %d: %d ca gian lận, PR-AUC=%.3f", i + 1, int(y.iloc[va].sum()), m["pr_auc"])
    summary = {}
    for k, v in folds.items():
        summary[f"{k}_mean"], summary[f"{k}_std"] = float(np.mean(v)), float(np.std(v))
    summary["fit_time_mean"] = float(np.mean(fit_times))
    return summary, oof


def find_threshold(y_true, y_proba, strategy="f1", target_recall=0.8):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec = prec[:-1], rec[:-1]                   # prec/rec dài hơn thr đúng 1 phần tử
    if strategy == "f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return float(thr[int(np.argmax(f1))])
    if strategy == "recall":                          # ngưỡng cao nhất vẫn đạt Recall mục tiêu
        ok = np.where(rec >= target_recall)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    raise ValueError(strategy)
```

Vẽ PR curve: dùng `precision_recall_curve`, thêm **đường nền ngang tại `y_true.mean()`** (tỷ lệ lớp dương, mức PR-AUC của mô hình ngẫu nhiên), ghi PR-AUC trong chú giải, lưu rồi `plt.close()`. Confusion matrix: tính từ `compute_metrics` (`tn, fp, fn, tp`) rồi vẽ bảng 2x2 bằng `plt.imshow` hoặc `seaborn.heatmap`.

`save_metrics(obj, name)`: nếu `obj` là `dict` thì `pd.DataFrame([obj])`, nếu là `DataFrame` thì giữ nguyên; ghi `config.METRICS_DIR / f"{name}.csv"` với `index=False`; trả về đường dẫn.

**Lỗi hay gặp:**
- Dùng `cross_validate` rồi `cross_val_predict` riêng: huấn luyện **hai lần**, rất tốn thời gian khi tuning. Dùng một vòng lặp như trên.
- Quên `clone(pipeline)`: các fold dùng chung một đối tượng, kết quả sai.
- `X`, `y` không có index mặc định thì `iloc` và nhãn lệch nhau; `load_processed()` phải trả `X`, `y` có index mặc định.
- Chọn ngưỡng trên **test**: cấm. Chỉ chọn trên out-of-fold.
- Dùng `plt.show()` hoặc quên `plt.close()` (tràn bộ nhớ khi vẽ nhiều hình).

**Xong khi:** `run_cv` trả đúng `(dict, mảng dài bằng len(y))`; dict có đủ khóa `pr_auc_mean, pr_auc_std, precision_*, recall_*, f1_*, fit_time_mean`; đẩy lên sớm (hạn: sáng ngày 2).

---

## D.6 `scripts/04_baseline_xgb.py` (P2)

**Làm gì:** mô hình XGBoost cơ sở (`imbalance="none"`, tham số cơ sở trong `config`), đánh giá bằng CV trên train.

**Các bước trong `main()`:**

1. `X_train, _, y_train, _ = load_processed()`, log số ca gian lận.
2. `pipe = build_pipeline("none")`
3. `summary, oof = run_cv(pipe, X_train, y_train)`
4. `save_metrics({"imbalance": "none", **summary}, "baseline")`
5. Ghi `results/predictions/xgb_baseline_oof.csv` với hai cột `y_true`, `y_proba`.
6. `plot_pr_curve(y_train, oof, config.FIGURES_DIR / "04_baseline_pr_curve.png", label="baseline")`
7. **Huấn luyện một mô hình trên toàn bộ train** và lưu `joblib.dump(pipe.fit(X_train, y_train), config.BASELINE_MODEL_PATH)` để P5 thử SHAP.

**Lỗi hay gặp:** để `imbalance="weight"` (baseline là mô hình **chưa xử lý mất cân bằng**); quên bước 7 nên P5 không có mô hình để thử.

**Xong khi:** `python scripts/04_baseline_xgb.py` ra `baseline.csv` (1 dòng), `xgb_baseline_oof.csv`, `04_baseline_pr_curve.png`, `xgb_baseline.joblib`.

---

## D.7 `scripts/07_evaluate.py` (P2)

**Làm gì:** huấn luyện mô hình cuối trên toàn bộ train, chọn ngưỡng bằng out-of-fold, đánh giá **test đúng một lần**.

**Tùy chọn dòng lệnh:** `--no-test`: chỉ tạo mô hình và ngưỡng, **không đụng tập test** (dùng khi P4 trễ `best_params.json`, hoặc để thử logic).

**Các bước trong `main()`:**

1. `X_train, X_test, y_train, y_test = load_processed()`
2. `res = load_params()` (đọc `best_params.json`: có `imbalance`, `scale`, `params`)
3. `spw = compute_scale_pos_weight(y_train) if res["imbalance"] == "weight" else None`
4. `pipe = build_pipeline(res["imbalance"], scale=res["scale"], params=res["params"], spw=spw)`
5. `summary, oof = run_cv(pipe, X_train, y_train)` rồi `thr = find_threshold(y_train, oof, config.THRESHOLD_STRATEGY, config.TARGET_RECALL)`; ghi `threshold.json` dạng `{"strategy": "f1", "threshold": 0.43}`
6. `pipe.fit(X_train, y_train)` rồi `joblib.dump(pipe, config.FINAL_MODEL_PATH)`
7. Nếu **không** có `--no-test`:
   - `proba = pipe.predict_proba(X_test)[:, 1]`
   - `compute_metrics` ở ngưỡng 0,5 **và** ở ngưỡng `thr`, ghi `final_test.csv` (2 dòng, cột `threshold_type, threshold, pr_auc, precision, recall, f1, tn, fp, fn, tp`)
   - ghi `results/predictions/xgb_final_test.csv` (cột `y_true`, `y_proba`)
   - vẽ `07_pr_curve.png` (có đường nền) và `07_confusion_matrix.png` (tại ngưỡng đã chọn)

**Lỗi hay gặp:**
- Chọn ngưỡng trên `proba` của test: **cấm tuyệt đối**.
- Chạy `07` đầy đủ nhiều lần với các tham số khác nhau rồi chọn lần đẹp nhất: biến test thành tập validation. Chỉ chạy đầy đủ **một lần** khi đã có `best_params.json` cuối cùng.
- Quên `scale=res["scale"]` và `spw`: pipeline lúc đánh giá khác lúc tuning.

**Xong khi:** `python scripts/07_evaluate.py` ra đủ `xgb_final.joblib`, `threshold.json`, `xgb_final_test.csv`, `final_test.csv`, 2 hình `07_*.png`.

---

## D.8 `src/compare.py` (P3)

**Làm gì:** so sánh công bằng 4 cách xử lý mất cân bằng và chọn một cách. Không tự dựng pipeline hay CV riêng, chỉ gọi `build_pipeline` và `run_cv` của P2.

**Các bước:**

1. `compare_variants(X, y, variants)`: với mỗi biến thể dựng pipeline, `run_cv`, thu `summary` và `oof`.
2. `choose_imbalance(df)`: chọn cách tốt nhất, trả `{"imbalance": ..., "reason": ...}`.
3. `plot_variant_comparison(df, path)`: biểu đồ cột PR-AUC trung bình với thanh sai số là độ lệch chuẩn.

**Code gợi ý:**

```python
PRIORITY = ["weight", "none", "under", "smote"]   # thứ tự ưu tiên khi các cách ngang nhau


def compare_variants(X, y, variants=config.IMBALANCE_CHOICES):
    spw = compute_scale_pos_weight(y)
    rows, oofs = [], {}
    for v in variants:
        pipe = build_pipeline(imbalance=v, spw=spw if v == "weight" else None)
        summary, oof = run_cv(pipe, X, y)
        rows.append({"imbalance": v, **summary})
        oofs[v] = oof
        log.info("%s: PR-AUC = %.3f ± %.3f", v, summary["pr_auc_mean"], summary["pr_auc_std"])
    return pd.DataFrame(rows), oofs


def choose_imbalance(df):
    best = df.loc[df["pr_auc_mean"].idxmax()]
    close = df[df["pr_auc_mean"] >= best["pr_auc_mean"] - best["pr_auc_std"]]
    chosen = min(close["imbalance"], key=PRIORITY.index)
    reason = (f"PR-AUC cao nhất: {best['imbalance']} ({best['pr_auc_mean']:.3f}). "
              f"Các cách chênh nhỏ hơn 1 độ lệch chuẩn ({best['pr_auc_std']:.3f}) coi là "
              f"tương đương: {list(close['imbalance'])}. Chọn {chosen} theo thứ tự ưu tiên "
              f"{PRIORITY} (đơn giản hơn khi ngang nhau).")
    return {"imbalance": chosen, "reason": reason}
```

Quy tắc chọn: lấy cách có PR-AUC trung bình cao nhất; những cách chênh **nhỏ hơn một độ lệch chuẩn** thì coi là tương đương, và chọn cách **đơn giản hơn** theo `PRIORITY`. Cả nhóm có thể đổi `PRIORITY`, nhưng phải ghi rõ lý do vào báo cáo.

**Lỗi hay gặp:**
- Mỗi biến thể một tham số XGBoost khác nhau: so sánh **không công bằng**. Chỉ thay `imbalance`.
- Quên `spw` cho `weight` (code sẽ báo lỗi).
- Chọn theo Recall hoặc F1 ở ngưỡng 0,5: chỉ số chính là **PR-AUC**.

**Xong khi:** `compare_variants` trả bảng 4 dòng và `dict` 4 mảng xác suất; `choose_imbalance` trả `dict` có trường `reason` dễ đọc.

---

## D.9 `scripts/05_imbalance_compare.py` (P3)

**Làm gì:** chạy `compare_variants`, lưu toàn bộ kết quả, ra quyết định.

**Mẫu script đầy đủ (dùng làm khuôn cho các script khác):**

```python
"""05_imbalance_compare.py: so sánh 4 cách xử lý mất cân bằng với XGBoost.

Đọc : data/processed/train.parquet
Ghi : results/metrics/imbalance_compare.csv, results/predictions/xgb_<biến thể>_oof.csv,
      results/params/chosen_imbalance.json, figures/05_*.png
"""
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # để import được src/

import pandas as pd

from src import config
from src.compare import choose_imbalance, compare_variants, plot_variant_comparison
from src.data import load_processed
from src.evaluate import plot_pr_curves, save_metrics

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
log = logging.getLogger(__name__)


def main() -> None:
    config.ensure_dirs()
    X_train, _, y_train, _ = load_processed()
    log.info("Train: %d dòng, %d ca gian lận", len(y_train), int(y_train.sum()))

    df, oofs = compare_variants(X_train, y_train)
    save_metrics(df, "imbalance_compare")
    for v, oof in oofs.items():
        pd.DataFrame({"y_true": y_train.values, "y_proba": oof}).to_csv(
            config.PREDICTIONS_DIR / f"xgb_{v}_oof.csv", index=False)

    plot_variant_comparison(df, config.FIGURES_DIR / "05_imbalance_compare.png")
    plot_pr_curves({v: (y_train, o) for v, o in oofs.items()},
                   config.FIGURES_DIR / "05_pr_curves.png")

    choice = choose_imbalance(df)
    config.CHOSEN_IMBALANCE_PATH.write_text(json.dumps(choice, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    log.info("Chọn: %s | %s", choice["imbalance"], choice["reason"])


if __name__ == "__main__":
    main()
```

**Lỗi hay gặp:** quên in thời gian: SMOTE chậm hơn các cách khác, **ghi lại thời gian từng biến thể** (đã có `fit_time_mean`). Tính tổng thời gian chạy một lần, không chạy khi đang cần dùng máy.

**Xong khi:** `python scripts/05_imbalance_compare.py` ra `imbalance_compare.csv` (4 dòng), 4 file `xgb_<biến thể>_oof.csv`, 2 hình `05_*.png`, và `chosen_imbalance.json` có trường `reason`.

---

## D.10 `src/tune.py` (P4)

**Làm gì:** tìm tham số XGBoost tốt nhất bằng Optuna, mục tiêu là PR-AUC trung bình trên CV.

**Các bước:**

1. `_suggest(trial)`: khai báo không gian tìm kiếm 8 tham số.
2. `run_tuning(X, y, n_trials, imbalance, seed)`: dựng `objective` gọi `build_pipeline` và `run_cv`, chạy Optuna, trả `(result, study)`.
3. `save_params(result, path)` và `load_params(path)`: ghi/đọc JSON.
4. `plot_optimization_history(study, path)`: vẽ PR-AUC theo từng vòng và đường tốt nhất tới thời điểm đó.

**Code gợi ý:**

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)      # bớt in log mỗi vòng


def _suggest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def run_tuning(X, y, n_trials=config.N_TRIALS, imbalance="weight", seed=config.SEED):
    spw = compute_scale_pos_weight(y) if imbalance == "weight" else None

    def objective(trial):
        pipe = build_pipeline(imbalance, params=_suggest(trial), spw=spw)
        summary, _ = run_cv(pipe, X, y)
        trial.set_user_attr("pr_auc_std", summary["pr_auc_std"])
        return summary["pr_auc_mean"]

    study = optuna.create_study(direction=config.OPTUNA_DIRECTION,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    result = {"imbalance": imbalance, "scale": False, "params": best.params,
              "cv_pr_auc_mean": best.value, "cv_pr_auc_std": best.user_attrs["pr_auc_std"],
              "n_trials": n_trials, "seed": seed}
    return result, study
```

Đã kiểm tra: chạy hai lần cùng `seed` cho **cùng bộ tham số** (tái lập được).

**Lỗi hay gặp:**
- Đặt `n_jobs` của Optuna lớn hơn 1: XGBoost đã dùng hết lõi CPU (`n_jobs=-1`), chạy song song chỉ làm chậm.
- Dùng tập test hoặc `early_stopping` với `eval_set` lấy từ test: **cấm**.
- Chạy 50 vòng ngay khi chưa biết một vòng mất bao lâu. **Chạy thử 3-5 vòng và đo thời gian trước.**
- `best_params.json` thiếu khóa `imbalance` hoặc `scale`: P2 sẽ không đọc được.

**Xong khi:** thử 3-5 vòng chạy được, `result` có đủ các khóa `imbalance, scale, params, cv_pr_auc_mean, cv_pr_auc_std, n_trials, seed`.

---

## D.11 `scripts/06_tune_xgb.py` (P4)

**Làm gì:** chạy tuning chính thức, lưu kết quả và so sánh trước/sau tuning.

**Các bước trong `main()`:**

1. `X_train, _, y_train, _ = load_processed()`
2. Đọc `chosen_imbalance.json`. Nếu **chưa có** thì dùng `config.DEFAULT_IMBALANCE` (`weight`) và **in cảnh báo rõ ràng**.
3. `result, study = run_tuning(X_train, y_train, n_trials=config.N_TRIALS, imbalance=imb)`
4. `save_params(result)` (ghi `best_params.json`)
5. `study.trials_dataframe().to_csv(config.METRICS_DIR / "tuning_trials.csv", index=False)`
6. **So sánh trước/sau**: chạy `run_cv` với tham số mặc định cho **cùng `imbalance`** (`build_pipeline(imb, spw=...)`), so với kết quả đã tuning; thêm một dòng baseline (đọc `baseline.csv` nếu có). Ghi `tuning_compare.csv` với cột: `setting, pr_auc_mean, pr_auc_std, improvement_pct`.
7. `plot_optimization_history(study, config.FIGURES_DIR / "06_optimization_history.png")`

**Lỗi hay gặp:** so sánh tuned với baseline `none` rồi kết luận "tuning giúp tăng X%": phần tăng lẫn cả hiệu quả của xử lý mất cân bằng. So với **cùng `imbalance` ở tham số mặc định** mới đúng. Điểm tốt nhất của Optuna là giá trị **lớn nhất** trong nhiều lần thử nên hơi lạc quan; con số chính thức là điểm trên test ở `07`.

**Xong khi:** `python scripts/06_tune_xgb.py` ra `best_params.json` (đủ khóa), `tuning_trials.csv`, `tuning_compare.csv`, `06_optimization_history.png`.

---

## D.12 `scripts/03_eda_features.py` (P5)

**Làm gì:** phân tích V1-V28, tương quan và xu hướng (mục 5.4-5.6). Dùng toàn bộ dữ liệu đã làm sạch, **chỉ để mô tả**, không dùng để loại cột khỏi mô hình.

**Các bước trong `main()` (vẽ thẳng trong `main()`, xem luật A.1.2):**

1. `df = load_processed_full()`
2. `corr = df[config.FEATURE_COLS + [config.TARGET_COL]].corr()[config.TARGET_COL].drop(config.TARGET_COL)`; lưu `eda_corr_with_class.csv`.
3. **`03_correlation_matrix.png`**: `seaborn.heatmap` của ma trận tương quan 31x31 (`cmap="coolwarm"`, `center=0`, không in số).
4. **`03_top_corr_features.png`**: biểu đồ cột ngang của 15 đặc trưng có |tương quan với `Class`| lớn nhất.
5. **`03_feature_dist_top.png`**: với `config.EDA_TOP_FEATURES` đặc trưng đầu bảng trên, vẽ phân bố hai lớp chồng nhau (`sns.histplot(..., hue="Class", stat="density", common_norm=False, element="step")`). **Ghi rõ tiêu chí chọn đặc trưng để vẽ** (|tương quan|) và rằng đây chỉ là minh họa.
6. **`03_fraud_vs_time.png`**: `hour = (df["Time"] // 3600).astype(int)`; nhóm theo `hour`, vẽ số giao dịch và **tỷ lệ gian lận** theo giờ (hai trục y).
7. **`03_fraud_vs_amount.png`**: chia `Amount` thành các khoảng (ví dụ `pd.cut(df["Amount"], [-0.01, 1, 10, 50, 100, 500, 1000, np.inf])`), vẽ tỷ lệ gian lận theo khoảng.

Hai hình `03_feature_dist_top.png` và `03_fraud_vs_amount.png` là phần **bổ sung** so với danh sách ban đầu, để đủ nội dung mục 5.4 và 5.6 của báo cáo.

**Lỗi hay gặp:** tính tỷ lệ gian lận theo giờ trên nhóm quá ít giao dịch nên rất nhiễu (**ghi số giao dịch của từng giờ**); dùng kết quả tương quan để xóa cột trong mô hình (cấm).

**Xong khi:** `python scripts/03_eda_features.py` ra đủ 5 hình và `eda_corr_with_class.csv`.

---

## D.13 `src/explain.py` (P5)

**Làm gì:** SHAP và phân tích các ca đoán sai. Không huấn luyện lại, không tune, không chọn ngưỡng.

**Các bước:**

1. `compute_shap(pipe, X, y, sample_size, seed)`: **lấy mẫu con giữ toàn bộ ca gian lận**, đưa qua `transform_features` rồi `TreeExplainer`.
2. `plot_shap_summary` và `plot_shap_bar`.
3. `get_errors(X, y_true, y_proba, threshold)`: gán nhãn `TP/TN/FP/FN`.
4. `summarize_errors(df)`: thống kê theo loại.

**Code gợi ý:**

```python
import shap
import numpy as np
import pandas as pd


def compute_shap(pipe, X, y, sample_size=config.SHAP_SAMPLE_SIZE, seed=config.SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y.to_numpy() == 1)[0]                       # giữ toàn bộ ca gian lận
    neg = np.where(y.to_numpy() == 0)[0]
    n_neg = min(max(sample_size - len(pos), 0), len(neg))
    idx = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    X_t = transform_features(pipe, X.iloc[idx])                # đặc trưng sau tiền xử lý
    values = shap.TreeExplainer(get_clf(pipe)).shap_values(X_t)
    return values, X_t


def plot_shap_summary(values, X_t, path):
    shap.summary_plot(values, X_t, show=False, max_display=config.SHAP_TOP_N)
    plt.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close()                       # plot_type="bar" cho bản cột (plot_shap_bar)


def get_errors(X, y_true, y_proba, threshold):
    df = X.copy()
    df.insert(0, "y_true", np.asarray(y_true))
    df.insert(1, "y_proba", np.asarray(y_proba))
    df.insert(2, "y_pred", (df["y_proba"] >= threshold).astype(int))
    df.insert(3, "error_type", np.select(
        [(df.y_true == 1) & (df.y_pred == 1),
         (df.y_true == 0) & (df.y_pred == 0),
         (df.y_true == 0) & (df.y_pred == 1)],
        ["TP", "TN", "FP"], default="FN"))
    return df


def summarize_errors(df):
    return (df.groupby("error_type")
              .agg(n=("y_true", "size"), proba_mean=("y_proba", "mean"),
                   amount_median=(config.AMOUNT_COL, "median"))
              .reset_index())
```

**Lỗi hay gặp:**
- Tính SHAP trên toàn bộ ~56.746 dòng test: chậm và tốn RAM. Dùng mẫu ~5.000 dòng.
- Mẫu giữ toàn bộ ca gian lận nên **tỷ lệ gian lận trong mẫu cao hơn thật**: biểu đồ SHAP không phải phân bố của toàn bộ dữ liệu. **Phải ghi rõ trong báo cáo.**
- Đưa dữ liệu thô vào SHAP khi pipeline có scaler: phải dùng `transform_features`.
- Diễn giải V1-V28 như có ý nghĩa nghiệp vụ: chỉ nói đặc trưng nào quan trọng (chúng đã ẩn danh qua PCA).

**Xong khi:** `compute_shap` trả `values.shape == X_t.shape`; `get_errors` có cột `error_type` đúng bốn loại; thử trên `xgb_baseline.joblib` chạy được.

---

## D.14 `scripts/08_explain.py` (P5)

**Làm gì:** giải thích mô hình cuối và phân tích lỗi trên test.

**Các bước trong `main()`:**

1. `_, X_test, _, y_test = load_processed()`
2. `pipe = joblib.load(config.FINAL_MODEL_PATH)`
3. Đọc ngưỡng từ `config.THRESHOLD_PATH`. Nếu **chưa có** thì dùng `config.DEFAULT_THRESHOLD` (0,5) và **in cảnh báo**.
4. `proba = pipe.predict_proba(X_test)[:, 1]`
5. `errors = get_errors(X_test, y_test, proba, thr)`; ghi:
   - `results/metrics/errors.csv`: **bản gọn của mọi dòng test** với các cột `y_true, y_proba, y_pred, error_type, Time, Amount` (đủ bốn loại TP, TN, FP, FN, nhẹ cho Git)
   - `results/metrics/errors_summary.csv`: `summarize_errors(errors)`
6. `values, X_t = compute_shap(pipe, X_test, y_test)`; vẽ `08_shap_summary.png` và `08_shap_bar.png`.
7. In ra: số ca FN và FP, `Amount` trung vị của từng nhóm (để viết mục 6.7).

**Lỗi hay gặp:** huấn luyện lại hay tune trong script này (cấm); đổi ngưỡng ở `08` (ngưỡng do `07` quyết định).

**Xong khi:** `python scripts/08_explain.py` ra `errors.csv` (đủ 4 loại), `errors_summary.csv`, 2 hình SHAP; số dòng `errors.csv` bằng số dòng test.

*Khi phát triển (ngày 3):* đổi tạm `FINAL_MODEL_PATH` thành `BASELINE_MODEL_PATH` trong **bản nháp của bạn**, ngưỡng 0,5. Nhớ đổi lại trước khi mở Pull Request.

---

## D.15 `scripts/run_all.py` (P5)

**Làm gì:** chạy tuần tự `01` đến `08`, dừng ngay khi có bước lỗi.

**Các bước:**

1. `STEPS` là danh sách tên 8 script theo thứ tự.
2. Nhận tùy chọn `--skip` (ví dụ `--skip 05 06` bỏ qua các script bắt đầu bằng `05` và `06`).
3. Với mỗi script: đo thời gian, chạy bằng `subprocess.run([sys.executable, path], cwd=config.ROOT)`, nếu mã thoát khác 0 thì log lỗi và dừng ngay.
4. Cuối cùng in bảng thời gian từng bước.

**Code gợi ý:**

```python
import argparse, subprocess, sys, time

STEPS = ["01_preprocess.py", "02_eda_overview.py", "03_eda_features.py",
         "04_baseline_xgb.py", "05_imbalance_compare.py", "06_tune_xgb.py",
         "07_evaluate.py", "08_explain.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="số đầu tên script cần bỏ qua, ví dụ 05 06")
    skip = set(ap.parse_args().skip)
    times = []
    for s in STEPS:
        if s[:2] in skip:
            log.info("BỎ QUA %s", s); continue
        t0 = time.time()
        r = subprocess.run([sys.executable, str(config.SCRIPTS_DIR / s)], cwd=config.ROOT)
        if r.returncode != 0:
            log.error("%s lỗi (mã %d), dừng.", s, r.returncode)
            sys.exit(r.returncode)
        times.append((s, time.time() - t0))
    for s, t in times:
        log.info("%-28s %8.1f giây", s, t)
```

**Lỗi hay gặp:** chạy `run_all.py` đầy đủ trong lúc thử nghiệm: bước `06` có thể mất hàng giờ. Khi kiểm tra tích hợp ở ngày 7 hãy dùng `--skip 05 06` nếu kết quả đã có.

**Xong khi:** `python scripts/run_all.py` (hoặc kèm `--skip`) chạy trọn từ đầu đến cuối, không lỗi.

---

## D.16 `src/config.py` (P2, đã viết sẵn)

**Làm gì:** chỉ chứa hằng số và hàm `ensure_dirs()`. Mọi người **chỉ đọc**, không sửa.

- Dùng: `from src import config` rồi `config.SEED`, `config.TRAIN_PATH`, `config.IMBALANCE_CHOICES`, ...
- Không import `pandas`, `sklearn`, `xgboost` trong file này.
- Cần thêm hằng số: nhắn P2.

---

# PHỤ LỤC: CHỮ KÝ HÀM VÀ LƯỢC ĐỒ FILE

> Đây là **hợp đồng** giữa các người. Muốn đổi chữ ký hàm hay tên file thì phải báo cả nhóm trước.

## 1. Chữ ký hàm

```python
# src/data.py (P1)
def fetch_raw() -> Path                                  # tải OpenML, lưu RAW_CSV (bỏ qua nếu đã có)
def load_raw() -> pd.DataFrame                           # đọc RAW_CSV
def clean(df) -> pd.DataFrame                            # ép kiểu, kiểm tra thiếu, XÓA TRÙNG cả 31 cột
def load_and_clean() -> pd.DataFrame
def split_data(df) -> tuple[X_train, X_test, y_train, y_test]
def summarize_preprocess(df_raw, df_clean, y_train, y_test) -> dict
def save_processed(X_train, X_test, y_train, y_test) -> None
def load_processed() -> tuple[X_train, X_test, y_train, y_test]
def load_processed_full() -> pd.DataFrame                # train + test nối lại, đủ 30 đặc trưng + Class

# src/pipeline.py (P2)
def compute_scale_pos_weight(y) -> float
def build_xgb(params=None, scale_pos_weight=1.0) -> XGBClassifier
def build_pipeline(imbalance="weight", scale=False, params=None, spw=None) -> imblearn Pipeline
#   imbalance ∈ {"none","weight","smote","under"}; "weight" BẮT BUỘC truyền spw
#   tên bước: "scaler" (nếu scale), "sampler" (nếu smote/under), "clf"
def get_clf(pipe) -> XGBClassifier
def transform_features(pipe, X) -> DataFrame             # bỏ qua bước sampler

# src/evaluate.py (P2)
def run_cv(pipeline, X, y, n_splits=N_FOLDS) -> tuple[dict, np.ndarray]
#   dict: {pr_auc,precision,recall,f1}_{mean,std} + fit_time_mean ; oof_proba dài len(y)
def compute_metrics(y_true, y_proba, threshold=0.5) -> dict   # pr_auc, precision, recall, f1, tn, fp, fn, tp
def find_threshold(y_true, y_proba, strategy="f1", target_recall=0.8) -> float
def plot_pr_curve(y_true, y_proba, path, label="model") -> None
def plot_pr_curves(curves: dict, path) -> None           # {tên: (y_true, y_proba)}
def plot_confusion_matrix(y_true, y_proba, threshold, path) -> None
def save_metrics(obj, name) -> Path                      # dict/DataFrame -> METRICS_DIR/{name}.csv

# src/compare.py (P3)
def compare_variants(X, y, variants=IMBALANCE_CHOICES) -> tuple[pd.DataFrame, dict]
def choose_imbalance(df) -> dict                         # {"imbalance": ..., "reason": ...}
def plot_variant_comparison(df, path) -> None

# src/tune.py (P4)
def run_tuning(X, y, n_trials=N_TRIALS, imbalance="weight", seed=SEED) -> tuple[dict, optuna.Study]
def save_params(result, path=BEST_PARAMS_PATH) -> None
def load_params(path=BEST_PARAMS_PATH) -> dict
def plot_optimization_history(study, path) -> None

# src/explain.py (P5)
def compute_shap(pipe, X, y, sample_size=SHAP_SAMPLE_SIZE, seed=SEED) -> tuple[values, DataFrame]
def plot_shap_summary(values, X_t, path) -> None
def plot_shap_bar(values, X_t, path) -> None
def get_errors(X, y_true, y_proba, threshold) -> pd.DataFrame   # cột y_true,y_proba,y_pred,error_type + đặc trưng
def summarize_errors(df) -> pd.DataFrame
```

## 2. Lược đồ dữ liệu

- `X_*`: DataFrame **30 cột** theo `config.FEATURE_COLS` (`Time`, `V1`...`V28`, `Amount`), index mặc định `0..n-1`.
- `y_*`: `pd.Series` kiểu `int` (0/1), tên `Class`, index mặc định.
- `train.parquet`, `test.parquet`: 30 cột đặc trưng cộng cột `Class`.

## 3. Các file trao đổi giữa các người

| File | Người ghi | Người đọc | Nội dung |
|---|---|---|---|
| `data/processed/train.parquet`, `test.parquet` | P1 (01) | P2, P3, P4, P5 | 30 cột đặc trưng + `Class` |
| `results/metrics/preprocess_summary.csv` | P1 (01) | báo cáo | `n_raw, n_dup, n_clean, fraud_raw, fraud_clean, train_n, train_fraud, test_n, test_fraud` |
| `results/metrics/baseline.csv` | P2 (04) | P4, báo cáo | 1 dòng: `imbalance` + các khóa của `run_cv` |
| `results/predictions/xgb_baseline_oof.csv` | P2 (04) | báo cáo | cột `y_true, y_proba` |
| `results/models/xgb_baseline.joblib` | P2 (04) | P5 (thử SHAP) | pipeline đã fit trên toàn bộ train |
| `results/metrics/imbalance_compare.csv` | P3 (05) | báo cáo | mỗi dòng một biến thể: `imbalance` + các khóa của `run_cv` |
| `results/predictions/xgb_<biến thể>_oof.csv` | P3 (05) | báo cáo | cột `y_true, y_proba` |
| `results/params/chosen_imbalance.json` | P3 (05) | P4 (06) | `{"imbalance": "weight", "reason": "..."}` |
| `results/params/best_params.json` | P4 (06) | P2 (07) | `{"imbalance", "scale", "params": {...}, "cv_pr_auc_mean", "cv_pr_auc_std", "n_trials", "seed"}` |
| `results/metrics/tuning_trials.csv`, `tuning_compare.csv` | P4 (06) | báo cáo | lịch sử các vòng; so sánh trước/sau tuning |
| `results/params/threshold.json` | P2 (07) | P5 (08) | `{"strategy": "f1", "threshold": 0.43}` |
| `results/models/xgb_final.joblib` | P2 (07) | P5 (08) | pipeline đã fit, tạo bởi `build_pipeline` |
| `results/predictions/xgb_final_test.csv` | P2 (07) | báo cáo | cột `y_true, y_proba` (tập test) |
| `results/metrics/final_test.csv` | P2 (07) | báo cáo | 2 dòng: ngưỡng 0,5 và ngưỡng đã chọn |
| `results/metrics/errors.csv`, `errors_summary.csv` | P5 (08) | báo cáo | bản gọn mọi dòng test với `error_type`; thống kê theo loại |
| `results/metrics/eda_*.csv` | P1 (02), P5 (03) | báo cáo | thống kê mô tả, đếm nhãn, tương quan với `Class` |

## 4. Thư viện cần cài (`requirements.txt`)

`pandas`, `numpy`, `scikit-learn` (từ 1.2 trở lên), `imbalanced-learn`, `xgboost`, `optuna`, `shap`, `matplotlib`, `seaborn`, `joblib`, `pyarrow`.

**Ngày 1:** cả nhóm cài thử, chạy được `import` tất cả, rồi P2 **ghim phiên bản (`==`)** vào `requirements.txt` để các máy giống nhau.
