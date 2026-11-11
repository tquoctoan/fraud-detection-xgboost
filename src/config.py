"""src/config.py — Hằng số, seed, đường dẫn, tham số mặc định.

Mọi file khác chỉ đọc file này, không sửa.
Cách dùng:
    from src import config
    config.SEED, config.TRAIN_PATH, config.IMBALANCE_CHOICES, ...

Luật:
    - Không import pandas, sklearn, xgboost trong file này.
    - Không có logic (vòng lặp, điều kiện) ngoài hàm ensure_dirs().
    - Mọi đường dẫn đều dẫn xuất từ ROOT.
"""
from __future__ import annotations

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Gốc repo
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # d:/ptdll/fraud-detection-xgboost

# ---------------------------------------------------------------------------
# Seed & tái lập
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dữ liệu OpenML
# ---------------------------------------------------------------------------
OPENML_ID           = 1597
OPENML_FETCH_TARGET = "Time"   # target_column="Time" để giữ đủ 31 cột

TARGET_COL = "Class"
AMOUNT_COL = "Amount"

# 30 cột đặc trưng: Time + V1..V28 + Amount
FEATURE_COLS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Hai cột cần scale khi bật scale=True trong build_pipeline
SCALE_COLS: list[str] = ["Time", "Amount"]

# ---------------------------------------------------------------------------
# Đường dẫn thư mục
# ---------------------------------------------------------------------------
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
RESULTS         = ROOT / "results"
METRICS_DIR     = RESULTS / "metrics"
PARAMS_DIR      = RESULTS / "params"
PREDICTIONS_DIR = RESULTS / "predictions"
MODELS_DIR      = RESULTS / "models"
FIGURES_DIR     = ROOT / "figures"
SCRIPTS_DIR     = ROOT / "scripts"

# ---------------------------------------------------------------------------
# Đường dẫn file cụ thể
# ---------------------------------------------------------------------------
RAW_CSV               = DATA_RAW / "creditcard.csv"
TRAIN_PATH            = DATA_PROCESSED / "train.parquet"
TEST_PATH             = DATA_PROCESSED / "test.parquet"

BEST_PARAMS_PATH      = PARAMS_DIR / "best_params.json"
CHOSEN_IMBALANCE_PATH = PARAMS_DIR / "chosen_imbalance.json"
THRESHOLD_PATH        = PARAMS_DIR / "threshold.json"

FINAL_MODEL_PATH      = MODELS_DIR / "xgb_final.joblib"
BASELINE_MODEL_PATH   = MODELS_DIR / "xgb_baseline.joblib"

# ---------------------------------------------------------------------------
# Chia train / test
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2   # 20 % dùng làm test
N_FOLDS   = 5     # số fold trong StratifiedKFold

# ---------------------------------------------------------------------------
# Số kiểm chứng (dùng để cảnh báo khi số liệu lệch)
# ---------------------------------------------------------------------------
EXPECTED_N_RAW       = 284_807   # tổng dòng file gốc
EXPECTED_N_CLEAN     = 283_726   # sau khi xóa 1.081 dòng trùng
EXPECTED_FRAUD_RAW   = 492       # ca gian lận trong file gốc
EXPECTED_FRAUD_CLEAN = 473       # ca gian lận sau khi làm sạch

# ---------------------------------------------------------------------------
# Tham số XGBoost cố định (không tuning)
# ---------------------------------------------------------------------------
XGB_FIXED_PARAMS: dict = {
    "tree_method"      : "hist",    # nhanh hơn "exact" trên dữ liệu lớn
    "eval_metric"      : "aucpr",   # PR-AUC — phù hợp dữ liệu mất cân bằng
    "use_label_encoder": False,     # tắt cảnh báo cũ
    "verbosity"        : 0,         # không in log của XGBoost
    "n_jobs"           : -1,        # dùng hết CPU
    "random_state"     : SEED,
}

# Tham số cơ sở (giá trị mặc định, dùng cho baseline và khởi đầu tuning)
XGB_BASE_PARAMS: dict = {
    "n_estimators"    : 300,
    "max_depth"       : 6,
    "learning_rate"   : 0.1,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma"           : 0.0,
    "reg_lambda"      : 1.0,
}

# ---------------------------------------------------------------------------
# Xử lý mất cân bằng
# ---------------------------------------------------------------------------
IMBALANCE_CHOICES = ("none", "weight", "smote", "under")
DEFAULT_IMBALANCE = "weight"           # dùng khi chosen_imbalance.json chưa có

# SMOTE
SMOTE_SAMPLING_STRATEGY = "auto"      # làm cho lớp thiểu số bằng lớp đa số
SMOTE_K_NEIGHBORS        = 5

# Random under-sampling
UNDER_SAMPLING_STRATEGY = "auto"

# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
N_TRIALS         = 50
OPTUNA_DIRECTION = "maximize"   # tối đa PR-AUC

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_SHUFFLE  = True
CV_METRICS: list[str] = ["pr_auc", "precision", "recall", "f1"]

# ---------------------------------------------------------------------------
# Chọn ngưỡng phân loại
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD  = 0.5
THRESHOLD_STRATEGY = "f1"    # chiến lược mặc định: tối đa F1
TARGET_RECALL      = 0.8    # dùng khi strategy="recall"

# ---------------------------------------------------------------------------
# Vẽ hình
# ---------------------------------------------------------------------------
FIG_DPI          = 150
EDA_TOP_FEATURES = 10   # số đặc trưng hiển thị trong hình phân bố

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
SHAP_SAMPLE_SIZE = 5_000   # ~5.000 dòng (giữ toàn bộ ca gian lận)
SHAP_TOP_N       = 20      # số đặc trưng hiển thị trong summary plot

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL  = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# Hàm tạo thư mục
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Tạo tất cả thư mục cần thiết (gọi ở đầu mỗi script)."""
    for d in (
        DATA_RAW,
        DATA_PROCESSED,
        METRICS_DIR,
        PARAMS_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
