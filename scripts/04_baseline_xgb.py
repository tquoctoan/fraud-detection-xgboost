"""04_baseline_xgb.py — Huấn luyện mô hình XGBoost cơ sở (Baseline chưa xử lý mất cân bằng).

Đầu vào:
  - data/processed/train.parquet

Đầu ra:
  - results/metrics/baseline.csv
  - results/predictions/xgb_baseline_oof.csv
  - figures/04_baseline_pr_curve.png
  - results/models/xgb_baseline.joblib
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src import config
from src.data import load_processed
from src.evaluate import plot_pr_curve, run_cv, save_metrics
from src.pipeline import build_pipeline

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
log = logging.getLogger(__name__)


def main() -> None:
    config.ensure_dirs()
    log.info("========== BẮT ĐẦU HUẤN LUYỆN BASELINE XGBOOST ==========")

    X_train, _, y_train, _ = load_processed()
    log.info("Train: %d dòng, %d ca gian lận (%.4f%%)", len(y_train), int(y_train.sum()), (y_train.sum() / len(y_train)) * 100)

    # Khởi tạo pipeline baseline (imbalance="none")
    pipe = build_pipeline("none")

    # Đánh giá bằng 5-Fold Cross-Validation trên tập Train
    summary, oof = run_cv(pipe, X_train, y_train)
    save_metrics({"imbalance": "none", **summary}, "baseline")

    # Lưu kết quả dự đoán out-of-fold
    pd.DataFrame({"y_true": y_train.values, "y_proba": oof}).to_csv(
        config.PREDICTIONS_DIR / "xgb_baseline_oof.csv", index=False
    )

    # Vẽ đường Precision-Recall curve
    plot_pr_curve(
        y_train,
        oof,
        config.FIGURES_DIR / "04_baseline_pr_curve.png",
        label="XGBoost Baseline (OOF)",
    )

    # Huấn luyện mô hình trên toàn bộ tập Train và lưu file joblib
    log.info("Huấn luyện mô hình trên toàn bộ train và lưu tại: %s", config.BASELINE_MODEL_PATH)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, config.BASELINE_MODEL_PATH)

    log.info("Baseline hoàn thành: PR-AUC = %.4f ± %.4f | F1 = %.4f ± %.4f",
             summary["pr_auc_mean"], summary["pr_auc_std"], summary["f1_mean"], summary["f1_std"])


if __name__ == "__main__":
    main()
