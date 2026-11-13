"""Baseline XGBoost: no imbalance handling, evaluated with 5-fold CV on train."""
import logging
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_processed
from src.evaluate import plot_pr_curve, run_cv, save_metrics
from src.pipeline import build_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    config.ensure_dirs()

    X_train, _, y_train, _ = load_processed()
    logger.info(
        "Train: %d dòng, %d ca gian lận (%.3f%%)",
        len(y_train),
        int(y_train.sum()),
        100 * float(y_train.mean()),
    )

    pipe = build_pipeline("none")
    summary, oof = run_cv(pipe, X_train, y_train)

    metrics_path = save_metrics({"imbalance": "none", **summary}, "baseline")
    predictions_path = config.PREDICTIONS_DIR / "xgb_baseline_oof.csv"
    pd.DataFrame({"y_true": y_train.values, "y_proba": oof}).to_csv(
        predictions_path, index=False
    )
    figure_path = plot_pr_curve(
        y_train, oof, config.FIGURES_DIR / "04_baseline_pr_curve.png", label="baseline"
    )

    # Refit on the whole train split so P5 can try SHAP before the tuned model exists.
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, config.BASELINE_MODEL_PATH)

    logger.info("Đã lưu mô hình baseline vào %s", config.BASELINE_MODEL_PATH)
    console = pd.DataFrame(
        [
            {
                "Chỉ số": name,
                "Trung bình": summary[f"{name}_mean"],
                "Độ lệch chuẩn": summary[f"{name}_std"],
            }
            for name in config.CV_METRICS
        ]
    )
    print(
        f"Baseline (imbalance='none', {config.N_FOLDS} fold) — "
        f"thời gian fit trung bình {summary['fit_time_mean']:.1f}s/fold"
    )
    print(console.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Đã lưu chỉ số: {metrics_path}")
    print(f"Đã lưu xác suất out-of-fold: {predictions_path}")
    print(f"Đã lưu biểu đồ: {figure_path}")


if __name__ == "__main__":
    main()