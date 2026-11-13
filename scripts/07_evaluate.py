"""Final XGBoost model: OOF threshold selection and a single test evaluation.

Usage:
    python scripts/07_evaluate.py            # full run, touches the test split once
    python scripts/07_evaluate.py --no-test  # build model + threshold only
"""
import argparse
import json
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
from src.evaluate import (
    compute_metrics,
    find_threshold,
    plot_confusion_matrix,
    plot_pr_curve,
    run_cv,
    save_metrics,
)
from src.pipeline import build_pipeline, compute_scale_pos_weight

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Chỉ tạo mô hình và ngưỡng, không đọc hay đánh giá tập test.",
    )
    return parser.parse_args()


def load_params(path: Path = config.BEST_PARAMS_PATH) -> dict:
    """Read the tuned settings produced by P4 (``best_params.json``)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. File này do P4 tạo bằng scripts/06_tune_xgb.py; "
            "chờ P4 bàn giao trước khi chạy 07."
        )
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)

    missing = {"imbalance", "scale", "params"} - set(result)
    if missing:
        raise ValueError(f"{path} thiếu khóa {sorted(missing)}.")
    if result["imbalance"] not in config.IMBALANCE_CHOICES:
        raise ValueError(
            f"imbalance trong {path} phải thuộc {config.IMBALANCE_CHOICES}, "
            f"nhận được {result['imbalance']!r}."
        )
    if not isinstance(result["scale"], bool):
        raise ValueError(f"scale trong {path} phải là bool, nhận được {result['scale']!r}.")
    if not isinstance(result["params"], dict):
        raise ValueError(
            f"params trong {path} phải là dict, "
            f"nhận được {type(result['params']).__name__}."
        )
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    config.ensure_dirs()

    X_train, X_test, y_train, y_test = load_processed()
    logger.info(
        "Train: %d dòng (%d gian lận); test: %d dòng (%d gian lận)",
        len(y_train),
        int(y_train.sum()),
        len(y_test),
        int(y_test.sum()),
    )

    result = load_params()
    imbalance = result["imbalance"]
    spw = compute_scale_pos_weight(y_train) if imbalance == "weight" else None
    pipe = build_pipeline(
        imbalance, scale=result["scale"], params=result["params"], spw=spw
    )
    logger.info(
        "Cấu hình cuối: imbalance=%s, scale=%s, spw=%s",
        imbalance,
        result["scale"],
        "None" if spw is None else f"{spw:.2f}",
    )

    # Threshold is chosen on out-of-fold train predictions, never on test.
    summary, oof = run_cv(pipe, X_train, y_train)
    threshold = find_threshold(
        y_train, oof, config.THRESHOLD_STRATEGY, config.TARGET_RECALL
    )
    config.THRESHOLD_PATH.write_text(
        json.dumps(
            {"strategy": config.THRESHOLD_STRATEGY, "threshold": threshold},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "CV PR-AUC = %.4f ± %.4f; ngưỡng (%s) = %.4f",
        summary["pr_auc_mean"],
        summary["pr_auc_std"],
        config.THRESHOLD_STRATEGY,
        threshold,
    )

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, config.FINAL_MODEL_PATH)
    logger.info("Đã lưu mô hình cuối vào %s", config.FINAL_MODEL_PATH)

    if args.no_test:
        logger.info("Đã bật --no-test: không đọc tập test, không ghi final_test.csv.")
        print(f"Đã lưu ngưỡng tại {config.THRESHOLD_PATH}")
        print(f"Đã lưu mô hình tại {config.FINAL_MODEL_PATH}")
        return

    proba = pipe.predict_proba(X_test)[:, 1]
    rows = [
        {
            "threshold_type": "default",
            "threshold": config.DEFAULT_THRESHOLD,
            **compute_metrics(y_test, proba, config.DEFAULT_THRESHOLD),
        },
        {
            "threshold_type": f"selected_{config.THRESHOLD_STRATEGY}",
            "threshold": threshold,
            **compute_metrics(y_test, proba, threshold),
        },
    ]
    final_metrics = pd.DataFrame(
        rows,
        columns=[
            "threshold_type",
            "threshold",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )
    metrics_path = save_metrics(final_metrics, "final_test")
    predictions_path = config.PREDICTIONS_DIR / "xgb_final_test.csv"
    pd.DataFrame({"y_true": y_test.values, "y_proba": proba}).to_csv(
        predictions_path, index=False
    )
    curve_path = plot_pr_curve(
        y_test, proba, config.FIGURES_DIR / "07_pr_curve.png", label="XGBoost cuối"
    )
    matrix_path = plot_confusion_matrix(
        y_test, proba, threshold, config.FIGURES_DIR / "07_confusion_matrix.png"
    )

    print("Đánh giá trên tập test (chỉ chạy một lần):")
    print(
        final_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"Đã lưu chỉ số: {metrics_path}")
    print(f"Đã lưu xác suất test: {predictions_path}")
    print(f"Đã lưu biểu đồ: {curve_path}, {matrix_path}")


if __name__ == "__main__":
    main()