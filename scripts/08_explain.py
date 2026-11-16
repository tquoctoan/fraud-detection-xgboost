"""Giải thích mô hình cuối: SHAP và phân tích ca đoán sai trên tập test.

Không huấn luyện lại, không tune, không đổi ngưỡng (ngưỡng do 07 quyết định).
Cờ --baseline: chạy thử trên mô hình baseline với ngưỡng 0,5 và ghi file có
hậu tố _baseline, để không lẫn với kết quả thật.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_processed
from src.explain import (
    compute_shap,
    get_errors,
    plot_shap_bar,
    plot_shap_summary,
    summarize_errors,
)

logger = logging.getLogger(__name__)

SLIM_COLS = ["y_true", "y_proba", "y_pred", "error_type", "Time", "Amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Chạy thử trên mô hình baseline, ngưỡng 0,5, ghi file hậu tố _baseline.",
    )
    return parser.parse_args()


def read_threshold() -> float:
    if not config.THRESHOLD_PATH.exists():
        print(
            f"CẢNH BÁO: chưa có {config.THRESHOLD_PATH}; "
            f"dùng ngưỡng mặc định {config.DEFAULT_THRESHOLD}."
        )
        return config.DEFAULT_THRESHOLD
    with config.THRESHOLD_PATH.open(encoding="utf-8") as handle:
        return float(json.load(handle)["threshold"])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    config.ensure_dirs()

    _, X_test, _, y_test = load_processed()

    if args.baseline:
        model_path = config.BASELINE_MODEL_PATH
        threshold = config.DEFAULT_THRESHOLD
        suffix = "_baseline"
        print(f"CHẾ ĐỘ THỬ: mô hình baseline, ngưỡng {threshold}, file có hậu tố {suffix}.")
    else:
        model_path = config.FINAL_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {model_path}. File này do P2 tạo bằng "
                "scripts/07_evaluate.py; chờ P2 bàn giao trước khi chạy 08."
            )
        threshold = read_threshold()
        suffix = ""

    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(X_test)[:, 1]

    errors = get_errors(X_test, y_test, proba, threshold)
    if len(errors) != len(X_test):
        raise ValueError("Số dòng errors khác số dòng test.")
    errors_path = config.METRICS_DIR / f"errors{suffix}.csv"
    errors[SLIM_COLS].to_csv(errors_path, index=False)

    summary = summarize_errors(errors)
    summary_path = config.METRICS_DIR / f"errors_summary{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    values, X_t = compute_shap(pipe, X_test, y_test)
    summary_fig = plot_shap_summary(
        values, X_t, config.FIGURES_DIR / f"08_shap_summary{suffix}.png"
    )
    bar_fig = plot_shap_bar(
        values, X_t, config.FIGURES_DIR / f"08_shap_bar{suffix}.png"
    )

    print(f"Ngưỡng dùng: {threshold:.4f}")
    print("Phân tích lỗi trên tập test:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n_fn = int(summary.loc[summary["error_type"] == "FN", "n"].iloc[0])
    n_fp = int(summary.loc[summary["error_type"] == "FP", "n"].iloc[0])
    print(f"Số ca lọt (FN): {n_fn}; số ca báo nhầm (FP): {n_fp}")
    print(f"Đã lưu: {errors_path}, {summary_path}")
    print(f"Đã lưu biểu đồ: {summary_fig}, {bar_fig}")


if __name__ == "__main__":
    main()