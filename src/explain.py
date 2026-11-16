"""src/explain.py — SHAP và phân tích ca đoán sai (P5).

Chỉ GIẢI THÍCH mô hình đã huấn luyện: không fit lại, không tune, không chọn ngưỡng.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.pipeline import get_clf, transform_features

logger = logging.getLogger(__name__)

ERROR_TYPES = ["TP", "TN", "FP", "FN"]


def compute_shap(
    pipe,
    X: pd.DataFrame,
    y,
    sample_size: int = config.SHAP_SAMPLE_SIZE,
    seed: int = config.SEED,
):
    """Tính SHAP trên mẫu con: giữ TOÀN BỘ ca gian lận, bốc ngẫu nhiên ca bình thường.

    Trả về (values, X_t): values có cùng shape với X_t (đặc trưng sau tiền xử lý).
    """
    labels = np.asarray(y).ravel()
    if len(labels) != len(X):
        raise ValueError(f"X có {len(X)} dòng nhưng y có {len(labels)} phần tử.")

    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_neg = min(max(sample_size - len(pos), 0), len(neg))
    idx = np.sort(np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)]))

    # Đi qua scaler (nếu có) của pipeline, bỏ qua bước sampler
    X_t = transform_features(pipe, X.iloc[idx])
    values = np.asarray(shap.TreeExplainer(get_clf(pipe)).shap_values(X_t))
    if values.shape != X_t.shape:
        raise ValueError(
            f"SHAP trả về shape {values.shape}, khác shape đặc trưng {X_t.shape}."
        )
    logger.info(
        "SHAP trên %d dòng (%d gian lận, %d bình thường)", len(idx), len(pos), n_neg
    )
    return values, X_t


def _finish_plot(path, title: str) -> Path:
    fig = plt.gcf()
    plt.title(title, fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_summary(values, X_t: pd.DataFrame, path, title: str | None = None) -> Path:
    """Beeswarm plot: mỗi chấm là một giao dịch."""
    shap.summary_plot(values, X_t, show=False, max_display=config.SHAP_TOP_N)
    default = f"SHAP beeswarm (mẫu {len(X_t):,} dòng, giữ toàn bộ ca gian lận)"
    return _finish_plot(path, title or default)


def plot_shap_bar(values, X_t: pd.DataFrame, path, title: str | None = None) -> Path:
    """Bar plot: trung bình |SHAP| của từng đặc trưng."""
    shap.summary_plot(
        values, X_t, plot_type="bar", show=False, max_display=config.SHAP_TOP_N
    )
    default = f"Mức quan trọng trung bình |SHAP| (mẫu {len(X_t):,} dòng)"
    return _finish_plot(path, title or default)


def get_errors(X: pd.DataFrame, y_true, y_proba, threshold: float) -> pd.DataFrame:
    """Gán TP/TN/FP/FN cho từng dòng. Dùng '>=' giống evaluate.compute_metrics."""
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba, dtype="float64").ravel()
    if not (len(X) == len(y_true) == len(y_proba)):
        raise ValueError(
            f"Độ dài không khớp: X={len(X)}, y_true={len(y_true)}, y_proba={len(y_proba)}."
        )

    df = X.copy().reset_index(drop=True)
    df.insert(0, "y_true", y_true.astype(int))
    df.insert(1, "y_proba", y_proba)
    df.insert(2, "y_pred", (y_proba >= threshold).astype(int))
    df.insert(
        3,
        "error_type",
        np.select(
            [
                (df["y_true"] == 1) & (df["y_pred"] == 1),
                (df["y_true"] == 0) & (df["y_pred"] == 0),
                (df["y_true"] == 0) & (df["y_pred"] == 1),
            ],
            ["TP", "TN", "FP"],
            default="FN",
        ),
    )
    return df


def summarize_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê theo loại: số ca, xác suất trung bình, Amount trung vị."""
    summary = (
        df.groupby("error_type")
        .agg(
            n=("y_true", "size"),
            proba_mean=("y_proba", "mean"),
            amount_median=(config.AMOUNT_COL, "median"),
        )
        .reindex(ERROR_TYPES)
        .reset_index()
    )
    summary["n"] = summary["n"].fillna(0).astype(int)
    return summary