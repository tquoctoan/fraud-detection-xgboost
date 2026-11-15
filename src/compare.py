from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evaluate import COLORS, run_cv
from src.pipeline import build_pipeline, compute_scale_pos_weight

log = logging.getLogger(__name__)

# Thứ tự ưu tiên khi các kỹ thuật ngang nhau (sai khác < 1 độ lệch chuẩn):
# ưu tiên phương pháp nhẹ và ít làm biến dạng phân phối gốc.
PRIORITY = ["weight", "none", "under", "smote"]


def compare_variants(
    X: pd.DataFrame,
    y: pd.Series,
    variants: tuple[str, ...] | list[str] = config.IMBALANCE_CHOICES,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Chạy 5-Fold Stratified CV cho từng biến thể cân bằng dữ liệu."""
    spw = compute_scale_pos_weight(y)
    rows: list[dict[str, float | str]] = []
    oofs: dict[str, np.ndarray] = {}

    for variant in variants:
        pipe = build_pipeline(imbalance=variant, spw=spw if variant == "weight" else None)
        summary, oof = run_cv(pipe, X, y)
        rows.append({"imbalance": variant, **summary})
        oofs[variant] = oof
        log.info(
            "%s: PR-AUC = %.4f ± %.4f | fit_time = %.2fs/fold",
            variant,
            summary["pr_auc_mean"],
            summary["pr_auc_std"],
            summary["fit_time_mean"],
        )

    return pd.DataFrame(rows), oofs


def choose_imbalance(df: pd.DataFrame) -> dict[str, str]:
    """Chọn kỹ thuật tối ưu theo PR-AUC trung bình và quy tắc 1 độ lệch chuẩn."""
    if df.empty:
        raise ValueError("df is empty; cannot choose imbalance method.")

    best = df.loc[df["pr_auc_mean"].idxmax()]
    threshold = best["pr_auc_mean"] - best["pr_auc_std"]
    close = df[df["pr_auc_mean"] >= threshold]
    chosen = min(close["imbalance"], key=PRIORITY.index)

    reason = (
        f"PR-AUC cao nhất: {best['imbalance']} ({best['pr_auc_mean']:.4f}). "
        f"Các cách chênh nhỏ hơn 1 độ lệch chuẩn ({best['pr_auc_std']:.4f}) coi là "
        f"tương đương: {list(close['imbalance'])}. Chọn {chosen} theo thứ tự ưu tiên "
        f"{PRIORITY} (đơn giản hơn khi ngang nhau)."
    )
    return {"imbalance": str(chosen), "reason": reason}


def plot_variant_comparison(df: pd.DataFrame, path: Path | str) -> Path:
    """Vẽ biểu đồ cột PR-AUC trung bình kèm thanh sai số độ lệch chuẩn."""
    if df.empty:
        raise ValueError("df is empty; cannot plot comparison.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    colors = [COLORS[i % len(COLORS)] for i in range(len(df))]
    bars = ax.bar(
        x_pos,
        df["pr_auc_mean"],
        yerr=df["pr_auc_std"],
        capsize=6,
        color=colors,
        edgecolor="black",
        alpha=0.88,
    )

    ax.set_xticks(x_pos, labels=df["imbalance"], fontsize=11)
    ax.set_xlabel("Kỹ thuật xử lý mất cân bằng", fontsize=11)
    ax.set_ylabel("PR-AUC trung bình (5-Fold CV)", fontsize=11)
    ax.set_title("So sánh PR-AUC giữa các kỹ thuật xử lý mất cân bằng", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, mean_val, std_val in zip(bars, df["pr_auc_mean"], df["pr_auc_std"]):
        ax.annotate(
            f"{mean_val:.4f}\n±{std_val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean_val + std_val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
