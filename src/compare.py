"""src/compare.py — So sánh các kỹ thuật xử lý dữ liệu mất cân bằng (P3).

Triển khai 4 phương pháp:
1. Baseline (Không xử lý mất cân bằng - imbalance='none')
2. Class Weight / scale_pos_weight (imbalance='weight')
3. SMOTE (Synthetic Minority Over-sampling Technique - imbalance='smote')
4. Random Undersampling (imbalance='under')

Tuân thủ:
- Không rò rỉ dữ liệu (Data Leakage): Sampler chỉ nằm trong pipeline huấn luyện.
- Cố định seed cho khả năng tái lập kết quả.
- Đánh giá toàn diện trên cả Cross-Validation và Test Set.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.evaluate import compute_metrics, run_cv
from src.pipeline import build_pipeline, compute_scale_pos_weight

log = logging.getLogger(__name__)

# Thứ tự ưu tiên khi các phương pháp có kết quả tương đương (chênh lệch trong khoảng 1 độ lệch chuẩn)
# Ưu tiên phương pháp tính toán nhẹ và ít can thiệp vào cấu trúc phân phối dữ liệu
PRIORITY = ["weight", "none", "under", "smote"]

METHOD_NAMES = {
    "none": "Baseline (Gốc)",
    "weight": "Class Weight",
    "smote": "SMOTE",
    "under": "Random Undersampling",
}


def compare_variants(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    variants: tuple[str, ...] = config.IMBALANCE_CHOICES,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """So sánh 4 kỹ thuật xử lý mất cân bằng thông qua Stratified 5-Fold Cross-Validation.

    Parameters
    ----------
    X_train : pd.DataFrame
        Đặc trưng tập huấn luyện.
    y_train : pd.Series
        Nhãn tập huấn luyện (0 hoặc 1).
    variants : tuple[str, ...]
        Danh sách các biến thể cần so sánh ('none', 'weight', 'smote', 'under').

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        - DataFrame tổng hợp các chỉ số trung bình và độ lệch chuẩn của từng biến thể.
        - Dictionary chứa mảng xác suất dự đoán out-of-fold (OOF) của từng biến thể.
    """
    spw = compute_scale_pos_weight(y_train)
    rows: list[dict[str, Any]] = []
    oofs: dict[str, np.ndarray] = {}

    log.info("Bắt đầu so sánh các phương pháp xử lý mất cân bằng trên Cross-Validation...")
    for v in variants:
        log.info(">>> Đang huấn luyện và đánh giá kỹ thuật: %s (%s)", v, METHOD_NAMES.get(v, v))
        pipe = build_pipeline(imbalance=v, spw=spw if v == "weight" else None)
        summary, oof = run_cv(pipe, X_train, y_train)

        row = {
            "imbalance": v,
            "method_name": METHOD_NAMES.get(v, v),
            **summary,
        }
        rows.append(row)
        oofs[v] = oof

        log.info(
            "%s: PR-AUC=%.4f ± %.4f | F1=%.4f ± %.4f | Recall=%.4f ± %.4f | Prec=%.4f ± %.4f | FitTime=%.2fs",
            v,
            summary["pr_auc_mean"],
            summary["pr_auc_std"],
            summary["f1_mean"],
            summary["f1_std"],
            summary["recall_mean"],
            summary["recall_std"],
            summary["precision_mean"],
            summary["precision_std"],
            summary["fit_time_mean"],
        )

    df_results = pd.DataFrame(rows)
    return df_results, oofs


def evaluate_variants_on_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    variants: tuple[str, ...] = config.IMBALANCE_CHOICES,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Huấn luyện từng phương pháp trên toàn bộ X_train và đánh giá trên cùng tập X_test.

    Đảm bảo:
    - Cùng tập Train và cùng tập Test gốc.
    - Cùng thuật toán XGBoost và cùng tham số.
    - Không làm thay đổi phân phối nhãn tập Test.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]
        - DataFrame bảng so sánh kết quả trên Test.
        - Dictionary xác suất dự đoán trên Test của từng phương pháp.
        - Dictionary các mô hình đã fit.
    """
    spw = compute_scale_pos_weight(y_train)
    rows: list[dict[str, Any]] = []
    test_probas: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    log.info("Bắt đầu đánh giá 4 phương pháp trên tập Test độc lập...")
    for v in variants:
        pipe = build_pipeline(imbalance=v, spw=spw if v == "weight" else None)
        pipe.fit(X_train, y_train)
        fitted_models[v] = pipe

        proba = pipe.predict_proba(X_test)[:, 1]
        test_probas[v] = proba

        m = compute_metrics(y_test, proba, threshold=threshold)
        rows.append({
            "Phương pháp": METHOD_NAMES.get(v, v),
            "imbalance_key": v,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-score": m["f1"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "PR-AUC": m["pr_auc"],
            "ROC-AUC": m["roc_auc"],
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
            "TN": m["tn"],
        })

    df_test_comparison = pd.DataFrame(rows)
    return df_test_comparison, test_probas, fitted_models


def choose_imbalance(df: pd.DataFrame) -> dict[str, str]:
    """Lựa chọn phương pháp xử lý mất cân bằng tối ưu dựa trên quy tắc thống kê học máy:

    1. Lấy phương pháp có PR-AUC trung bình cao nhất.
    2. Các phương pháp chênh lệch nhỏ hơn 1 độ lệch chuẩn (std) được coi là tương đương nhau.
    3. Chọn phương pháp đơn giản nhất theo thứ tự ưu tiên: weight -> none -> under -> smote.
    """
    best_idx = df["pr_auc_mean"].idxmax()
    best_row = df.loc[best_idx]
    threshold_value = best_row["pr_auc_mean"] - best_row["pr_auc_std"]

    close_candidates = df[df["pr_auc_mean"] >= threshold_value]
    chosen_method = min(close_candidates["imbalance"], key=PRIORITY.index)

    reason = (
        f"PR-AUC trung bình cao nhất thuộc về '{best_row['imbalance']}' ({best_row['pr_auc_mean']:.4f} ± {best_row['pr_auc_std']:.4f}). "
        f"Các kỹ thuật nằm trong khoảng 1 độ lệch chuẩn gồm: {list(close_candidates['imbalance'])}. "
        f"Theo quy tắc Parsimony và độ ổn định tính toán (thứ tự ưu tiên {PRIORITY}), "
        f"kỹ thuật '{chosen_method}' được lựa chọn vì mang lại sự cân bằng tối ưu giữa hiệu năng và độ phức tạp tính toán."
    )
    return {"imbalance": chosen_method, "reason": reason}


def plot_variant_comparison(df: pd.DataFrame, path: Path | str) -> None:
    """Vẽ biểu đồ thanh so sánh PR-AUC trung bình kèm sai số độ lệch chuẩn giữa các kỹ thuật."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    bars = ax.bar(
        x_pos,
        df["pr_auc_mean"],
        yerr=df["pr_auc_std"],
        capsize=6,
        color=["#4C78A8", "#54A24B", "#F58518", "#E45756"],
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_xticks(x_pos)
    labels = [METHOD_NAMES.get(name, name) for name in df["imbalance"]]
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("PR-AUC trung bình (5-Fold CV)", fontsize=11)
    ax.set_title("So sánh hiệu năng PR-AUC giữa các kỹ thuật xử lý mất cân bằng", fontsize=12, fontweight="bold")
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Ghi số điểm trên cột
    for bar, mean_val, std_val in zip(bars, df["pr_auc_mean"], df["pr_auc_std"]):
        ax.annotate(
            f"{mean_val:.4f}\n(±{std_val:.3f})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_f1_precision_recall_comparison(df: pd.DataFrame, path: Path | str) -> None:
    """Vẽ biểu đồ nhóm so sánh F1-score, Precision và Recall giữa 4 phương pháp."""
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = [METHOD_NAMES.get(m, m) for m in df["imbalance_key"]] if "imbalance_key" in df.columns else df["imbalance"]
    x = np.arange(len(methods))
    width = 0.25

    prec_vals = df["Precision"] if "Precision" in df.columns else df["precision_mean"]
    rec_vals = df["Recall"] if "Recall" in df.columns else df["recall_mean"]
    f1_vals = df["F1-score"] if "F1-score" in df.columns else df["f1_mean"]

    rects1 = ax.bar(x - width, prec_vals, width, label="Precision", color="#4C78A8", edgecolor="black")
    rects2 = ax.bar(x, rec_vals, width, label="Recall", color="#F58518", edgecolor="black")
    rects3 = ax.bar(x + width, f1_vals, width, label="F1-score", color="#54A24B", edgecolor="black")

    ax.set_ylabel("Điểm số (0 - 1)", fontsize=11)
    ax.set_title("So sánh Precision, Recall và F1-score giữa các phương pháp", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def _autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _autolabel(rects1)
    _autolabel(rects2)
    _autolabel(rects3)

    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_all_confusion_matrices(
    y_test: pd.Series,
    test_probas: dict[str, np.ndarray],
    threshold: float,
    path: Path | str,
) -> None:
    """Vẽ 4 ma trận nhầm lẫn (Confusion Matrix) trên cùng một figure 2x2 để trực quan hóa sự đánh đổi."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, (variant_key, proba) in enumerate(test_probas.items()):
        ax = axes[idx]
        y_pred = (proba >= threshold).astype(int)
        cm = pd.crosstab(
            pd.Series(y_test, name="Thực tế"),
            pd.Series(y_pred, name="Dự đoán"),
            rownames=["Thực tế"],
            colnames=["Dự đoán"],
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax,
        )
        tn = cm.iloc[0, 0] if 0 in cm.index and 0 in cm.columns else 0
        fp = cm.iloc[0, 1] if 0 in cm.index and 1 in cm.columns else 0
        fn = cm.iloc[1, 0] if 1 in cm.index and 0 in cm.columns else 0
        tp = cm.iloc[1, 1] if 1 in cm.index and 1 in cm.columns else 0

        title_str = (
            f"{METHOD_NAMES.get(variant_key, variant_key)}\n"
            f"TP={tp}, FN={fn} (Bỏ sót) | FP={fp} (Báo động giả)"
        )
        ax.set_title(title_str, fontsize=11, fontweight="bold")

    fig.suptitle(f"Ma trận nhầm lẫn của 4 phương pháp (Ngưỡng = {threshold})", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_resampling_distributions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    path: Path | str,
) -> None:
    """Vẽ biểu đồ phân phối số lượng mẫu của các lớp trước và sau khi áp dụng SMOTE và Undersampling."""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    # 1. Tập gốc
    counts_original = y_train.value_counts().to_dict()

    # 2. Sau SMOTE
    smote = SMOTE(
        sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
        random_state=config.SEED,
    )
    _, y_smote = smote.fit_resample(X_train, y_train)
    counts_smote = pd.Series(y_smote).value_counts().to_dict()

    # 3. Sau Undersampling
    under = RandomUnderSampler(
        sampling_strategy=config.UNDER_SAMPLING_STRATEGY,
        random_state=config.SEED,
    )
    _, y_under = under.fit_resample(X_train, y_train)
    counts_under = pd.Series(y_under).value_counts().to_dict()

    # Dữ liệu vẽ
    techs = ["Gốc / Class Weight", "Sau SMOTE", "Sau Undersampling"]
    class0_counts = [counts_original.get(0, 0), counts_smote.get(0, 0), counts_under.get(0, 0)]
    class1_counts = [counts_original.get(1, 0), counts_smote.get(1, 0), counts_under.get(1, 0)]

    x = np.arange(len(techs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars0 = ax.bar(x - width / 2, class0_counts, width, label="Lớp 0: Bình thường", color="#4C78A8", edgecolor="black")
    bars1 = ax.bar(x + width / 2, class1_counts, width, label="Lớp 1: Gian lận", color="#E45756", edgecolor="black")

    ax.set_yscale("log")
    ax.set_ylabel("Số lượng mẫu (Thang log)", fontsize=11)
    ax.set_title("Phân phối dữ liệu trước và sau khi áp dụng các kỹ thuật cân bằng", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(techs, fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Chú thích số lượng mẫu
    for bar in bars0:
        h = bar.get_height()
        ax.annotate(f"{int(h):,}", (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{int(h):,}", (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#B22222", fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
