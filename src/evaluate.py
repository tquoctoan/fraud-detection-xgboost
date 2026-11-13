"""src/evaluate.py — Đánh giá mô hình, Cross-Validation, tính chỉ số và vẽ biểu đồ."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Hỗ trợ render trên môi trường non-interactive / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src import config

log = logging.getLogger(__name__)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, Any]:
    """Tính toán toàn diện các chỉ số phân loại nhị phân tại ngưỡng threshold xác định.

    Bao gồm các chỉ số theo quy định dự án (pr_auc, precision, recall, f1, tn, fp, fn, tp)
    và các chỉ số mở rộng học thuật (accuracy, macro_f1, weighted_f1, roc_auc).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    # Tính toán an toàn, tránh lỗi chia cho 0
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_proba_arr))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
    except Exception:
        roc_auc = 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_classification_report_str(y_true, y_pred, target_names=("Bình thường (0)", "Gian lận (1)")) -> str:
    """Trả về báo cáo phân loại chi tiết (Classification Report) dưới dạng chuỗi."""
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def run_cv(pipeline, X, y, n_splits: int = config.N_FOLDS) -> tuple[dict[str, float], np.ndarray]:
    """Thực hiện Stratified K-Fold Cross-Validation và trích xuất dự đoán out-of-fold.

    Đảm bảo:
    - Mỗi fold sao chép (clone) một phiên bản pipeline độc lập, không rò rỉ dữ liệu (Data Leakage).
    - Bộ cân bằng dữ liệu (SMOTE/Under) chỉ được fit trên tập train của từng fold.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config.CV_SHUFFLE,
        random_state=config.SEED,
    )
    oof = np.zeros(len(y), dtype=float)

    # Khởi tạo danh sách lưu kết quả các folds cho mọi chỉ số
    tracked_metrics = list(config.CV_METRICS) + ["accuracy", "macro_f1", "weighted_f1", "roc_auc"]
    folds: dict[str, list[float]] = {m: [] for m in tracked_metrics}
    fit_times: list[float] = []

    for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
        model = clone(pipeline)
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va]

        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_times.append(time.time() - t0)

        proba = model.predict_proba(X_va)[:, 1]
        oof[va] = proba

        m = compute_metrics(y_va, proba, config.DEFAULT_THRESHOLD)
        for k in tracked_metrics:
            if k in m:
                folds[k].append(m[k])

        log.info(
            "Fold %d/%d: %d ca gian lận | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Prec=%.4f",
            fold_idx + 1,
            n_splits,
            int(y_va.sum()),
            m["pr_auc"],
            m["f1"],
            m["recall"],
            m["precision"],
        )

    summary: dict[str, float] = {}
    for k, v in folds.items():
        summary[f"{k}_mean"] = float(np.mean(v))
        summary[f"{k}_std"] = float(np.std(v))
    summary["fit_time_mean"] = float(np.mean(fit_times))

    return summary, oof


def find_threshold(
    y_true, y_proba, strategy: str = "f1", target_recall: float = 0.8
) -> float:
    """Tìm ngưỡng phân loại tối ưu dựa trên đường Precision-Recall trên tập out-of-fold."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec = prec[:-1], rec[:-1]

    if strategy == "f1":
        f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
        best_idx = int(np.argmax(f1))
        return float(thr[best_idx])
    if strategy == "recall":
        ok = np.where(rec >= target_recall)[0]
        return float(thr[ok[-1]]) if len(ok) > 0 else float(thr[0])
    raise ValueError(f"Chiến lược tìm ngưỡng không hợp lệ: {strategy}")


def plot_pr_curve(y_true, y_proba, path: Path | str, label: str = "model") -> None:
    """Vẽ đường cong Precision-Recall kèm đường chuẩn cơ sở (baseline ngẫu nhiên)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    baseline_ratio = float(np.mean(y_true))
    ax.plot(rec, prec, lw=2, label=f"{label} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(
        y=baseline_ratio,
        color="crimson",
        linestyle="--",
        label=f"Baseline ngẫu nhiên ({baseline_ratio:.4%})",
    )
    ax.set_xlabel("Recall (Độ nhạy)")
    ax.set_ylabel("Precision (Độ chính xác)")
    ax.set_title(f"Đường cong Precision-Recall - {label}")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(curves: dict[str, tuple[Any, Any]], path: Path | str) -> None:
    """Vẽ so sánh đường cong Precision-Recall của nhiều phương pháp trên cùng một đồ thị."""
    fig, ax = plt.subplots(figsize=(8, 6))
    first_y = None

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    for idx, (name, (y_true, y_proba)) in enumerate(curves.items()):
        if first_y is None:
            first_y = y_true
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        c = colors[idx % len(colors)]
        ax.plot(rec, prec, lw=2, color=c, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    if first_y is not None:
        baseline_ratio = float(np.mean(first_y))
        ax.axhline(
            y=baseline_ratio,
            color="black",
            linestyle=":",
            alpha=0.8,
            label=f"Baseline ngẫu nhiên ({baseline_ratio:.4%})",
        )

    ax.set_xlabel("Recall (Độ nhạy)")
    ax.set_ylabel("Precision (Độ chính xác)")
    ax.set_title("So sánh đường cong Precision-Recall giữa các phương pháp")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true, y_pred, path: Path | str, title: str = "Ma trận nhầm lẫn (Confusion Matrix)"
) -> None:
    """Vẽ ma trận nhầm lẫn (Confusion Matrix) với chú thích số lượng và phần trăm."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Bình thường (0)", "Gian lận (1)"],
        yticklabels=["Bình thường (0)", "Gian lận (1)"],
        ax=ax,
    )
    ax.set_xlabel("Nhãn dự đoán (Predicted Label)")
    ax.set_ylabel("Nhãn thực tế (True Label)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_metrics(obj: dict | pd.DataFrame, name: str) -> Path:
    """Lưu kết quả chỉ số vào thư mục kết quả metrics."""
    config.ensure_dirs()
    target_path = config.METRICS_DIR / f"{name}.csv"
    if isinstance(obj, dict):
        df = pd.DataFrame([obj])
    else:
        df = obj
    df.to_csv(target_path, index=False)
    log.info("Đã lưu chỉ số tại: %s", target_path)
    return target_path
