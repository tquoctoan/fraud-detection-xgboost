"""Cross-validation, metrics, threshold selection, and evaluation plots (P2).

Used by P3 (``05``), P4 (``06``), P2 (``04`` / ``07``) and P5 (``08``), so the
functions keep the signatures documented in ``QUY_DINH_CHUNG.md``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must run on a headless machine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src import config

logger = logging.getLogger(__name__)

COLORS = ["#4C78A8", "#E45756", "#72B7B2", "#F58518", "#54A24B", "#B279A2"]


def _as_1d(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype="float64").ravel()
    if array.size == 0:
        raise ValueError(f"{name} is empty.")
    return array


def compute_metrics(y_true, y_proba, threshold: float = config.DEFAULT_THRESHOLD) -> dict:
    """Return PR-AUC plus precision/recall/F1 and the confusion counts at ``threshold``."""
    y_true = _as_1d(y_true, "y_true")
    y_proba = _as_1d(y_proba, "y_proba")
    if y_true.shape != y_proba.shape:
        raise ValueError(
            f"y_true and y_proba must have the same length, got {y_true.size} and {y_proba.size}."
        )
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _take(data, indices):
    """Positional indexing for both pandas objects and numpy arrays."""
    return data.iloc[indices] if hasattr(data, "iloc") else data[indices]


def run_cv(
    pipeline,
    X,
    y,
    n_splits: int = config.N_FOLDS,
) -> tuple[dict, np.ndarray]:
    """Fit ``pipeline`` with StratifiedKFold and return (summary, out-of-fold proba).

    Each fold gets its own clone; ``oof_proba`` is filled exactly once per row so
    callers can select a threshold without ever touching the test split.
    """
    labels = pd.Series(np.asarray(y).ravel())
    n_positive = int((labels == 1).sum())
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
    if n_positive < n_splits:
        raise ValueError(
            f"StratifiedKFold(n_splits={n_splits}) needs at least {n_splits} "
            f"positive cases, got {n_positive}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config.CV_SHUFFLE,
        random_state=config.SEED,
    )
    oof = np.zeros(len(labels), dtype="float64")
    fold_scores: dict[str, list[float]] = {metric: [] for metric in config.CV_METRICS}
    fit_times: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, labels), start=1):
        model = clone(pipeline)  # a fresh unfitted pipeline per fold
        started = time.perf_counter()
        model.fit(_take(X, train_idx), labels.iloc[train_idx])
        fit_times.append(time.perf_counter() - started)

        proba = model.predict_proba(_take(X, valid_idx))[:, 1]
        oof[valid_idx] = proba

        fold_metrics = compute_metrics(
            labels.iloc[valid_idx], proba, config.DEFAULT_THRESHOLD
        )
        for metric in config.CV_METRICS:
            fold_scores[metric].append(fold_metrics[metric])
        logger.info(
            "Fold %d/%d: %d ca gian lận, PR-AUC=%.4f, F1=%.4f (%.1fs)",
            fold,
            n_splits,
            int(labels.iloc[valid_idx].sum()),
            fold_metrics["pr_auc"],
            fold_metrics["f1"],
            fit_times[-1],
        )

    summary: dict[str, float] = {}
    for metric, values in fold_scores.items():
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    summary["fit_time_mean"] = float(np.mean(fit_times))
    return summary, oof


def find_threshold(
    y_true,
    y_proba,
    strategy: str = "f1",
    target_recall: float = config.TARGET_RECALL,
) -> float:
    """Pick a decision threshold from probabilities (out-of-fold only, never test)."""
    y_true = _as_1d(y_true, "y_true")
    y_proba = _as_1d(y_proba, "y_proba")
    if y_true.shape != y_proba.shape:
        raise ValueError(
            f"y_true and y_proba must have the same length, got {y_true.size} and {y_proba.size}."
        )

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one extra element compared with the threshold grid
    precision, recall = precision[:-1], recall[:-1]

    if strategy == "f1":
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return float(thresholds[int(np.argmax(f1))])
    if strategy == "recall":
        if not 0.0 < target_recall <= 1.0:
            raise ValueError(f"target_recall must be in (0, 1], got {target_recall}.")
        eligible = np.where(recall >= target_recall)[0]
        if eligible.size == 0:
            raise ValueError(
                f"No threshold reaches recall >= {target_recall}; "
                f"max recall is {recall.max():.4f}."
            )
        return float(thresholds[eligible[-1]])
    raise ValueError(
        f"Unknown threshold strategy {strategy!r}; expected 'f1' or 'recall'."
    )


def _save_figure(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
    return path


def plot_pr_curve(y_true, y_proba, path, label: str = "model") -> Path:
    """Draw one precision-recall curve with a random-classifier baseline."""
    y_true = _as_1d(y_true, "y_true")
    y_proba = _as_1d(y_proba, "y_proba")
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    prevalence = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=COLORS[0], label=f"{label} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"Ngẫu nhiên (tỷ lệ gian lận = {prevalence:.3%})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Đường Precision–Recall: {label}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    return _save_figure(fig, path)


def plot_pr_curves(curves: dict, path) -> Path:
    """Draw several PR curves at once; ``curves`` maps a name to ``(y_true, y_proba)``."""
    if not curves:
        raise ValueError("curves is empty; expected {name: (y_true, y_proba)}.")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    prevalences = []
    for index, (name, (y_true, y_proba)) in enumerate(curves.items()):
        y_true = _as_1d(y_true, "y_true")
        y_proba = _as_1d(y_proba, "y_proba")
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        prevalences.append(float(y_true.mean()))
        ax.plot(
            recall,
            precision,
            color=COLORS[index % len(COLORS)],
            label=f"{name} (PR-AUC = {pr_auc:.4f})",
        )

    if np.allclose(prevalences, prevalences[0]):
        ax.axhline(
            prevalences[0],
            linestyle="--",
            color="gray",
            label=f"Ngẫu nhiên (tỷ lệ gian lận = {prevalences[0]:.3%})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("So sánh đường Precision–Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    return _save_figure(fig, path)


def plot_confusion_matrix(y_true, y_proba, threshold, path) -> Path:
    """Draw the 2x2 confusion matrix at ``threshold``."""
    metrics = compute_metrics(y_true, y_proba, threshold)
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    class_labels = ["Bình thường", "Gian lận"]
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(
        f"Ma trận nhầm lẫn tại ngưỡng {threshold:.4f}\n"
        f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}"
    )
    limit = matrix.max() if matrix.max() > 0 else 1
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > limit / 2 else "black",
                fontsize=11,
            )
    return _save_figure(fig, path)


def save_metrics(obj, name: str) -> Path:
    """Write a metric dict or DataFrame to ``results/metrics/<name>.csv``."""
    if isinstance(obj, dict):
        frame = pd.DataFrame([obj])
    elif isinstance(obj, pd.DataFrame):
        frame = obj
    else:
        raise TypeError(f"save_metrics expects dict or DataFrame, got {type(obj).__name__}.")

    path = config.METRICS_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Đã lưu chỉ số vào %s", path)
    return path