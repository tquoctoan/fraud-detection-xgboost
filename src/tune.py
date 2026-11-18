"""Optuna/TPE hyperparameter tuning for the project's XGBoost pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src import config
from src.evaluate import run_cv
from src.pipeline import build_pipeline, compute_scale_pos_weight

logger = logging.getLogger(__name__)


def suggest_xgb_params(trial: optuna.Trial) -> dict:
    """Search space for XGBoost; ranges are deliberately bounded."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def load_imbalance_choice() -> str:
    """Read P3's selected imbalance strategy."""
    path = config.CHOSEN_IMBALANCE_PATH

    if not path.exists():
        logger.warning(
            "Không tìm thấy %s; dùng DEFAULT_IMBALANCE=%s.",
            path,
            config.DEFAULT_IMBALANCE,
        )
        return config.DEFAULT_IMBALANCE

    with path.open(encoding="utf-8") as handle:
        choice = json.load(handle)

    imbalance = choice.get("imbalance")

    if imbalance not in config.IMBALANCE_CHOICES:
        raise ValueError(
            f"Giá trị imbalance trong {path} không hợp lệ: {imbalance!r}."
        )

    return imbalance


def objective_factory(X_train, y_train, imbalance: str):
    """Create an Optuna objective using the same CV for every trial."""
    spw = (
        compute_scale_pos_weight(y_train)
        if imbalance == "weight"
        else None
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)

        pipeline = build_pipeline(
            imbalance=imbalance,
            scale=False,
            params=params,
            spw=spw,
        )

        summary, _ = run_cv(pipeline, X_train, y_train)
        score = float(summary["pr_auc_mean"])

        trial.set_user_attr(
            "pr_auc_std", float(summary["pr_auc_std"])
        )
        trial.set_user_attr(
            "fit_time_mean", float(summary["fit_time_mean"])
        )

        logger.info(
            "Trial %d | PR-AUC=%.6f ± %.6f | params=%s",
            trial.number,
            score,
            summary["pr_auc_std"],
            params,
        )

        return score

    return objective


def save_optimization_history(
    study: optuna.Study,
    path: Path | str | None = None,
) -> Path:
    """Save trial PR-AUC and cumulative best PR-AUC as a PNG chart."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]

    if not completed_trials:
        raise RuntimeError("Không có trial hoàn thành để vẽ biểu đồ.")

    trial_numbers = [trial.number + 1 for trial in completed_trials]
    scores = [float(trial.value) for trial in completed_trials]

    best_scores = []
    current_best = float("-inf")

    for score in scores:
        current_best = max(current_best, score)
        best_scores.append(current_best)

    output_path = (
        Path(path)
        if path is not None
        else config.FIGURES_DIR / "06_optimization_history.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        trial_numbers,
        scores,
        alpha=0.65,
        label="PR-AUC từng trial",
    )
    ax.plot(
        trial_numbers,
        best_scores,
        marker="o",
        markersize=3,
        linewidth=2,
        label="PR-AUC tốt nhất tích lũy",
    )

    ax.set_title("Optuna Optimization History - XGBoost")
    ax.set_xlabel("Trial")
    ax.set_ylabel("PR-AUC (5-fold CV)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Đã lưu biểu đồ optimization history: %s", output_path)

    return output_path


# Alias matching QUY_DINH_CHUNG.md specification
plot_optimization_history = save_optimization_history


def save_params(result: dict, path: Path | str = config.BEST_PARAMS_PATH) -> Path:
    """Save best parameters dictionary to JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_params(path: Path | str = config.BEST_PARAMS_PATH) -> dict:
    """Load best parameters from JSON file."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {target}. File này do P4 tạo bằng scripts/06_tune_xgb.py."
        )
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_tuning_comparison(
    X_train,
    y_train,
    imbalance: str,
    best_params: dict,
):
    """Compare baseline and optimized models using the same CV procedure."""
    spw = (
        compute_scale_pos_weight(y_train)
        if imbalance == "weight"
        else None
    )

    models = {
        "Baseline": dict(config.XGB_BASE_PARAMS),
        "Optimized": dict(best_params),
    }

    rows = []

    for model_name, params in models.items():
        logger.info("Đang đánh giá mô hình %s bằng %d-fold CV.",
                    model_name, config.N_FOLDS)

        pipeline = build_pipeline(
            imbalance=imbalance,
            scale=False,
            params=params,
            spw=spw,
        )

        summary, _ = run_cv(pipeline, X_train, y_train)

        row = {
            "model": model_name,
            "imbalance": imbalance,
            "cv_folds": config.N_FOLDS,
        }

        for metric in config.CV_METRICS:
            row[f"{metric}_mean"] = summary[f"{metric}_mean"]
            row[f"{metric}_std"] = summary[f"{metric}_std"]

        row["fit_time_mean_sec"] = summary["fit_time_mean"]
        rows.append(row)

    if len(rows) >= 2:
        base_score = rows[0]["pr_auc_mean"]
        opt_score = rows[1]["pr_auc_mean"]
        imp = ((opt_score - base_score) / base_score) * 100 if base_score != 0 else 0.0
        rows[0]["improvement_pct"] = 0.0
        rows[1]["improvement_pct"] = imp

    comparison_df = pd.DataFrame(rows)

    output_path = config.METRICS_DIR / "tuning_compare.csv"
    comparison_df.to_csv(output_path, index=False)

    logger.info("Đã lưu bảng so sánh Baseline/Optimized: %s", output_path)

    return output_path, comparison_df


def run_tuning(X_train, y_train, n_trials: int | None = None) -> dict:
    """Run TPE search and save all P4 outputs."""
    config.ensure_dirs()

    n_trials = config.N_TRIALS if n_trials is None else int(n_trials)

    if n_trials < 1:
        raise ValueError("n_trials phải lớn hơn hoặc bằng 1.")

    imbalance = load_imbalance_choice()

    logger.info(
        "Bắt đầu tuning: %d trials, %d-fold CV, imbalance=%s, metric=PR-AUC.",
        n_trials,
        config.N_FOLDS,
        imbalance,
    )

    sampler = TPESampler(seed=config.SEED)

    study = optuna.create_study(
        direction=config.OPTUNA_DIRECTION,
        sampler=sampler,
        study_name="xgboost_fraud_detection",
    )

    study.optimize(
        objective_factory(X_train, y_train, imbalance),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    best = {
        "imbalance": imbalance,
        "scale": False,
        "params": study.best_params,
        "cv_pr_auc_mean": float(study.best_value),
        "cv_pr_auc_std": float(study.best_trial.user_attrs.get("pr_auc_std", 0.0)),
        "seed": config.SEED,
        "best_pr_auc_cv": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "n_trials": len(study.trials),
        "sampler": "TPESampler",
        "cv_folds": config.N_FOLDS,
        "metric": "pr_auc_mean",
    }

    # 1. Lưu siêu tham số tốt nhất.
    save_params(best, config.BEST_PARAMS_PATH)

    # 2. Lưu lịch sử các trial.
    trials_path = config.METRICS_DIR / "tuning_trials.csv"

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state", "user_attrs")
    )

    trials_df.to_csv(trials_path, index=False)

    # 3. Lưu biểu đồ lịch sử tối ưu.
    history_path = save_optimization_history(study)

    # 4. So sánh Baseline và Optimized trên cùng quy trình CV.
    comparison_path, comparison_df = save_tuning_comparison(
        X_train=X_train,
        y_train=y_train,
        imbalance=imbalance,
        best_params=study.best_params,
    )

    logger.info(
        "Hoàn tất tuning: best PR-AUC=%.6f ở trial %d.",
        study.best_value,
        study.best_trial.number,
    )
    logger.info("Best params: %s", config.BEST_PARAMS_PATH)
    logger.info("Trial history: %s", trials_path)
    logger.info("Optimization history: %s", history_path)
    logger.info("Tuning comparison: %s", comparison_path)

    return best
