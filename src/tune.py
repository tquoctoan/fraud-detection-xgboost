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
    """Search space for XGBoost; ranges are deliberately bounded for this dataset."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def load_imbalance_choice() -> str:
    """Read P3's selected imbalance strategy; use config default only if absent."""
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
    """Create an Optuna objective that evaluates each trial using the same CV."""
    spw = compute_scale_pos_weight(y_train) if imbalance == "weight" else None

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
        trial.set_user_attr("pr_auc_std", float(summary["pr_auc_std"]))
        trial.set_user_attr("fit_time_mean", float(summary["fit_time_mean"]))
        logger.info(
            "Trial %d | PR-AUC=%.6f ± %.6f | params=%s",
            trial.number,
            score,
            summary["pr_auc_std"],
            params,
        )
        return score

    return objective


def run_tuning(X_train, y_train, n_trials: int | None = None) -> dict:
    """Run TPE search and persist best parameters plus the trial history."""
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
        "best_pr_auc_cv": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "n_trials": len(study.trials),
        "sampler": "TPESampler",
        "cv_folds": config.N_FOLDS,
        "metric": "pr_auc_mean",
    }
    config.BEST_PARAMS_PATH.write_text(
        json.dumps(best, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    trials_path = config.METRICS_DIR / "tuning_trials.csv"
    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state", "user_attrs")
    )
    trials_df.to_csv(trials_path, index=False)

    logger.info(
        "Hoàn tất tuning: best PR-AUC=%.6f ở trial %d.",
        study.best_value,
        study.best_trial.number,
    )
    logger.info("Đã lưu best params: %s", config.BEST_PARAMS_PATH)
    logger.info("Đã lưu lịch sử trials: %s", trials_path)
    return best
