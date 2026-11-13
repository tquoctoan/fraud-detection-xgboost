"""Build the XGBoost pipelines shared by every experiment (P2).

Order of steps: ``scaler`` (optional), ``sampler`` (optional), ``clf``.
The sampler step only exists for ``smote`` / ``under``; ``weight`` expresses the
imbalance through ``scale_pos_weight`` and ``none`` keeps the data untouched.
"""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src import config


def compute_scale_pos_weight(y) -> float:
    """Return ``n_negative / n_positive`` computed from ``y`` itself."""
    labels = pd.Series(y)
    n_positive = int((labels == 1).sum())
    if n_positive == 0:
        raise ValueError("Cannot compute scale_pos_weight without a positive class.")
    return float((labels == 0).sum() / n_positive)


def build_xgb(params: dict | None = None, scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Create an ``XGBClassifier`` from the fixed + base config and ``params`` overrides."""
    overrides = dict(params or {})
    collisions = {"scale_pos_weight"} & set(overrides)
    if collisions:
        raise ValueError(
            "params must not contain "
            f"{sorted(collisions)}; pass spw/scale_pos_weight separately."
        )
    merged = {**config.XGB_FIXED_PARAMS, **config.XGB_BASE_PARAMS, **overrides}
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **merged)


def build_pipeline(
    imbalance: str = "weight",
    scale: bool = False,
    params: dict | None = None,
    spw: float | None = None,
) -> Pipeline:
    """Assemble scaler (optional) -> sampler (optional) -> XGBoost classifier."""
    if imbalance not in config.IMBALANCE_CHOICES:
        raise ValueError(
            f"imbalance must be one of {config.IMBALANCE_CHOICES}, got {imbalance!r}."
        )
    if imbalance == "weight" and spw is None:
        raise ValueError("imbalance='weight' requires spw=compute_scale_pos_weight(y_train).")

    steps: list[tuple[str, object]] = []
    if scale:
        steps.append(
            (
                "scaler",
                ColumnTransformer(
                    [("scale", RobustScaler(), config.SCALE_COLS)],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            )
        )
    if imbalance == "smote":
        steps.append(
            (
                "sampler",
                SMOTE(
                    sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
                    k_neighbors=config.SMOTE_K_NEIGHBORS,
                    random_state=config.SEED,
                ),
            )
        )
    elif imbalance == "under":
        steps.append(
            (
                "sampler",
                RandomUnderSampler(
                    sampling_strategy=config.UNDER_SAMPLING_STRATEGY,
                    random_state=config.SEED,
                ),
            )
        )

    # Only "weight" uses scale_pos_weight; the other variants must not balance twice.
    spw_used = spw if imbalance == "weight" else 1.0
    steps.append(("clf", build_xgb(params, spw_used)))
    return Pipeline(steps)


def get_clf(pipe: Pipeline) -> XGBClassifier:
    """Return the fitted XGBoost step of ``pipe``."""
    return pipe.named_steps["clf"]


def transform_features(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every preprocessing step of ``pipe`` except the sampler and the model."""
    transformed = X
    for name, step in pipe.steps[:-1]:
        if name == "sampler":
            continue
        transformed = step.transform(transformed)
    return transformed