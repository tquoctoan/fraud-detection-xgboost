"""src/pipeline.py — Dựng pipeline XGBoost, scaler và bộ cân bằng dữ liệu."""
from __future__ import annotations

import logging
from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src import config

log = logging.getLogger(__name__)


def compute_scale_pos_weight(y) -> float:
    """Tính tỷ lệ giữa số lượng mẫu lớp 0 và lớp 1: (y == 0).sum() / (y == 1).sum()."""
    n_neg = float((y == 0).sum())
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        raise ValueError("Lớp dương không có mẫu nào để tính scale_pos_weight.")
    return n_neg / n_pos


def build_xgb(params: dict[str, Any] | None = None, scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Khởi tạo mô hình XGBClassifier kết hợp các tham số cố định và tham số cơ sở."""
    p = {**config.XGB_FIXED_PARAMS, **config.XGB_BASE_PARAMS, **(params or {})}
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **p)


def build_pipeline(
    imbalance: str = "weight",
    scale: bool = False,
    params: dict[str, Any] | None = None,
    spw: float | None = None,
) -> Pipeline:
    """Dựng imblearn Pipeline gồm tiền xử lý (tùy chọn), bộ cân bằng (tùy chọn) và XGBoost.

    Parameters
    ----------
    imbalance : str
        Một trong 4 phương pháp: 'none', 'weight', 'smote', 'under'.
    scale : bool
        Nếu True, áp dụng RobustScaler lên các cột Time và Amount.
    params : dict, optional
        Tham số bổ sung ghi đè cho XGBoost.
    spw : float, optional
        Tỷ lệ scale_pos_weight, bắt buộc phải truyền nếu imbalance='weight'.
    """
    if imbalance not in config.IMBALANCE_CHOICES:
        raise ValueError(
            f"imbalance='{imbalance}' không hợp lệ. Phải thuộc {config.IMBALANCE_CHOICES}"
        )
    if imbalance == "weight" and spw is None:
        raise ValueError(
            "Khi imbalance='weight', bắt buộc phải truyền spw=compute_scale_pos_weight(y_train)"
        )

    steps: list[tuple[str, Any]] = []

    if scale:
        steps.append(
            (
                "scaler",
                ColumnTransformer(
                    [("s", RobustScaler(), config.SCALE_COLS)],
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

    # Nếu dùng smote, under, hoặc none thì scale_pos_weight luôn là 1.0
    spw_used = spw if imbalance == "weight" else 1.0
    steps.append(("clf", build_xgb(params=params, scale_pos_weight=spw_used)))

    return Pipeline(steps)


def get_clf(pipe: Pipeline) -> XGBClassifier:
    """Trích xuất đối tượng XGBClassifier từ pipeline."""
    return pipe.named_steps["clf"]


def transform_features(pipe: Pipeline, X) -> Any:
    """Biến đổi các đặc trưng qua các bước tiền xử lý của pipeline (bỏ qua sampler)."""
    Xt = X
    for name, step in pipe.steps[:-1]:
        if name == "sampler":
            continue
        Xt = step.transform(Xt)
    return Xt
