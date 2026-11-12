"""Shared loading, validation, cleaning, splitting, and persistence for the data."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)
EXPECTED_COLUMNS = config.FEATURE_COLS + [config.TARGET_COL]


def fetch_raw() -> Path:
    """Keep an existing raw CSV or download OpenML 1597 with the Time column."""
    config.ensure_dirs()
    if config.RAW_CSV.exists():
        logger.info("Sử dụng tệp dữ liệu gốc có sẵn: %s", config.RAW_CSV)
        return config.RAW_CSV

    dataset = fetch_openml(
        data_id=config.OPENML_ID,
        as_frame=True,
        target_column=config.OPENML_FETCH_TARGET,
        parser="auto",
    )
    if dataset.frame is None:
        raise ValueError(f"OpenML ID {config.OPENML_ID} không trả về dữ liệu dạng bảng.")

    frame = dataset.frame.copy()
    _validate_columns(frame)
    frame = frame.loc[:, EXPECTED_COLUMNS]
    frame[config.TARGET_COL] = pd.to_numeric(
        frame[config.TARGET_COL], errors="raise"
    ).astype("int64")
    frame.to_csv(config.RAW_CSV, index=False)
    logger.info("Đã tải dữ liệu gốc về %s (%s)", config.RAW_CSV, frame.shape)
    return config.RAW_CSV


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, downloading the OpenML source when it is absent."""
    fetch_raw()
    return pd.read_csv(config.RAW_CSV)


def _validate_columns(df: pd.DataFrame) -> None:
    if "Time" not in df.columns:
        raise ValueError(
            "Thiếu cột bắt buộc 'Time'; cần đủ 31 cột để xóa dòng trùng chính xác."
        )
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    if missing_columns or extra_columns or len(df.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            "Dữ liệu phải có đúng 31 cột của bộ thẻ tín dụng "
            f"(thiếu={missing_columns}, không mong đợi={extra_columns})."
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate all 31 columns, check missing values, cast Class, and deduplicate."""
    _validate_columns(df)
    cleaned = df.loc[:, EXPECTED_COLUMNS].copy()
    missing = cleaned.isna().sum()
    missing_columns = missing[missing > 0]
    if not missing_columns.empty:
        raise ValueError(
            "Phát hiện giá trị thiếu trong dữ liệu gốc: "
            f"{missing_columns.to_dict()}"
        )

    cleaned[config.TARGET_COL] = pd.to_numeric(
        cleaned[config.TARGET_COL], errors="raise"
    )
    if not cleaned[config.TARGET_COL].isin([0, 1]).all():
        raise ValueError("Class chỉ được chứa hai nhãn 0 và 1.")
    if not (cleaned[config.TARGET_COL] % 1 == 0).all():
        raise ValueError("Class chứa nhãn không phải số nguyên.")
    cleaned[config.TARGET_COL] = cleaned[config.TARGET_COL].astype("int64")

    n_duplicates = int(cleaned.duplicated().sum())
    result = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
    logger.info(
        "Đã kiểm tra %d dòng; giá trị thiếu=0; đã xóa %d dòng trùng; "
        "còn %d dòng (%d gian lận).",
        len(cleaned),
        n_duplicates,
        len(result),
        int(result[config.TARGET_COL].sum()),
    )
    return result


def load_and_clean() -> pd.DataFrame:
    return clean(load_raw())


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible 80/20 split while preserving the Class ratio."""
    _validate_columns(df)
    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL].astype("int64")
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.SEED,
    )


def summarize_preprocess(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, int]:
    """Return the row and fraud counts written to preprocess_summary.csv."""
    return {
        "n_raw": int(len(df_raw)),
        "n_dup": int(df_raw.loc[:, EXPECTED_COLUMNS].duplicated().sum()),
        "n_clean": int(len(df_clean)),
        "fraud_raw": int(pd.to_numeric(df_raw[config.TARGET_COL]).sum()),
        "fraud_clean": int(df_clean[config.TARGET_COL].sum()),
        "train_n": int(len(y_train)),
        "train_fraud": int(y_train.sum()),
        "test_n": int(len(y_test)),
        "test_fraud": int(y_test.sum()),
    }


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    config.ensure_dirs()
    train = X_train.assign(**{config.TARGET_COL: y_train})
    test = X_test.assign(**{config.TARGET_COL: y_test})
    train.to_parquet(config.TRAIN_PATH, index=False)
    test.to_parquet(config.TEST_PATH, index=False)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Read the processed train and test splits as X_train, X_test, y_train, y_test."""
    train = pd.read_parquet(config.TRAIN_PATH)
    test = pd.read_parquet(config.TEST_PATH)
    expected = EXPECTED_COLUMNS
    for name, frame in (("train", train), ("test", test)):
        if frame.columns.tolist() != expected:
            raise ValueError(f"Tệp parquet {name} phải có đủ 31 cột theo quy định.")
        if not pd.api.types.is_integer_dtype(frame[config.TARGET_COL]):
            raise ValueError(f"Cột Class trong tệp parquet {name} phải có kiểu số nguyên.")
    return (
        train[config.FEATURE_COLS],
        test[config.FEATURE_COLS],
        train[config.TARGET_COL],
        test[config.TARGET_COL],
    )


def load_processed_full() -> pd.DataFrame:
    """Read both processed splits and combine them for descriptive EDA."""
    train = pd.read_parquet(config.TRAIN_PATH)
    test = pd.read_parquet(config.TEST_PATH)
    return pd.concat([train, test], ignore_index=True)
