"""Download, validate, deduplicate, split, and save the credit-card data."""
import logging
import sys
from pathlib import Path

import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import clean, load_raw, save_processed, split_data, summarize_preprocess


def main() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    config.ensure_dirs()

    raw = load_raw()
    if "Time" not in raw.columns:
        raise ValueError("Raw dataset is missing Time; refusing to deduplicate incomplete rows.")
    if raw.shape != (config.EXPECTED_N_RAW, 31):
        raise ValueError(
            f"Expected {config.EXPECTED_N_RAW} rows and 31 columns, got {raw.shape}."
        )

    cleaned = clean(raw)
    if len(cleaned) != config.EXPECTED_N_CLEAN:
        raise ValueError(
            f"Expected {config.EXPECTED_N_CLEAN} rows after deduplication, got {len(cleaned)}."
        )
    if int(cleaned[config.TARGET_COL].sum()) != config.EXPECTED_FRAUD_CLEAN:
        raise ValueError(
            "Expected 473 fraud cases after deduplication, got "
            f"{int(cleaned[config.TARGET_COL].sum())}."
        )

    X_train, X_test, y_train, y_test = split_data(cleaned)
    save_processed(X_train, X_test, y_train, y_test)

    summary = summarize_preprocess(raw, cleaned, y_train, y_test)
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(
        config.METRICS_DIR / "preprocess_summary.csv", index=False
    )
    console_labels = {
        "n_raw": "Số dòng gốc",
        "n_dup": "Số dòng trùng",
        "n_clean": "Số dòng sau làm sạch",
        "fraud_raw": "Số ca gian lận ban đầu",
        "fraud_clean": "Số ca gian lận sau làm sạch",
        "train_n": "Số dòng tập huấn luyện",
        "train_fraud": "Số ca gian lận tập huấn luyện",
        "test_n": "Số dòng tập kiểm thử",
        "test_fraud": "Số ca gian lận tập kiểm thử",
    }
    print("Tóm tắt tiền xử lý:")
    print(summary_frame.rename(columns=console_labels).to_string(index=False))
    print(f"Đã lưu dữ liệu đã xử lý tại {config.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
