"""Tải, kiểm tra, làm sạch, chia tập và lưu dữ liệu thẻ tín dụng."""
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


logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
config.ensure_dirs()

raw = load_raw()
if "Time" not in raw.columns:
    raise ValueError("Dữ liệu gốc thiếu cột Time; không thể xóa trùng an toàn.")
if raw.shape != (config.EXPECTED_N_RAW, 31):
    raise ValueError(
        f"Cần {config.EXPECTED_N_RAW} dòng và 31 cột, nhưng nhận được {raw.shape}."
    )

cleaned = clean(raw)
if len(cleaned) != config.EXPECTED_N_CLEAN:
    raise ValueError(
        f"Cần {config.EXPECTED_N_CLEAN} dòng sau khi xóa trùng, "
        f"nhưng nhận được {len(cleaned)}."
    )
if int(cleaned[config.TARGET_COL].sum()) != config.EXPECTED_FRAUD_CLEAN:
    raise ValueError(
        "Cần 473 ca gian lận sau khi làm sạch, nhưng nhận được "
        f"{int(cleaned[config.TARGET_COL].sum())}."
    )

X_train, X_test, y_train, y_test = split_data(cleaned)
save_processed(X_train, X_test, y_train, y_test)

summary = summarize_preprocess(raw, cleaned, y_train, y_test)
summary_frame = pd.DataFrame([summary])
summary_frame.to_csv(config.METRICS_DIR / "preprocess_summary.csv", index=False)
console_labels = {
    "n_raw": "Số dòng gốc",
    "n_dup": "Số dòng trùng",
    "n_clean": "Số dòng sau làm sạch",
    "fraud_raw": "Số ca gian lận ban đầu",
    "fraud_clean": "Số ca gian lận sau làm sạch",
    "train_n": "Số dòng train",
    "train_fraud": "Số ca gian lận train",
    "test_n": "Số dòng test",
    "test_fraud": "Số ca gian lận test",
}
print("Tóm tắt tiền xử lý:")
print(summary_frame.rename(columns=console_labels).to_string(index=False))
print(f"Đã lưu dữ liệu đã xử lý tại {config.DATA_PROCESSED}")
