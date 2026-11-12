"""Tạo thống kê mô tả và biểu đồ tổng quan cho dữ liệu."""
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_processed_full

LABELS = {0: "Bình thường", 1: "Gian lận"}

config.ensure_dirs()
sns.set_theme(style="whitegrid")

df = load_processed_full()
if "Time" not in df.columns:
    raise ValueError("Dữ liệu đã xử lý thiếu cột Time.")
if not pd.api.types.is_integer_dtype(df[config.TARGET_COL]):
    raise ValueError("Cột Class trong dữ liệu đã xử lý phải có kiểu số nguyên.")

describe = df.describe().T
describe.to_csv(config.METRICS_DIR / "eda_describe.csv", index_label="feature")

label_counts = (
    df[config.TARGET_COL]
    .value_counts()
    .sort_index()
    .rename_axis("class")
    .reset_index(name="count")
)
label_counts["ratio"] = label_counts["count"] / len(df)
label_counts.to_csv(config.METRICS_DIR / "eda_label_counts.csv", index=False)

plot_labels = [LABELS[int(value)] for value in label_counts["class"]]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(plot_labels, label_counts["count"], color=["#4C78A8", "#E45756"])
ax.set_yscale("log")
ax.set_ylabel("Số giao dịch (thang log)")
ax.set_title("Phân bố nhãn sau khi làm sạch")
for bar, count in zip(bars, label_counts["count"]):
    ax.annotate(
        f"{count:,}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
    )
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "02_label_distribution.png", dpi=config.FIG_DPI)
plt.close(fig)

plot_data = df[["Amount", config.TARGET_COL]].copy()
plot_data["log_amount"] = np.log1p(plot_data["Amount"])
plot_data["label"] = plot_data[config.TARGET_COL].map(LABELS)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
    data=plot_data,
    x="log_amount",
    hue="label",
    bins=60,
    stat="density",
    common_norm=False,
    element="step",
    ax=ax,
)
ax.set_xlabel("log(1 + Amount)")
ax.set_ylabel("Mật độ")
ax.set_title("Phân bố Amount trên thang logarit")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "02_amount_hist.png", dpi=config.FIG_DPI)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(
    data=plot_data,
    x="label",
    y="log_amount",
    order=[LABELS[0], LABELS[1]],
    color="#72B7B2",
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("log(1 + Amount)")
ax.set_title("Boxplot Amount theo nhãn (thang logarit)")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "02_amount_box.png", dpi=config.FIG_DPI)
plt.close(fig)

time_data = df[["Time", config.TARGET_COL]].copy()
time_data["hour_from_start"] = (time_data["Time"] // 3600).astype("int64")
hourly = time_data.groupby("hour_from_start")[config.TARGET_COL].agg(
    transactions="size", fraud_rate="mean"
)
fig, count_ax = plt.subplots(figsize=(10, 5))
count_ax.bar(hourly.index, hourly["transactions"], color="#4C78A8", alpha=0.8)
count_ax.set_xlabel("Giờ kể từ thời điểm bắt đầu (Time // 3600)")
count_ax.set_ylabel("Số giao dịch")
rate_ax = count_ax.twinx()
rate_ax.plot(
    hourly.index,
    hourly["fraud_rate"] * 100,
    color="#E45756",
    marker="o",
    markersize=3,
)
rate_ax.set_ylabel("Tỷ lệ gian lận (%)")
count_ax.set_title("Số giao dịch và tỷ lệ gian lận theo Time")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "02_time_dist.png", dpi=config.FIG_DPI)
plt.close(fig)

console_counts = label_counts.rename(
    columns={
        "class": "Nhãn",
        "count": "Số giao dịch",
        "ratio": "Tỷ lệ",
    }
)
print(f"Kích thước dữ liệu: {df.shape[0]:,} dòng x {df.shape[1]} cột")
print(f"Số giá trị thiếu: {int(df.isna().sum().sum())}")
print(f"Số dòng trùng: {int(df.duplicated().sum())}")
print("Phân bố nhãn:")
print(console_counts.to_string(index=False))
print(f"Đã lưu các bảng EDA tại {config.METRICS_DIR}")
print(f"Đã lưu các biểu đồ tại {config.FIGURES_DIR}")
