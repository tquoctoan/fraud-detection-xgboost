"""EDA đặc trưng: V1-V28, tương quan với Class, xu hướng theo Time và Amount.

Chỉ để MÔ TẢ. Không dùng kết quả ở đây để loại cột khỏi mô hình.
"""
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_processed_full

LABELS = {0: "Bình thường", 1: "Gian lận"}
PALETTE = {LABELS[0]: "#4C78A8", LABELS[1]: "#E45756"}
TOP_BAR = 15              # số đặc trưng trong biểu đồ cột tương quan
MIN_TX_RELIABLE = 500     # giờ có ít hơn số giao dịch này thì tỷ lệ gian lận rất nhiễu


def save_fig(fig, name: str, rect=None) -> None:
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=config.FIG_DPI)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    sns.set_theme(style="whitegrid")

    df = load_processed_full()
    target = config.TARGET_COL
    cols = config.FEATURE_COLS + [target]

    # 1. Tương quan với Class -> bảng CSV
    corr_matrix = df[cols].corr()
    corr = corr_matrix[target].drop(target)
    corr_table = (
        pd.DataFrame(
            {"feature": corr.index, "corr": corr.values, "abs_corr": corr.abs().values}
        )
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    corr_table.to_csv(config.METRICS_DIR / "eda_corr_with_class.csv", index=False)

    # 2. Ma trận tương quan 31x31
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, cmap="coolwarm", center=0, annot=False, square=True,
        cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title("Ma trận tương quan (30 đặc trưng và Class)")
    save_fig(fig, "03_correlation_matrix.png")

    # 3. Top đặc trưng theo |tương quan| với Class
    top = corr_table.head(TOP_BAR).iloc[::-1]
    colors = ["#E45756" if value > 0 else "#4C78A8" for value in top["corr"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["corr"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hệ số tương quan Pearson với Class")
    ax.set_title(f"{TOP_BAR} đặc trưng có |tương quan| với Class lớn nhất")
    save_fig(fig, "03_top_corr_features.png")

    # 4. Phân bố hai lớp của các đặc trưng đầu bảng (chỉ minh họa)
    top_features = corr_table["feature"].head(config.EDA_TOP_FEATURES).tolist()
    plot_data = df[top_features + [target]].copy()
    plot_data["label"] = plot_data[target].map(LABELS)
    n_cols = 5
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False
    )
    for ax, feature in zip(axes.ravel(), top_features):
        sns.histplot(
            data=plot_data, x=feature, hue="label",
            hue_order=[LABELS[0], LABELS[1]], palette=PALETTE,
            bins=50, stat="density", common_norm=False, element="step",
            legend=False, ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Mật độ")
    for ax in axes.ravel()[len(top_features):]:
        ax.set_visible(False)
    fig.legend(
        handles=[Patch(color=PALETTE[LABELS[k]], label=LABELS[k]) for k in (0, 1)],
        loc="upper right",
    )
    fig.suptitle(
        f"Phân bố hai lớp của {len(top_features)} đặc trưng có |tương quan với Class| "
        "lớn nhất (chỉ minh họa)"
    )
    save_fig(fig, "03_feature_dist_top.png", rect=[0, 0, 1, 0.94])

    # 5. Số giao dịch và tỷ lệ gian lận theo giờ
    time_data = df[["Time", target]].copy()
    time_data["hour"] = (time_data["Time"] // 3600).astype("int64")
    hourly = (
        time_data.groupby("hour")[target]
        .agg(n_transactions="size", n_fraud="sum", fraud_rate="mean")
        .reset_index()
    )
    hourly.to_csv(config.METRICS_DIR / "eda_fraud_by_hour.csv", index=False)
    reliable = hourly["n_transactions"] >= MIN_TX_RELIABLE

    fig, count_ax = plt.subplots(figsize=(10, 5))
    count_ax.bar(hourly["hour"], hourly["n_transactions"], color="#4C78A8", alpha=0.7)
    count_ax.set_xlabel("Giờ kể từ thời điểm bắt đầu (Time // 3600)")
    count_ax.set_ylabel("Số giao dịch")
    rate_ax = count_ax.twinx()
    rate_ax.plot(hourly["hour"], hourly["fraud_rate"] * 100, color="#E45756", alpha=0.4)
    rate_ax.scatter(
        hourly.loc[reliable, "hour"], hourly.loc[reliable, "fraud_rate"] * 100,
        color="#E45756", label=f"≥ {MIN_TX_RELIABLE} giao dịch",
    )
    rate_ax.scatter(
        hourly.loc[~reliable, "hour"], hourly.loc[~reliable, "fraud_rate"] * 100,
        color="gray", marker="x", label=f"< {MIN_TX_RELIABLE} giao dịch (nhiễu)",
    )
    rate_ax.set_ylabel("Tỷ lệ gian lận (%)")
    rate_ax.grid(False)
    rate_ax.legend(loc="upper right")
    count_ax.set_title("Số giao dịch và tỷ lệ gian lận theo giờ")
    save_fig(fig, "03_fraud_vs_time.png")

    # 6. Tỷ lệ gian lận theo khoảng Amount
    bins = [-0.01, 1, 10, 50, 100, 500, 1000, np.inf]
    bin_labels = ["≤1", "1–10", "10–50", "50–100", "100–500", "500–1000", ">1000"]
    amount_data = df[[config.AMOUNT_COL, target]].copy()
    amount_data["amount_bin"] = pd.cut(
        amount_data[config.AMOUNT_COL], bins=bins, labels=bin_labels
    )
    by_amount = (
        amount_data.groupby("amount_bin", observed=False)[target]
        .agg(n_transactions="size", n_fraud="sum", fraud_rate="mean")
        .reset_index()
        .fillna({"fraud_rate": 0.0})
    )
    by_amount.to_csv(config.METRICS_DIR / "eda_fraud_by_amount.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    rates = by_amount["fraud_rate"] * 100
    bars = ax.bar(by_amount["amount_bin"].astype(str), rates, color="#E45756")
    for bar, rate, n in zip(bars, rates, by_amount["n_transactions"]):
        ax.annotate(
            f"{rate:.2f}%\n(n={n:,})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, rates.max() * 1.25)
    ax.set_xlabel("Khoảng Amount")
    ax.set_ylabel("Tỷ lệ gian lận (%)")
    ax.set_title("Tỷ lệ gian lận theo khoảng Amount")
    save_fig(fig, "03_fraud_vs_amount.png")

    print("5 đặc trưng có |tương quan với Class| lớn nhất:")
    print(corr_table.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Số giờ có < {MIN_TX_RELIABLE} giao dịch: {int((~reliable).sum())}/{len(hourly)}")
    print(f"Đã lưu các bảng tại {config.METRICS_DIR}")
    print(f"Đã lưu các biểu đồ tại {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()