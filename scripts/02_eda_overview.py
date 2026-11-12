"""EDA tổng quan: chất lượng dữ liệu, phân phối lớp, Amount, Time.
Chỉ đọc train.parquet (không đụng test). Biểu đồ lưu vào figures/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # lưu ảnh, không mở cửa sổ
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
LABELS = {0: "Bình thường", 1: "Gian lận"}

train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")

# ---------- 1. Thông tin chung ----------
print("=== 1. THÔNG TIN CHUNG ===")
print("Kích thước:", train.shape)
print("Tổng giá trị thiếu:", int(train.isna().sum().sum()))
print("Số dòng trùng:", int(train.duplicated().sum()))
print("\nKiểu dữ liệu:")
print(train.dtypes.value_counts())
print("\nThống kê Time, Amount:")
print(train[["Time", "Amount"]].describe().round(2))

# ---------- 2. Phân phối lớp ----------
print("\n=== 2. PHÂN PHỐI LỚP ===")
counts = train["Class"].value_counts().sort_index()
ratio = counts[0] / counts[1]
print(counts)
print("Tỉ lệ gian lận: %.4f%%" % (train["Class"].mean() * 100))
print("Tỉ lệ mất cân bằng: %d bình thường : 1 gian lận" % round(ratio))

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=[LABELS[i] for i in counts.index], y=counts.values, ax=ax,
            hue=[LABELS[i] for i in counts.index], legend=False)
ax.set_yscale("log")
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_ylabel("Số giao dịch (thang log)")
ax.set_title("Phân phối lớp (tập train)")
fig.tight_layout()
fig.savefig(FIG / "eda_class_dist.png", dpi=150)
plt.close(fig)

# ---------- 3. Amount ----------
print("\n=== 3. AMOUNT THEO LỚP ===")
print(train.groupby("Class")["Amount"].describe().round(2))
print("Số giao dịch Amount = 0:", int((train["Amount"] == 0).sum()))

train["log_Amount"] = np.log1p(train["Amount"])   # log1p để xử lý Amount = 0
train["Nhãn"] = train["Class"].map(LABELS)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(train["Amount"], bins=100, ax=axes[0])
axes[0].set_title("Amount gốc (lệch phải mạnh)")

sns.histplot(data=train, x="log_Amount", hue="Nhãn", bins=60,
             stat="density", common_norm=False, element="step", ax=axes[1])
axes[1].set_title("log(1 + Amount) theo lớp")

sns.boxplot(data=train, x="Nhãn", y="log_Amount", hue="Nhãn", legend=False, ax=axes[2])
axes[2].set_title("Boxplot log(1 + Amount)")

fig.tight_layout()
fig.savefig(FIG / "eda_amount.png", dpi=150)
plt.close(fig)

# ---------- 4. Time ----------
print("\n=== 4. TIME ===")
print("Khoảng Time: %.0f giây ≈ %.1f giờ" % (train["Time"].max(), train["Time"].max() / 3600))

train["Hour"] = (train["Time"] // 3600) % 24
fraud_by_hour = train.groupby("Hour")["Class"].agg(["count", "sum", "mean"])
fraud_by_hour.columns = ["so_giao_dich", "so_gian_lan", "ty_le_gian_lan"]
print(fraud_by_hour.round(5))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.histplot(data=train, x="Hour", hue="Nhãn", bins=24, stat="density",
             common_norm=False, element="step", ax=axes[0])
axes[0].set_title("Phân phối giao dịch theo giờ trong ngày")
axes[0].set_xlabel("Giờ (Time // 3600 % 24)")

axes[1].bar(fraud_by_hour.index, fraud_by_hour["ty_le_gian_lan"] * 100)
axes[1].set_title("Tỉ lệ gian lận theo giờ (%)")
axes[1].set_xlabel("Giờ")
axes[1].set_ylabel("% gian lận")

fig.tight_layout()
fig.savefig(FIG / "eda_time_hour.png", dpi=150)
plt.close(fig)

print("\nĐã lưu biểu đồ vào:", FIG)