"""EDA đặc trưng: tương quan với Class, phân phối V theo lớp, outlier.
Chỉ đọc train.parquet (không đụng test). Biểu đồ lưu vào figures/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
LABELS = {0: "Bình thường", 1: "Gian lận"}

train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time"] + V_COLS + ["Amount"]

# ---------- 1. Tương quan với Class ----------
print("=== 1. TƯƠNG QUAN VỚI CLASS (Pearson) ===")
corr = train[FEATURES + ["Class"]].corr()["Class"].drop("Class").sort_values()
print(corr.round(4).to_string())

fig, ax = plt.subplots(figsize=(6, 9))
colors = ["tab:red" if v < 0 else "tab:blue" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.set_title("Tương quan Pearson với Class")
ax.set_xlabel("Hệ số tương quan")
fig.tight_layout()
fig.savefig(FIG / "eda_corr_class.png", dpi=150)
plt.close(fig)

# ---------- 2. Khả năng phân tách từng đặc trưng (AUC đơn biến) ----------
# AUC = 0.5: không phân tách; càng xa 0.5 càng phân tách tốt (dùng |AUC - 0.5|)
print("\n=== 2. KHẢ NĂNG PHÂN TÁCH ĐƠN BIẾN (AUC) ===")
auc = {c: roc_auc_score(train["Class"], train[c]) for c in FEATURES}
sep = pd.DataFrame({"AUC": pd.Series(auc)})
sep["do_phan_tach"] = (sep["AUC"] - 0.5).abs()
sep = sep.sort_values("do_phan_tach", ascending=False)
print(sep.head(12).round(4).to_string())

# ---------- 3. Phân phối các đặc trưng phân tách tốt nhất theo lớp ----------
top = sep.index[:8].tolist()
print("\nTop 8 đặc trưng phân tách tốt:", top)

train["Nhãn"] = train["Class"].map(LABELS)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), top):
    sns.kdeplot(data=train, x=col, hue="Nhãn", common_norm=False,
                fill=True, alpha=0.3, ax=ax)
    ax.set_title(f"{col} (AUC={auc[col]:.3f})")
fig.tight_layout()
fig.savefig(FIG / "eda_top_features_kde.png", dpi=150)
plt.close(fig)

# ---------- 4. Tương quan giữa các đặc trưng ----------
print("\n=== 4. TƯƠNG QUAN GIỮA CÁC ĐẶC TRƯNG ===")
cm = train[FEATURES].corr()
mask = np.triu(np.ones_like(cm, dtype=bool))
off_diag = cm.where(~mask).stack().abs().sort_values(ascending=False)
print("Top 8 cặp tương quan mạnh nhất:")
print(off_diag.head(8).round(4).to_string())

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(cm, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Ma trận tương quan giữa các đặc trưng")
fig.tight_layout()
fig.savefig(FIG / "eda_corr_heatmap.png", dpi=150)
plt.close(fig)

# ---------- 5. Outlier ----------
print("\n=== 5. OUTLIER (quy tắc IQR, tính trên toàn train) ===")
q1, q3 = train[FEATURES].quantile(0.25), train[FEATURES].quantile(0.75)
iqr = q3 - q1
is_out = (train[FEATURES] < q1 - 1.5 * iqr) | (train[FEATURES] > q3 + 1.5 * iqr)
out_pct = (is_out.mean() * 100).sort_values(ascending=False)
print("Top 8 đặc trưng nhiều outlier nhất (% dòng):")
print(out_pct.head(8).round(2).to_string())

# Trong nhóm gian lận, bao nhiêu % có ít nhất một đặc trưng là outlier?
fraud_mask = train["Class"] == 1
print("\n%% giao dịch có >=1 outlier: bình thường = %.1f%%, gian lận = %.1f%%" % (
    is_out[~fraud_mask].any(axis=1).mean() * 100,
    is_out[fraud_mask].any(axis=1).mean() * 100))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=train[V_COLS], orient="h", fliersize=1, ax=axes[0])
axes[0].set_title("Boxplot V1-V28 (toàn train)")
sns.boxplot(data=train[top + ["Class"]].melt(id_vars="Class"),
            x="value", y="variable", hue="Class", orient="h",
            fliersize=1, ax=axes[1])
axes[1].set_title("Boxplot top đặc trưng theo lớp (0 = bình thường, 1 = gian lận)")
axes[1].set_xlabel("Giá trị")
axes[1].set_ylabel("")
fig.tight_layout()
fig.savefig(FIG / "eda_outliers.png", dpi=150)
plt.close(fig)

print("\nĐã lưu biểu đồ vào:", FIG)