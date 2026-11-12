from pathlib import Path
import pandas as pd
import openml
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent      # thư mục gốc project
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Bước 1: Tải dữ liệu OpenML 1597 (đủ 31 cột, có Time) ----------
ds = openml.datasets.get_dataset(1597)
df, _, _, _ = ds.get_data(include_row_id=True, include_ignore_attribute=True)

# Nhãn trên OpenML là chuỗi '0'/'1' → đổi sang int
df["Class"] = df["Class"].astype(int)

print("Shape gốc:", df.shape)                      # (284807, 31)
print("Cột:", list(df.columns))
print(df["Class"].value_counts())                  # 0: 284315, 1: 492
assert "Time" in df.columns, "Thiếu cột Time"
assert df.shape == (284807, 31)

# ---------- Bước 2: Xóa dòng trùng lặp ----------
n_dup = df.duplicated().sum()
print("Số dòng trùng lặp:", n_dup)                 # 1081
assert n_dup == 1081

df = df.drop_duplicates(keep="first").reset_index(drop=True)
print("Shape sau khi xóa trùng:", df.shape)        # (283726, 31)
assert len(df) == 283726

# ---------- Bước 3: Chia 80/20 Stratified và lưu parquet ----------
X = df.drop(columns="Class")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

train.to_parquet(OUT_DIR / "train.parquet", index=False)
test.to_parquet(OUT_DIR / "test.parquet", index=False)

print("Train:", train.shape, "| tỉ lệ gian lận: %.4f%%" % (train["Class"].mean() * 100))
print("Test :", test.shape, "| tỉ lệ gian lận: %.4f%%" % (test["Class"].mean() * 100))