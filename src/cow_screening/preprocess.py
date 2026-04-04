import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIG
# =========================
base_path = r"D:\Final_Year_Project\Cattle_Breed\datasets\cow_screening\raw"

output_base = r"D:\Final_Year_Project\Cattle_Breed\datasets\cow_screening\processed"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
test_dir = os.path.join(output_base, "test")

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

window_size = 256
stride = window_size // 2
downsample_step = 2
chunk_size = 5000

# =========================
# WINDOW FUNCTION
# =========================
def make_windows(df, window_size, stride):
    signals = df.values
    windows = []
    for i in range(0, len(signals) - window_size, stride):
        windows.append(signals[i:i + window_size])
    return np.array(windows, dtype=np.float32)

# =========================
# LOAD ALL DATA FIRST
# =========================
all_X = []
all_y = []

print("🚀 Preprocessing + collecting data...")

for folder in tqdm(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(folder_path, file)

        try:
            df = pd.read_csv(path, sep=r'\s+', engine='python')

            if df.shape[1] == 1:
                df = df.iloc[:, 0].str.split(expand=True)

            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            if len(df) < window_size:
                continue

            df = df.astype(np.float32)
            df = (df - df.mean()) / (df.std() + 1e-8)
            df = df.iloc[::downsample_step, :]

            windows = make_windows(df, window_size, stride)
            if len(windows) == 0:
                continue

            match = re.search(r"Illnessdegree_(\d+)", file)
            label = int(match.group(1)) if match else 0

            keep_ratio = 0.2 if label == 0 else 0.4
            sample_size = max(1, int(len(windows) * keep_ratio))

            idx = np.random.choice(len(windows), sample_size, replace=False)
            sampled = windows[idx]

            for w in sampled:
                all_X.append(w)
                all_y.append(label)

        except Exception as e:
            print(f"⚠️ {file}: {e}")

# =========================
# CONVERT TO ARRAYS
# =========================
X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.int64)

print(f"Total samples: {len(X)}")

# =========================
# SHUFFLE
# =========================
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# =========================
# SPLIT (75/15/10)
# =========================
n = len(X)
train_end = int(0.75 * n)
val_end = int(0.90 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# =========================
# SAVE FUNCTION
# =========================
def save_chunks(X, y, folder, prefix):
    idx = 0
    for i in range(0, len(X), chunk_size):
        X_chunk = X[i:i+chunk_size]
        y_chunk = y[i:i+chunk_size]

        np.savez_compressed(
            os.path.join(folder, f"{prefix}_chunk_{idx}.npz"),
            X=X_chunk,
            y=y_chunk
        )
        idx += 1

# =========================
# SAVE ALL SPLITS
# =========================
save_chunks(X_train, y_train, train_dir, "train")
save_chunks(X_val, y_val, val_dir, "val")
save_chunks(X_test, y_test, test_dir, "test")

print("\nDONE!")
