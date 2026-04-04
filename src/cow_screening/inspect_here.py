import os
import numpy as np

base_dir = r"D:\Final_Year_Project\Cattle_Breed\datasets\cow_screening\processed"

splits = ["train", "val", "test"]

total_counts = {}

for split in splits:
    folder = os.path.join(base_dir, split)

    print(f"\n📂 Checking {split.upper()} folder: {folder}")

    if not os.path.exists(folder):
        print("❌ Folder not found!")
        continue

    files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])

    if not files:
        print("❌ No NPZ files found!")
        continue

    print(f"✅ Found {len(files)} files")

    split_counts = {}
    total_samples = 0

    for file in files:
        path = os.path.join(folder, file)
        data = np.load(path)

        X = data["X"]
        y = data["y"]

        # Basic checks
        print(f"\n📄 File: {file}")
        print("   X shape:", X.shape)
        print("   y shape:", y.shape)

        total_samples += len(y)

        # Count labels
        unique, counts = np.unique(y, return_counts=True)

        for u, c in zip(unique, counts):
            split_counts[u] = split_counts.get(u, 0) + c
            total_counts[u] = total_counts.get(u, 0) + c

    print(f"\n📊 {split.upper()} SUMMARY")
    print("Total samples:", total_samples)
    print("Class distribution:", split_counts)

# =====================
# FINAL SUMMARY
# =====================
print("\n==============================")
print("📊 OVERALL DATA SUMMARY")
print("==============================")

total_all = sum(total_counts.values())
print("Total samples (all splits):", total_all)
print("Overall class distribution:", total_counts)