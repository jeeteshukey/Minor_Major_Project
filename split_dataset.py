import os
import random
import shutil

import os
import random
import shutil

random.seed(42)

# Paths
BASE_DIR = "datasets/breed_classification"
RAW_DIR = os.path.join(BASE_DIR, "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Loop through each breed folder
for breed in os.listdir(RAW_DIR):

    breed_path = os.path.join(RAW_DIR, breed)

    if not os.path.isdir(breed_path):
        continue

    # ✅ Only take image files
    images = [f for f in os.listdir(breed_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # ✅ Skip empty folders
    if len(images) == 0:
        print(f"{breed} skipped (no images)")
        continue

    random.shuffle(images)

    total = len(images)

    train_split = int(total * train_ratio)
    val_split = int(total * val_ratio)

    train_images = images[:train_split]
    val_images = images[train_split:train_split + val_split]
    test_images = images[train_split + val_split:]

    for img in train_images:
        shutil.move(
            os.path.join(breed_path, img),
            os.path.join(TRAIN_DIR, breed, img)
        )

    for img in val_images:
        shutil.move(
            os.path.join(breed_path, img),
            os.path.join(VAL_DIR, breed, img)
        )

    for img in test_images:
        shutil.move(
            os.path.join(breed_path, img),
            os.path.join(TEST_DIR, breed, img)
        )

    print(f"{breed} split completed")

print("Dataset successfully split!")