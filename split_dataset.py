import os
import random
import shutil


# Source folder containing all unknown images
source_folder = r"D:\Final_Year_Project\Dataset\Unknown"


# Destination folders
base_path = r"D:\Final_Year_Project\Cattle_Breed\datasets\breed_classification_2"

train_folder = os.path.join(base_path, "train", "Unknown")
val_folder = os.path.join(base_path, "val", "Unknown")
test_folder = os.path.join(base_path, "test", "Unknown")


# Create folders if not exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


# Supported image formats
valid_extensions = (".jpg", ".jpeg", ".png", ".webp")


# Read all images
images = [
    img for img in os.listdir(source_folder)
    if img.lower().endswith(valid_extensions)
]


# Shuffle images randomly
random.shuffle(images)


# Total images
total_images = len(images)

print(f"Total Images Found: {total_images}")


# Split sizes
train_split = int(total_images * 0.78)
val_split = int(total_images * 0.14)
test_split = total_images - train_split - val_split


# Split images
train_images = images[:train_split]
val_images = images[train_split:train_split + val_split]
test_images = images[train_split + val_split:]


# Function to copy images
def copy_images(image_list, destination_folder):

    for image in image_list:

        src_path = os.path.join(source_folder, image)

        dst_path = os.path.join(destination_folder, image)

        shutil.copy2(src_path, dst_path)


# Copy files
copy_images(train_images, train_folder)
copy_images(val_images, val_folder)
copy_images(test_images, test_folder)


# Final counts
print("\nDataset Split Completed\n")

print(f"Train Images: {len(train_images)}")
print(f"Validation Images: {len(val_images)}")
print(f"Test Images: {len(test_images)}")