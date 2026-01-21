import os
import shutil
import random

# Path to raw dataset (already inside your project)
SOURCE_DIR = r"Covid19_dataset"

COVID_SRC = os.path.join(SOURCE_DIR, "COVID", "images")
NORMAL_SRC = os.path.join(SOURCE_DIR, "Normal", "images")

# Where split data will be stored
TARGET_DIR = "data"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def split_and_copy(src_dir, class_name):
    images = os.listdir(src_dir)
    images = [img for img in images if img.lower().endswith((".png", ".jpg", ".jpeg"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        dest_folder = os.path.join(TARGET_DIR, split, class_name)
        for file in files:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dest_folder, file)
            shutil.copy(src_path, dst_path)

    print(f"{class_name.upper()} -> {total} images split successfully")


# Run splitting
split_and_copy(COVID_SRC, "covid")
split_and_copy(NORMAL_SRC, "normal")

print("✅ Dataset split completed successfully!")
