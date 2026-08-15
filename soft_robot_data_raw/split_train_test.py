

#!/usr/bin/env python3
import os
import shutil
import random

def split_dataset(raw_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Randomly split images in raw_dir into train and test folders under output_dir.
    """
    random.seed(seed)
    # List all files in raw_images directory
    images = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
    num_images = len(images)
    num_train = int(num_images * train_ratio)

    # Shuffle and split
    random.shuffle(images)
    train_images = images[:num_train]
    test_images = images[num_train:]

    # Prepare output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy files to respective folders
    for img in train_images:
        shutil.copy2(os.path.join(raw_dir, img), train_dir)
    for img in test_images:
        shutil.copy2(os.path.join(raw_dir, img), test_dir)

    # Print summary
    print(f"Total images: {num_images}")
    print(f"Training images (80%): {len(train_images)}")
    print(f"Test images (20%): {len(test_images)}")

if __name__ == "__main__":
    # Assume raw_images folder in current working directory
    cwd = os.getcwd()
    raw_images_dir = os.path.join(cwd, 'raw_images')
    split_dataset(raw_images_dir, "split_raw_datasets")