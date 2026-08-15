import csv
import os
import shutil
import sys

def load_labels(label_file):
    """
    Return dict mapping stripped filename (e.g., "0001.jpg" → "1.jpg") → lighting label.
    """
    mapping = {}
    with open(label_file, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            raw_fn = row['file_name']
            lighting = row['Lighting']
            # Strip leading zeros, e.g., "0001.jpg" → "1.jpg"
            name, ext = os.path.splitext(raw_fn)
            try:
                idx = int(name)
            except ValueError:
                print(f"⚠️  Skipping invalid label filename: {raw_fn}")
                continue
            stripped_fn = f"{idx}{ext}"
            mapping[stripped_fn] = lighting
    return mapping

def process_images(source_dir, target_dir, mapping, lighting_to_idx):
    for filename in os.listdir(source_dir):
        label = mapping.get(filename)
        if label is None:
            print(f"⚠️  No label for {filename}, skipping.")
            continue
        idx = lighting_to_idx.get(label)
        if idx is None:
            print(f"⚠️  No class index for lighting {label}, skipping.")
            continue
        class_dir = os.path.join(target_dir, str(idx))
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy2(os.path.join(source_dir, filename),
                     os.path.join(class_dir, filename))
        continue

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate_il.py <label_file> <raw_train_dir> <raw_test_dir> <output_dir>")
        sys.exit(1)
    label_file, raw_train_dir, raw_test_dir, output_dir = sys.argv[1:]
    mapping = load_labels(label_file)
    # Determine all classes from the mapping values
    classes = sorted(set(mapping.values()))
    # Map each lighting value to a sequential class index 0..4
    lighting_to_idx = {lighting: idx for idx, lighting in enumerate(classes)}
    for subset_name, src_dir in [("train", raw_train_dir), ("test", raw_test_dir)]:
        subset_dir = os.path.join(output_dir, subset_name)
        # Create a subfolder for each class
        for idx in range(len(classes)):
            os.makedirs(os.path.join(subset_dir, str(idx)), exist_ok=True)
        # Copy images into their class-specific subfolders
        process_images(src_dir, subset_dir, mapping, lighting_to_idx)
