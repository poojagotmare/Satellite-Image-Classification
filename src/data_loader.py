"""
Module 1: Data Acquisition, Cleaning & Preparation
Handles dataset verification, train/validation/test splitting,
resizing, color normalization, and data augmentation.
"""

import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

def clean_and_verify_dataset(data_dir="data"):
    """
    Scans through all images in data_dir, identifies any corrupted images
    or non-RGB color modes (e.g. CMYK), and converts them to standard RGB.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    total_scanned = 0
    converted_cmyk = 0
    corrupted_files = []

    for cat in sorted(categories):
        cat_path = os.path.join(data_dir, cat)
        files = [f for f in os.listdir(cat_path) if f.lower().endswith(valid_extensions)]
        print(f"[Verification] Checking class '{cat}': {len(files)} files found.")
        
        for fname in files:
            fpath = os.path.join(cat_path, fname)
            total_scanned += 1
            try:
                with Image.open(fpath) as img:
                    img.verify()  # verify integrity
                
                # Check color mode
                with Image.open(fpath) as img:
                    if img.mode != 'RGB':
                        rgb_img = img.convert('RGB')
                        rgb_img.save(fpath)
                        converted_cmyk += 1
            except Exception as e:
                print(f"[Warning] Corrupted image detected and excluded: {fpath} ({e})")
                corrupted_files.append(fpath)
                try:
                    os.remove(fpath)
                except Exception:
                    pass

    print(f"\n[Verification Summary]")
    print(f"Total scanned: {total_scanned}")
    print(f"Non-RGB / CMYK converted to RGB: {converted_cmyk}")
    print(f"Corrupted files removed: {len(corrupted_files)}")
    return total_scanned - len(corrupted_files)


def build_augmentation_layer():
    """
    Data augmentation specifically tuned for satellite / aerial imagery:
    rotations and horizontal/vertical flips preserve land cover semantics.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="satellite_augmentation")


def load_dataset_splits(
    data_dir="data",
    img_size=(128, 128),
    batch_size=32,
    seed=42
):
    """
    Loads images, splits into 80% train, 10% validation, 10% test,
    applies augmentation to training set, and optimizes pipeline throughput.
    """
    # 1. Load full dataset using image_dataset_from_directory
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=True,
        seed=seed
    )
    
    class_names = full_ds.class_names
    total_batches = tf.data.experimental.cardinality(full_ds).numpy()

    # 2. Partition into 80% train, 10% val, 10% test
    train_size = int(0.80 * total_batches)
    val_size = int(0.10 * total_batches)
    test_size = total_batches - train_size - val_size

    train_ds = full_ds.take(train_size)
    remaining_ds = full_ds.skip(train_size)
    val_ds = remaining_ds.take(val_size)
    test_ds = remaining_ds.skip(val_size)

    # 3. Augmentation on training data
    augmentation = build_augmentation_layer()
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # 4. Pipeline caching and prefetching for CPU/GPU efficiency
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    split_info = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "total_batches": total_batches,
        "train_batches": train_size,
        "val_batches": val_size,
        "test_batches": test_size,
        "batch_size": batch_size,
        "img_size": img_size
    }

    return train_ds, val_ds, test_ds, split_info


if __name__ == "__main__":
    print("--- Running Module 1: Data Verification & Loader Test ---")
    clean_and_verify_dataset("data")
    train_ds, val_ds, test_ds, info = load_dataset_splits("data")
    print(f"\nClasses: {info['class_names']}")
    print(f"Batches - Train: {info['train_batches']}, Val: {info['val_batches']}, Test: {info['test_batches']}")
    
    # Inspect a single batch
    for images, labels in train_ds.take(1):
        print(f"Sample Batch Image Shape: {images.shape}")
        print(f"Sample Batch Label Shape: {labels.shape}")
        print("Module 1 completed successfully!")
        break
