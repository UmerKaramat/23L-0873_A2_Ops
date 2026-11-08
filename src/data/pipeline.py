"""
VisionGuard - tf.data Input Pipeline Module
File: src/data/pipeline.py

Purpose:
Builds a high-performance, reproducible tf.data input pipeline for:
- Discovering image files and labels from data/raw/AiArtData and data/raw/RealArt.
- Stratified Train / Validation / Test splitting (70 / 15 / 15).
- Image decoding, resizing to 224x224, and tensor casting.
- Training-only data augmentation (flip, rotation, zoom, contrast).
- Batching, caching, and prefetching.
- Generating augmented batch visual evidence for reports.
"""

import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Standard image dimensions and classes
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Label mapping: Real = 0, AI = 1
CLASS_NAMES = ["Real", "AI"]
LABEL_MAP = {
    "realart": 0,
    "real": 0,
    "aiartdata": 1,
    "ai": 1,
}


def set_seed(seed=RANDOM_SEED):
    """Sets random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def discover_image_paths_and_labels(data_dir="data/raw"):
    """
    Recursively scans the raw dataset directory to find images and infer class labels.

    Args:
        data_dir (str or Path): Path to the raw dataset root.

    Returns:
        tuple: (file_paths, labels) where labels are 0 (Real) and 1 (AI).
    """
    data_path = Path(data_dir)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    file_paths = []
    labels = []

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_path.resolve()}' not found. "
            "Ensure the dataset is extracted into data/raw/AiArtData and data/raw/RealArt."
        )

    for root, _, files in os.walk(data_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions and not file.startswith("."):
                full_path = Path(root) / file
                parent_name = full_path.parent.name.lower()

                # Infer label from folder name
                if "ai" in parent_name or "fake" in parent_name or "synth" in parent_name:
                    file_paths.append(str(full_path))
                    labels.append(1)  # AI-Generated
                elif "real" in parent_name or "orig" in parent_name or "authentic" in parent_name:
                    file_paths.append(str(full_path))
                    labels.append(0)  # Real

    if len(file_paths) == 0:
        raise ValueError(
            f"No valid image files found in '{data_path.resolve()}'. "
            "Please check folder naming (expected: AiArtData and RealArt)."
        )

    return file_paths, labels


def get_dataset_splits(data_dir="data/raw", test_size=0.15, val_size=0.15, seed=RANDOM_SEED):
    """
    Creates stratified Train / Validation / Test splits (default: 70% train, 15% val, 15% test).

    Returns:
        dict: Containing train, val, and test file path lists and labels.
    """
    file_paths, labels = discover_image_paths_and_labels(data_dir)

    # First split: Train + Val vs Test (e.g., 85% vs 15%)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    # Second split: Train vs Val (e.g. 70% of total vs 15% of total)
    adjusted_val_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=adjusted_val_size,
        stratify=train_val_labels,
        random_state=seed
    )

    print("-" * 65)
    print(f"Dataset Split Summary (Seed={seed}):")
    print(f"  Training Set   : {len(train_paths):6d} images ({len(train_paths)/len(file_paths)*100:.1f}%)")
    print(f"  Validation Set : {len(val_paths):6d} images ({len(val_paths)/len(file_paths)*100:.1f}%)")
    print(f"  Test Set       : {len(test_paths):6d} images ({len(test_paths)/len(file_paths)*100:.1f}%)")
    print(f"  Total Images   : {len(file_paths):6d} images")
    print("-" * 65)

    return {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "val_paths": val_paths,
        "val_labels": val_labels,
        "test_paths": test_paths,
        "test_labels": test_labels,
    }


def parse_and_decode_image(path, label, img_size=IMG_SIZE):
    """
    Loads, decodes, resizes, and casts a single image tensor.
    """
    image_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(image_bytes, channels=3, try_recover_truncated=True, acceptable_fraction=0.5)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    img.set_shape([img_size[0], img_size[1], 3])
    return img, label


def build_augmentation_layer():
    """
    Constructs a Keras Sequential layer for training-time data augmentation.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="training_augmentation")


def create_tf_dataset(paths, labels, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
                      is_training=False, augment=False, shuffle_buffer=1000):
    """
    Builds an optimized tf.data.Dataset pipeline.
    Augmentation is applied ONLY when is_training=True and augment=True.
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), shuffle_buffer), seed=RANDOM_SEED)

    # Parallel mapping
    dataset = dataset.map(
        lambda p, l: parse_and_decode_image(p, l, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # FIX: Instruct TensorFlow to silently drop any fatally corrupted Kaggle JPEGs
    dataset = dataset.ignore_errors()

    # Batching
    dataset = dataset.batch(batch_size)

    # Apply data augmentation strictly to training data
    if is_training and augment:
        augmentation_layer = build_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Prefetching for CPU execution efficiency
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def save_augmented_batch_grid(dataset, output_path="reports/figures/task2_1_augmented_batch.png", grid_size=(3, 3)):
    """
    Extracts one batch of images from the training dataset and saves a 3x3 grid plot.
    This serves as visual evidence for Task 2.1 in the assignment report.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch one batch
    for images, labels in dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        break

    total_images = grid_size[0] * grid_size[1]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    axes = axes.flatten()

    for i in range(total_images):
        if i < len(images_np):
            img = images_np[i]

            # Normalize to [0, 1] for matplotlib display if values are in [0, 255]
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)

            label_idx = int(labels_np[i])
            label_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"Class {label_idx}"

            axes[i].imshow(img)
            axes[i].set_title(f"Augmented Sample #{i+1}\nLabel: {label_name} ({label_idx})", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("VisionGuard Task 2.1: Augmented Training Batch Visualization", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual Evidence Generated: {Path(output_path).resolve()}")

