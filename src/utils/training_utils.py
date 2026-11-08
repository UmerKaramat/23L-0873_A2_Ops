"""
VisionGuard - Training Utilities & Callbacks Module
File: src/utils/training_utils.py

Purpose:
Enforces rigorous training discipline for Task 2:
- Random seed standardization for deterministic execution.
- Callbacks: EarlyStopping, ModelCheckpoint (monitoring val_loss), ReduceLROnPlateau.
- Visual Evidence Generator: Loss & Accuracy plots with explicit vertical lines/markers
  highlighting the exact "best epoch" saved by the ModelCheckpoint.
"""

import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def set_reproducible_seeds(seed=42):
    """Fixes all random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_standard_callbacks(checkpoint_path, patience_early_stopping=4, patience_lr=2):
    """
    Constructs the standard trio of Keras callbacks for disciplined training.

    Args:
        checkpoint_path (str or Path): Filepath to save best model (e.g. models/custom_cnn_best.keras).
        patience_early_stopping (int): Epochs to wait before early stopping.
        patience_lr (int): Epochs to wait before reducing learning rate.

    Returns:
        list: List of configured Keras callback objects.
    """
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # 1. Save only the model weights with lowest validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        # 2. Reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1
        ),
        # 3. Stop training early if validation loss fails to improve
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1
        )
    ]
    return callbacks


def plot_training_curves_with_best_marker(history, model_name, output_path):
    """
    Plots training vs validation accuracy and loss curves.
    Explicitly places a vertical marker and annotation at the best checkpoint epoch.

    Args:
        history (tf.keras.callbacks.History): Keras training history object.
        model_name (str): Display name for the model in plot titles.
        output_path (str or Path): Destination path for the saved figure.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(loss) + 1)

    # Detect the best epoch saved by ModelCheckpoint (min val_loss)
    best_epoch = int(np.argmin(val_loss)) + 1
    best_loss_val = val_loss[best_epoch - 1]
    best_acc_val = val_acc[best_epoch - 1] if val_acc else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Accuracy ---
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2980b9", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#27ae60", linewidth=2)
    # Vertical line for best epoch
    ax1.axvline(x=best_epoch, color="#c0392b", linestyle="--", linewidth=1.8,
                label=f"Best Model Checkpoint (Epoch {best_epoch})")
    ax1.scatter([best_epoch], [best_acc_val], color="#c0392b", s=80, zorder=5)

    ax1.set_title(f"{model_name}: Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # --- Subplot 2: Loss ---
    ax2.plot(epochs_range, loss, label="Training Loss", color="#2980b9", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#e74c3c", linewidth=2)
    # Vertical line for best epoch
    ax2.axvline(x=best_epoch, color="#c0392b", linestyle="--", linewidth=1.8,
                label=f"Best Model Checkpoint (Epoch {best_epoch}: {best_loss_val:.4f})")
    ax2.scatter([best_epoch], [best_loss_val], color="#c0392b", s=80, zorder=5)

    ax2.set_title(f"{model_name}: Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Binary Crossentropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle(f"VisionGuard Task 2.4: {model_name} Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual Evidence Generated: {Path(output_path).resolve()}")

