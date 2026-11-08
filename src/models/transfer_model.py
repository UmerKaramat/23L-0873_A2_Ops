"""
VisionGuard - Transfer Learning Module
File: src/models/transfer_model.py

Purpose:
Implements Transfer Learning using pre-trained Keras Application (MobileNetV2):
- Feature extraction phase: Freezes backbone, trains classification head with base LR.
- Fine-tuning phase: Unfreezes top convolutional blocks with lower learning rate.
- Incorporates native MobileNetV2 preprocessing.
- Emits trainable layer lists before and after unfreezing as required for Task 2.3 evidence.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_transfer_model(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Constructs a Transfer Learning model using pre-trained MobileNetV2 backbone.
    Initial state: Backbone is completely frozen (Feature Extraction phase).

    Args:
        input_shape (tuple): (224, 224, 3)
        learning_rate (float): Initial learning rate for classifier head.

    Returns:
        tuple: (model, base_model)
    """
    # 1. Base Pre-trained Model (excluding ImageNet classification head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze all weights

    # 2. Functional API architecture with backbone preprocessing
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    # MobileNetV2 expects pixel values scaled to [-1, 1]
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Pass through frozen backbone (explicitly training=False keeps batch norm stats frozen)
    features = base_model(preprocessed, training=False)

    # 3. Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="transfer_gap")(features)
    x = layers.BatchNormalization(name="transfer_bn")(x)
    x = layers.Dense(128, activation="relu", name="transfer_dense")(x)
    x = layers.Dropout(0.3, name="transfer_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="transfer_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VisionGuard_MobileNetV2_Transfer")

    # Compile for initial feature extraction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model, base_model


def unfreeze_for_fine_tuning(model, base_model, unfreeze_layers=30, fine_tune_lr=1e-5):
    """
    Unfreezes the top N layers of the MobileNetV2 backbone for fine-tuning.

    Args:
        model (tf.keras.Model): Overall transfer model.
        base_model (tf.keras.Model): Pre-trained backbone.
        unfreeze_layers (int): Number of layers from the top of the backbone to unfreeze.
        fine_tune_lr (float): Reduced learning rate to avoid destroying pre-trained weights.

    Returns:
        tf.keras.Model: Recompiled model ready for fine-tuning.
    """
    base_model.trainable = True

    # Freeze all layers except the last `unfreeze_layers`
    total_base_layers = len(base_model.layers)
    freeze_until = max(0, total_base_layers - unfreeze_layers)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        # Keep BatchNormalization layers frozen during fine-tuning for stability
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Recompile with small learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model


def print_trainable_layer_status(model, stage_title="Phase"):
    """
    Inspects and prints all trainable vs frozen layers in terminal.
    Required evidence for Task 2.3 in the assignment report.
    """
    print("\n" + "=" * 70)
    print(f"      Trainable Layer Audit: {stage_title}")
    print("=" * 70)

    trainable_layers = []
    frozen_layers = []

    # Check top-level and nested backbone layers
    for layer in model.layers:
        if hasattr(layer, "layers"):  # Nested model (like base_model)
            for sublayer in layer.layers:
                if sublayer.trainable and len(sublayer.weights) > 0:
                    trainable_layers.append(f"{layer.name} -> {sublayer.name} ({sublayer.__class__.__name__})")
                else:
                    frozen_layers.append(f"{layer.name} -> {sublayer.name}")
        else:
            if layer.trainable and len(layer.weights) > 0:
                trainable_layers.append(f"{layer.name} ({layer.__class__.__name__})")
            else:
                frozen_layers.append(layer.name)

    print(f"Total Trainable Layers / Blocks : {len(trainable_layers)}")
    print(f"Total Frozen Layers / Blocks    : {len(frozen_layers)}")
    print("-" * 70)
    print("Trainable Layers List:")
    for name in trainable_layers[-20:]:  # Show top 20 trainable
        print(f"  [TRAINABLE] {name}")
    if len(trainable_layers) > 20:
        print(f"  ... and {len(trainable_layers) - 20} more trainable layers.")
    print("=" * 70 + "\n")


def plot_combined_transfer_curves(initial_history, fine_tune_history,
                                  output_path="reports/figures/task2_3_transfer_curves.png"):
    """
    Combines initial feature extraction curves and fine-tuning curves into a single
    publication-quality figure, with an explicit marker separating the two phases and
    a marker pointing out the overall best checkpoint epoch.
    Required evidence for Task 2.3 and Task 2.4.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate metrics
    init_acc = initial_history.history.get("accuracy", [])
    init_val_acc = initial_history.history.get("val_accuracy", [])
    init_loss = initial_history.history.get("loss", [])
    init_val_loss = initial_history.history.get("val_loss", [])

    fine_acc = fine_tune_history.history.get("accuracy", [])
    fine_val_acc = fine_tune_history.history.get("val_accuracy", [])
    fine_loss = fine_tune_history.history.get("loss", [])
    fine_val_loss = fine_tune_history.history.get("val_loss", [])

    total_acc = init_acc + fine_acc
    total_val_acc = init_val_acc + fine_val_acc
    total_loss = init_loss + fine_loss
    total_val_loss = init_val_loss + fine_val_loss

    epochs_range = range(1, len(total_acc) + 1)
    unfreeze_boundary = len(init_acc)

    # Identify best overall epoch based on validation loss
    best_overall_epoch = int(np.argmin(total_val_loss)) + 1
    best_val_loss = total_val_loss[best_overall_epoch - 1]
    best_val_acc = total_val_acc[best_overall_epoch - 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Accuracy Subplot ---
    ax1.plot(epochs_range, total_acc, label="Training Accuracy", color="#2980b9", linewidth=2)
    ax1.plot(epochs_range, total_val_acc, label="Validation Accuracy", color="#27ae60", linewidth=2)
    # Unfreeze boundary
    ax1.axvline(x=unfreeze_boundary, color="#e67e22", linestyle="--", linewidth=1.5,
                label=f"Fine-Tuning Start (Epoch {unfreeze_boundary})")
    # Best epoch marker
    ax1.axvline(x=best_overall_epoch, color="#c0392b", linestyle=":", linewidth=2,
                label=f"Best Checkpoint (Epoch {best_overall_epoch}: {best_val_acc:.3f})")
    ax1.scatter([best_overall_epoch], [best_val_acc], color="#c0392b", s=70, zorder=5)

    ax1.set_title("Transfer Learning: Accuracy Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # --- Loss Subplot ---
    ax2.plot(epochs_range, total_loss, label="Training Loss", color="#2980b9", linewidth=2)
    ax2.plot(epochs_range, total_val_loss, label="Validation Loss", color="#e74c3c", linewidth=2)
    # Unfreeze boundary
    ax2.axvline(x=unfreeze_boundary, color="#e67e22", linestyle="--", linewidth=1.5,
                label=f"Fine-Tuning Start (Epoch {unfreeze_boundary})")
    # Best epoch marker
    ax2.axvline(x=best_overall_epoch, color="#c0392b", linestyle=":", linewidth=2,
                label=f"Best Checkpoint (Epoch {best_overall_epoch}: {best_val_loss:.3f})")
    ax2.scatter([best_overall_epoch], [best_val_loss], color="#c0392b", s=70, zorder=5)

    ax2.set_title("Transfer Learning: Loss Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Binary Crossentropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("VisionGuard Task 2.3: MobileNetV2 Initial & Fine-Tuning Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual Evidence Generated: {Path(output_path).resolve()}")

