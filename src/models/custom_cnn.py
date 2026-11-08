"""
VisionGuard - Baseline Custom CNN Module
File: src/models/custom_cnn.py

Purpose:
Implements a clean, modular Convolutional Neural Network (CNN) from scratch:
- Stacks of Conv2D, BatchNormalization, MaxPooling2D, and Dropout layers.
- Sigmoid output for binary classification (Real vs AI).
- Complies with loss, optimizer, and standard metrics (Accuracy, Precision, Recall, AUC).
- Exports model parameter details to CSV for Task 2.2 evidence generation.
"""

from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Constructs and compiles a lightweight, effective Custom CNN for image classification.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        learning_rate (float): Initial Adam optimizer learning rate.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = models.Sequential(name="VisionGuard_Custom_CNN")

    # Input & Rescaling: [0, 255] -> [0, 1]
    model.add(layers.Input(shape=input_shape, name="input_layer"))
    model.add(layers.Rescaling(1.0 / 255.0, name="rescaling_layer"))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))

    # Block 4
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4"))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4"))

    # Classifier Head
    model.add(layers.GlobalAveragePooling2D(name="gap"))
    model.add(layers.Dense(128, activation="relu", name="dense_features"))
    model.add(layers.Dropout(0.4, name="dropout_classifier"))
    model.add(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))

    # Compile with standard classification metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model


def export_model_summary_to_csv(model, output_csv="reports/task2_2_cnn_params.csv"):
    """
    Exports layer names, layer types, output shapes, and parameter counts to a CSV file.
    This serves as the required evidence for Task 2.2 in the report.

    Args:
        model (tf.keras.Model): Model to inspect.
        output_csv (str or Path): Destination path for CSV.
    """
    records = []
    total_params = 0
    trainable_params = 0

    for idx, layer in enumerate(model.layers, start=1):
        # Format output shape cleanly
        output_shape = str(getattr(layer, "output_shape", "Unknown"))
        params = layer.count_params()
        trainable = layer.trainable

        total_params += params
        if trainable:
            trainable_params += params

        records.append({
            "Layer_Index": idx,
            "Layer_Name": layer.name,
            "Layer_Type": layer.__class__.__name__,
            "Output_Shape": output_shape,
            "Parameter_Count": params,
            "Is_Trainable": trainable,
        })

    # Summary row
    records.append({
        "Layer_Index": "TOTAL",
        "Layer_Name": f"Total Layers: {len(model.layers)}",
        "Layer_Type": f"Trainable: {trainable_params:,}",
        "Output_Shape": f"Non-Trainable: {total_params - trainable_params:,}",
        "Parameter_Count": total_params,
        "Is_Trainable": True,
    })

    df = pd.DataFrame(records)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("           Custom CNN Architecture Parameter Summary")
    print("=" * 70)
    model.summary()
    print("-" * 70)
    print(f"Exported Layer Parameters Table: {out_path.resolve()}")
    print("=" * 70 + "\n")

