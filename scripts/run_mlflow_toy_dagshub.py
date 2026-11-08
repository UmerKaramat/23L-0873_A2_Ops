"""
VisionGuard - Task 3.2 DagsHub MLflow Toy Experiment
File: scripts/run_mlflow_toy_dagshub.py

Purpose:
Demonstrates explicit manual MLflow logging to a remote DagsHub tracking server:
1. Imports 'dotenv' and calls load_dotenv() at the very beginning to load credentials from .env.
2. Reads the MLFLOW_TRACKING_URI environment variable via os.getenv().
3. Creates and sets the experiment named 'Task_3_2_DagsHub_Toy'.
4. Loads the built-in Fashion-MNIST dataset.
5. Builds a tiny Keras Sequential network (Flatten -> Dense -> Output).
6. Uses manual MLflow logging (no autolog):
   - mlflow.log_param() for hyperparameters (learning_rate, epochs, batch_size).
   - Custom Keras callback logging 'accuracy' and 'loss' per epoch via mlflow.log_metric().
   - Plots training curves, saves to disk, and logs via mlflow.log_artifact().
   - Logs the trained model via mlflow.keras.log_model().
"""

import argparse
import os
import sys
from pathlib import Path

# 1. Load credentials and environment variables from .env at the very start
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Notice: 'python-dotenv' package is not installed.")
    print("If your credentials are in a .env file, install it with: pip install python-dotenv")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    import mlflow
    import mlflow.keras
except ImportError:
    print("Error: 'mlflow' is required. Install it using: pip install mlflow")
    sys.exit(1)


class MLflowEpochMetricCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback that explicitly logs training and validation metrics
    to the active remote MLflow run at the end of each epoch using mlflow.log_metric().
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        step = epoch + 1  # 1-indexed epoch step for clear dashboard charting

        if "loss" in logs:
            mlflow.log_metric("loss", float(logs["loss"]), step=step)
        if "accuracy" in logs:
            mlflow.log_metric("accuracy", float(logs["accuracy"]), step=step)
        if "val_loss" in logs:
            mlflow.log_metric("val_loss", float(logs["val_loss"]), step=step)
        if "val_accuracy" in logs:
            mlflow.log_metric("val_accuracy", float(logs["val_accuracy"]), step=step)

        print(f"  [Remote MLflow Callback] Logged epoch {step} metrics: "
              f"loss={logs.get('loss', 0):.4f}, acc={logs.get('accuracy', 0):.4f}")


def build_toy_model(learning_rate=0.01):
    """
    Builds and compiles a tiny Keras network for Fashion-MNIST classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten_input"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_hidden"),
        tf.keras.layers.Dense(10, activation="softmax", name="dense_output")
    ], name="Tiny_FashionMNIST_DagsHub_Model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_and_save_curves(history, learning_rate, output_path):
    """
    Plots training loss and accuracy curves and saves the figure for remote MLflow artifact logging.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1.plot(epochs_range, history.history["loss"], label="Train Loss", color="#2980b9", linewidth=2)
    if "val_loss" in history.history:
        ax1.plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#e74c3c", linewidth=2)
    ax1.set_title("Loss vs. Epochs", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Sparse Categorical Crossentropy", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs_range, history.history["accuracy"], label="Train Accuracy", color="#2980b9", linewidth=2)
    if "val_accuracy" in history.history:
        ax2.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", color="#27ae60", linewidth=2)
    ax2.set_title("Accuracy vs. Epochs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.suptitle(f"Task 3.2 DagsHub Toy Experiment (Fashion-MNIST | LR={learning_rate})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    return out_file


def run_experiment(learning_rate=0.01, epochs=5, batch_size=64, subsample=12000):
    print("=" * 80)
    print("      VisionGuard Task 3.2: DagsHub Remote MLflow Toy Experiment")
    print("=" * 80)

    # 2. Retrieve tracking URI from environment variable
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: Environment variable 'MLFLOW_TRACKING_URI' is not set.")
        print("Please define it in your .env file or environment:")
        print("  MLFLOW_TRACKING_URI=https://dagshub.com/<USERNAME>/VisionGuard.mlflow")
        print("  MLFLOW_TRACKING_USERNAME=<USERNAME>")
        print("  MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>")
        sys.exit(1)

    # Set remote tracking URI and target experiment
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Task_3_2_DagsHub_Toy"
    mlflow.set_experiment(experiment_name)

    username_status = "DETECTED" if os.getenv("MLFLOW_TRACKING_USERNAME") else "NOT DETECTED (ensure auth is configured)"
    print(f"MLflow Remote Tracking URI : {tracking_uri}")
    print(f"MLflow Experiment Name     : {experiment_name}")
    print(f"MLflow Tracking User       : {username_status}")
    print(f"Target Hyperparameters     : LR = {learning_rate}, Epochs = {epochs}, Batch Size = {batch_size}")
    print("-" * 80)

    # 3. Load Fashion-MNIST Dataset
    print("Loading built-in Fashion-MNIST dataset...")
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # Subsample for fast CPU verification
    if subsample and subsample < len(x_train):
        x_train = x_train[:subsample]
        y_train = y_train[:subsample]
        x_val = x_val[:subsample // 4]
        y_val = y_val[:subsample // 4]
        print(f"Using {len(x_train)} training and {len(x_val)} validation samples for fast CPU execution.")

    # 4. Build Tiny Keras Network
    model = build_toy_model(learning_rate=learning_rate)

    # 5. Start MLflow Run with EXPLICIT Manual Logging to DagsHub
    with mlflow.start_run(run_name=f"dagshub_toy_lr_{learning_rate}") as active_run:
        run_id = active_run.info.run_id
        print(f"\n[DagsHub MLflow Run Started] Run ID: {run_id}")

        # A. Log parameters explicitly
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("dataset", "Fashion-MNIST")
        mlflow.log_param("architecture", "Flatten -> Dense(64, ReLU) -> Dense(10, Softmax)")
        mlflow.log_param("tracking_target", "DagsHub")
        print("  - Logged parameters to DagsHub via mlflow.log_param()")

        # B. Train with custom callback logging metrics per epoch
        print("\nBeginning training (logging epoch metrics to DagsHub via custom callback)...")
        metric_callback = MLflowEpochMetricCallback()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[metric_callback],
            verbose=1
        )

        # C. Generate plot and log as artifact
        curve_image_path = Path("reports/figures/task3_2_dagshub_toy_curves.png")
        plot_and_save_curves(history, learning_rate, curve_image_path)
        print("  - Uploading training curves artifact to DagsHub via mlflow.log_artifact()...")
        mlflow.log_artifact(str(curve_image_path.resolve()), artifact_path="training_plots")

        # D. Log trained model using mlflow.keras.log_model
        print("  - Uploading serialized model artifact to DagsHub via mlflow.keras.log_model()...")
        mlflow.keras.log_model(model, artifact_path="toy_model")

        final_loss = float(history.history["loss"][-1])
        final_acc = float(history.history["accuracy"][-1])
        final_val_loss = float(history.history["val_loss"][-1])
        final_val_acc = float(history.history["val_accuracy"][-1])

    print("\n" + "=" * 80)
    print("                 TASK 3.2 DAGSHUB RUN COMPLETED")
    print("=" * 80)
    print(f"Run ID            : {run_id}")
    print(f"Final Train Loss  : {final_loss:.4f} | Train Acc: {final_acc:.4f}")
    print(f"Final Val Loss    : {final_val_loss:.4f} | Val Acc  : {final_val_acc:.4f}")
    print(f"Remote Server     : {tracking_uri}")
    print(f"Local Curve Plot  : {curve_image_path.resolve()}")
    print("-" * 80)
    print("View your run online in your DagsHub repository:")
    print("  1. Open your DagsHub repository dashboard in your browser.")
    print("  2. Click on the 'Experiments' tab.")
    print(f"  3. Select experiment '{experiment_name}' to see your metrics, curves, and model.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Task 3.2: Run DagsHub remote toy MLflow experiment with manual parameter, metric, and artifact logging."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for Adam optimizer (default: 0.01)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=12000,
        help="Number of Fashion-MNIST training samples to use for fast CPU execution (default: 12000)"
    )

    args = parser.parse_args()
    run_experiment(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subsample=args.subsample
    )


if __name__ == "__main__":
    main()

