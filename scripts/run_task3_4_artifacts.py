"""
VisionGuard - Task 3.4 Artifact Logging with MLflow & DagsHub
File: scripts/run_task3_4_artifacts.py

Purpose:
Demonstrates multi-type artifact generation and logging to DagsHub MLflow:
1. Connects to DagsHub MLflow URI using credentials from .env.
2. Sets experiment name to 'Task_3_4_Artifacts'.
3. Trains a Custom CNN on a 400-image subset for 1 epoch for rapid execution.
4. Generates local artifacts prior to upload:
   - Confusion Matrix plot (reports/figures/task3_4_cm.png)
   - ROC curve plot (reports/figures/task3_4_roc.png)
   - Split file path lists (reports/task3_4_split_lists.txt)
   - Environment package manifest via pip freeze (reports/task3_4_packages.txt)
5. Attaches all 4 artifacts using mlflow.log_artifact() and logs the model via mlflow.keras.log_model().
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 1. Ingest credentials from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Notice: 'python-dotenv' package is not installed. Install with: pip install python-dotenv")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
import tensorflow as tf

try:
    import mlflow
    import mlflow.keras
except ImportError:
    print("Error: 'mlflow' is required. Install it using: pip install mlflow")
    sys.exit(1)

from src.data.pipeline import (
    CLASS_NAMES,
    create_tf_dataset,
    get_dataset_splits,
    set_seed,
)
from src.models.custom_cnn import build_custom_cnn


def generate_confusion_matrix_plot(y_true, y_pred, output_path, class_names=CLASS_NAMES):
    """Generates and saves a publication-quality Confusion Matrix heatmap."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=11)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_title("Task 3.4: Test Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("True Ground Truth Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    return out_file


def generate_roc_curve_plot(y_true, y_pred_probs, output_path):
    """Generates and saves an ROC Curve plot with AUC annotation."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2.2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Task 3.4: Test ROC Curve", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    return out_file


def save_split_lists_file(splits, output_path):
    """Exports train, val, and test relative file paths into a structured text file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("VisionGuard Task 3.4: Dataset Split Manifest\n")
        f.write(f"Total Train Samples      : {len(splits['train_paths'])}\n")
        f.write(f"Total Validation Samples : {len(splits['val_paths'])}\n")
        f.write(f"Total Test Samples       : {len(splits['test_paths'])}\n")
        f.write("=" * 80 + "\n\n")

        for section_name, path_key, label_key in [
            ("TRAINING SPLIT", "train_paths", "train_labels"),
            ("VALIDATION SPLIT", "val_paths", "val_labels"),
            ("TEST SPLIT", "test_paths", "test_labels")
        ]:
            f.write(f"[{section_name}] ({len(splits[path_key])} files)\n")
            f.write("-" * 80 + "\n")
            for p, l in zip(splits[path_key], splits[label_key]):
                class_str = CLASS_NAMES[l] if l < len(CLASS_NAMES) else str(l)
                f.write(f"{class_str}\t{p}\n")
            f.write("\n")

    return out_file


def capture_environment_packages(output_path):
    """Runs 'pip freeze' via subprocess and exports the exact environment package versions."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        package_manifest = result.stdout
    except Exception as err:
        package_manifest = f"# Error running pip freeze: {err}\n"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# VisionGuard Task 3.4: Captured Python Environment Package Manifest\n")
        f.write(f"# Python Executable: {sys.executable}\n")
        f.write("# " + "=" * 76 + "\n")
        f.write(package_manifest)

    return out_file


def run_artifact_pipeline(
    data_dir="data/raw",
    subsample=400,
    epochs=1,
    batch_size=32,
    seed=42
):
    print("=" * 80)
    print("         VisionGuard Task 3.4: Multi-Artifact MLflow Pipeline")
    print("=" * 80)

    # 1. Verify DagsHub Remote Connection
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: 'MLFLOW_TRACKING_URI' is not set in environment or .env file.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Task_3_4_Artifacts"
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Remote URI : {tracking_uri}")
    print(f"MLflow Experiment : {experiment_name}")
    print(f"Dataset Subset    : {subsample} samples (1 epoch training for rapid artifact demonstration)")
    print("-" * 80)

    # 2. Data Preparation
    set_seed(seed)
    splits = get_dataset_splits(data_dir=data_dir, test_size=0.15, val_size=0.15, seed=seed)

    train_paths = splits["train_paths"][:subsample]
    train_labels = splits["train_labels"][:subsample]
    val_paths = splits["val_paths"][:subsample // 4]
    val_labels = splits["val_labels"][:subsample // 4]
    test_paths = splits["test_paths"][:subsample // 4]
    test_labels = splits["test_labels"][:subsample // 4]

    subset_splits = {
        "train_paths": train_paths, "train_labels": train_labels,
        "val_paths": val_paths, "val_labels": val_labels,
        "test_paths": test_paths, "test_labels": test_labels,
    }

    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True, augment=True)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False, augment=False)
    test_ds = create_tf_dataset(test_paths, test_labels, batch_size=batch_size, is_training=False, augment=False)

    # 3. Model Training
    print("\nTraining Custom CNN for 1 epoch...")
    model = build_custom_cnn(input_shape=(224, 224, 3), learning_rate=1e-3)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

    # 4. Generate Predictions & Local Artifacts
    print("\nGenerating test predictions & compiling local artifacts...")
    y_pred_probs = model.predict(test_ds, verbose=0).ravel()
    y_true = np.array(test_labels, dtype=int)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Artifact 1: Confusion Matrix
    cm_path = generate_confusion_matrix_plot(y_true, y_pred, "reports/figures/task3_4_cm.png")
    print(f"  [1/4] Generated Confusion Matrix -> {cm_path}")

    # Artifact 2: ROC Curve
    roc_path = generate_roc_curve_plot(y_true, y_pred_probs, "reports/figures/task3_4_roc.png")
    print(f"  [2/4] Generated ROC Curve        -> {roc_path}")

    # Artifact 3: Dataset Split Manifest
    splits_path = save_split_lists_file(subset_splits, "reports/task3_4_split_lists.txt")
    print(f"  [3/4] Generated Split Manifest   -> {splits_path}")

    # Artifact 4: Environment Package Manifest via subprocess
    packages_path = capture_environment_packages("reports/task3_4_packages.txt")
    print(f"  [4/4] Generated Package Manifest -> {packages_path}")

    # 5. MLflow Run & Remote Artifact Upload
    print("\nUploading artifacts and model to DagsHub MLflow...")
    with mlflow.start_run(run_name="Task_3_4_Artifact_Demo") as run:
        run_id = run.info.run_id

        # Log basic run metadata
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("subsample_samples", subsample)
        mlflow.log_param("model_architecture", "Custom_CNN")
        mlflow.set_tag("artifact_pipeline", "Task_3_4_Demonstration")

        # Log evaluation metrics
        test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_auc", float(test_auc))

        # Upload the 4 local artifacts
        print("  - Uploading Confusion Matrix plot to 'plots' artifact path...")
        mlflow.log_artifact(str(cm_path.resolve()), artifact_path="plots")

        print("  - Uploading ROC Curve plot to 'plots' artifact path...")
        mlflow.log_artifact(str(roc_path.resolve()), artifact_path="plots")

        print("  - Uploading Dataset Split Manifest to 'manifests' artifact path...")
        mlflow.log_artifact(str(splits_path.resolve()), artifact_path="manifests")

        print("  - Uploading Captured Environment Packages to 'environment' artifact path...")
        mlflow.log_artifact(str(packages_path.resolve()), artifact_path="environment")

        # Upload trained Keras model
        print("  - Uploading trained Keras model artifact via mlflow.keras.log_model()...")
        mlflow.keras.log_model(model, artifact_path="model")

    print("\n" + "=" * 80)
    print("                    TASK 3.4 ARTIFACT LOGGING COMPLETE")
    print("=" * 80)
    print(f"Run ID                : {run_id}")
    print(f"DagsHub Experiment    : {experiment_name}")
    print("\nUploaded Artifacts:")
    print(f"  1. Confusion Matrix : {cm_path.resolve()}")
    print(f"  2. ROC Curve        : {roc_path.resolve()}")
    print(f"  3. Split Lists      : {splits_path.resolve()}")
    print(f"  4. Pip Freeze       : {packages_path.resolve()}")
    print(f"  5. Keras Model      : logged under 'model/' directory")
    print("-" * 80)
    print("To view logged artifacts in DagsHub:")
    print("  1. Open your DagsHub repository in your browser.")
    print("  2. Go to the 'Experiments' tab and click on experiment 'Task_3_4_Artifacts'.")
    print("  3. Click on run 'Task_3_4_Artifact_Demo', then click on the 'Artifacts' tab.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Task 3.4: Generate and log multi-type artifacts to DagsHub MLflow."
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset.")
    parser.add_argument("--subsample", type=int, default=400, help="Number of samples to use (default: 400).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()
    run_artifact_pipeline(
        data_dir=args.data_dir,
        subsample=args.subsample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

