"""
VisionGuard - Task 3.3 Training Instrumentation with MLflow & DagsHub
File: scripts/run_task3_3_instrumentation.py

Purpose:
Instruments VisionGuard model training with MLflow tracking connected to DagsHub:
1. Ingests credentials from .env via dotenv and connects to MLFLOW_TRACKING_URI.
2. Sets the experiment name to 'Task_3_3_VisionGuard_Tracking'.
3. Retrieves git_commit and dvc_version metadata via subprocess and fixes random seed to 42.
4. Uses existing src/ modules (src.data.pipeline and src.models.custom_cnn).
5. Run 1: 'Manual_Logging' - Explicitly logs tags, parameters, epoch metrics (custom callback),
   test evaluation metrics, and the model artifact.
6. Run 2: 'TF_Autologging' - Uses mlflow.tensorflow.autolog() to automatically capture parameters,
   training dynamics, and model artifacts, then disables autologging.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 1. Load credentials from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Notice: 'python-dotenv' package is not installed. Install with: pip install python-dotenv")

import tensorflow as tf

try:
    import mlflow
    import mlflow.keras
    import mlflow.tensorflow
except ImportError:
    print("Error: 'mlflow' is required. Install it using: pip install mlflow")
    sys.exit(1)

from src.data.pipeline import create_tf_dataset, get_dataset_splits, set_seed
from src.models.custom_cnn import build_custom_cnn


def get_git_commit_hash():
    """Fetches the current HEAD git commit hash via subprocess."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "git_commit_unavailable"


def get_dvc_version():
    """
    Fetches the latest commit hash for data/raw.dvc or DVC CLI version via subprocess.
    """
    # 1. Check git log for the DVC pointer file
    try:
        result = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%H", "--", "data/raw.dvc"],
            capture_output=True,
            text=True,
            check=True
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass

    # 2. Check git log for .dvc/config
    try:
        result = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%H", "--", ".dvc/config"],
            capture_output=True,
            text=True,
            check=True
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass

    # 3. Fallback to DVC CLI version
    try:
        result = subprocess.run(
            ["dvc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return f"dvc_{result.stdout.strip()}"
    except Exception:
        return "dvc_version_unavailable"


class ManualEpochMetricCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback for Run 1: explicitly logs loss, accuracy,
    val_loss, and val_accuracy per epoch using mlflow.log_metric().
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        step = epoch + 1
        for metric_name, metric_val in logs.items():
            if metric_val is not None:
                mlflow.log_metric(metric_name, float(metric_val), step=step)
        print(f"  [Manual Metric Log] Epoch {step}: "
              f"loss={logs.get('loss', 0):.4f}, acc={logs.get('accuracy', 0):.4f}, "
              f"val_loss={logs.get('val_loss', 0):.4f}, val_acc={logs.get('val_accuracy', 0):.4f}")


def run_instrumentation(
    data_dir="data/raw",
    epochs=2,
    batch_size=32,
    learning_rate=1e-3,
    subsample=600,
    seed=42
):
    print("=" * 82)
    print("      VisionGuard Task 3.3: Training Instrumentation (Manual vs Autolog)")
    print("=" * 82)

    # 1. Connect to DagsHub MLflow Tracking Server
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: Environment variable 'MLFLOW_TRACKING_URI' is not set.")
        print("Please define it in your .env file:")
        print("  MLFLOW_TRACKING_URI=https://dagshub.com/<USERNAME>/VisionGuard.mlflow")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Task_3_3_VisionGuard_Tracking"
    mlflow.set_experiment(experiment_name)

    # 2. Extract Git Commit & DVC Version Metadata
    git_commit = get_git_commit_hash()
    dvc_version = get_dvc_version()

    print(f"MLflow Remote URI     : {tracking_uri}")
    print(f"MLflow Experiment     : {experiment_name}")
    print(f"Git Commit Hash       : {git_commit}")
    print(f"DVC Version Tracking  : {dvc_version}")
    print(f"Fixed Random Seed     : {seed}")
    print(f"Execution Parameters  : Epochs = {epochs}, Batch Size = {batch_size}, LR = {learning_rate}")
    print("-" * 82)

    # 3. Prepare Dataset Splits
    set_seed(seed)
    print("\nLoading dataset splits from existing src/data/pipeline.py...")
    splits = get_dataset_splits(data_dir=data_dir, test_size=0.15, val_size=0.15, seed=seed)

    train_paths = splits["train_paths"]
    train_labels = splits["train_labels"]
    val_paths = splits["val_paths"]
    val_labels = splits["val_labels"]
    test_paths = splits["test_paths"]
    test_labels = splits["test_labels"]

    # Subsample for CPU efficiency
    if subsample and subsample < len(train_paths):
        print(f"Applying CPU subsampling: {subsample} train, {subsample // 4} val, {subsample // 4} test samples.")
        train_paths = train_paths[:subsample]
        train_labels = train_labels[:subsample]
        val_paths = val_paths[:subsample // 4]
        val_labels = val_labels[:subsample // 4]
        test_paths = test_paths[:subsample // 4]
        test_labels = test_labels[:subsample // 4]

    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True, augment=True)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False, augment=False)
    test_ds = create_tf_dataset(test_paths, test_labels, batch_size=batch_size, is_training=False, augment=False)

    # ==========================================================================
    # RUN 1: MANUAL LOGGING
    # ==========================================================================
    print("\n" + "=" * 82)
    print(">>> RUN 1: Explicit Manual Logging ('Manual_Logging')")
    print("=" * 82)

    with mlflow.start_run(run_name="Manual_Logging") as run1:
        run1_id = run1.info.run_id
        print(f"Run 1 Started | Run ID: {run1_id}")

        # Set Tags
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_version", dvc_version)
        mlflow.set_tag("seed", str(seed))
        mlflow.set_tag("logging_paradigm", "Manual")
        mlflow.set_tag("model_family", "Custom_CNN")
        print("  [Run 1] Set tags: git_commit, dvc_version, seed, logging_paradigm")

        # Log Hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("architecture", "Custom_CNN (4 blocks + GAP)")
        mlflow.log_param("loss_function", "binary_crossentropy")
        print("  [Run 1] Logged hyperparameters via mlflow.log_param()")

        # Build fresh CNN
        set_seed(seed)
        model1 = build_custom_cnn(input_shape=(224, 224, 3), learning_rate=learning_rate)

        # Train with custom callback
        print("  [Run 1] Training CNN with ManualEpochMetricCallback...")
        model1.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[ManualEpochMetricCallback()],
            verbose=1
        )

        # Evaluate on Test Set & Log Test Metrics
        print("  [Run 1] Evaluating on unseen test set...")
        test_eval = model1.evaluate(test_ds, return_dict=True, verbose=0)
        for metric_name, metric_val in test_eval.items():
            mlflow.log_metric(f"test_{metric_name}", float(metric_val))
            print(f"    -> Logged test_{metric_name}: {metric_val:.4f}")

        # Log Model Artifact
        print("  [Run 1] Serializing and logging model via mlflow.keras.log_model()...")
        mlflow.keras.log_model(model1, artifact_path="model")
        print(f"  [Run 1 Complete] Run ID: {run1_id}")

    # ==========================================================================
    # RUN 2: TENSORFLOW AUTOLOGGING
    # ==========================================================================
    print("\n" + "=" * 82)
    print(">>> RUN 2: TensorFlow Autologging ('TF_Autologging')")
    print("=" * 82)

    # Enable TensorFlow autologging
    mlflow.tensorflow.autolog(
        log_models=True,
        log_datasets=False,
        registered_model_name=None
    )
    print("  [Run 2] mlflow.tensorflow.autolog() ENABLED.")

    try:
        with mlflow.start_run(run_name="TF_Autologging") as run2:
            run2_id = run2.info.run_id
            print(f"Run 2 Started | Run ID: {run2_id}")

            # Set the exact same tags
            mlflow.set_tag("git_commit", git_commit)
            mlflow.set_tag("dvc_version", dvc_version)
            mlflow.set_tag("seed", str(seed))
            mlflow.set_tag("logging_paradigm", "Autolog")
            mlflow.set_tag("model_family", "Custom_CNN")
            print("  [Run 2] Set tags: git_commit, dvc_version, seed, logging_paradigm")

            # Build a fresh CNN
            set_seed(seed + 100)
            model2 = build_custom_cnn(input_shape=(224, 224, 3), learning_rate=learning_rate)

            # Train (autolog automatically captures params, optimizer, loss/acc curves, and model)
            print("  [Run 2] Training fresh CNN (Autolog is capturing all dynamics)...")
            model2.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1
            )

            # Evaluate on Test Set
            print("  [Run 2] Evaluating fresh CNN on test set...")
            model2.evaluate(test_ds, verbose=1)
            print(f"  [Run 2 Complete] Run ID: {run2_id}")

    finally:
        # Crucial requirement: disable autologging at the end
        mlflow.tensorflow.autolog(disable=True)
        print("  [Run 2] mlflow.tensorflow.autolog(disable=True) DISABLED.")

    print("\n" + "=" * 82)
    print("                 TASK 3.3 INSTRUMENTATION COMPLETE")
    print("=" * 82)
    print(f"DagsHub MLflow Server : {tracking_uri}")
    print(f"Experiment Name       : {experiment_name}")
    print(f"Run 1 (Manual) ID     : {run1_id}")
    print(f"Run 2 (Autolog) ID    : {run2_id}")
    print("-" * 82)
    print("View the comparison in your DagsHub repository:")
    print("  1. Navigate to your DagsHub repository in your browser.")
    print("  2. Open the 'Experiments' tab and select 'Task_3_3_VisionGuard_Tracking'.")
    print("  3. Check both 'Manual_Logging' and 'TF_Autologging' to compare metrics, tags, and parameters.")
    print("=" * 82 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Task 3.3: Compare Manual MLflow Logging vs. TensorFlow Autologging on DagsHub."
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per run (default: 2).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 0.001).")
    parser.add_argument("--subsample", type=int, default=600,
                        help="Subsample count for fast CPU verification (default: 600).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()
    run_instrumentation(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

