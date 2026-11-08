"""
VisionGuard - Task 3.5 Six Experiment Comparison Script
File: scripts/run_task3_5_compare.py

Purpose:
Automates six distinct, meaningful training experiments logged to DagsHub MLflow under
the experiment 'Task_3_5_Comparison' as required by the assignment rubric:
- Run 1: Baseline Custom CNN (LR=1e-3, Augmentation=True, Size=224x224)
- Run 2: Baseline Custom CNN (LR=1e-4) -> Controlled Change: Learning Rate
- Run 3: Baseline Custom CNN (Augmentation=False) -> Controlled Change: Augmentation
- Run 4: Baseline Custom CNN (Size=128x128) -> Controlled Change: Image Resolution
- Run 5: MobileNetV2 Transfer (Backbone=Frozen, LR=1e-3) -> Controlled Change: Pre-trained Backbone
- Run 6: MobileNetV2 Transfer (Backbone=Unfrozen Top 30, LR=1e-4) -> Controlled Change: Fine-Tuning Unfreezing

Each run trains for 1 epoch on 300 training and 100 validation samples for fast CPU execution (~30-40s total).
Logs parameters, metrics, run names, and exports a comparison table to reports/task3_5_six_experiments_summary.csv.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 1. Ingest credentials from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Notice: 'python-dotenv' package is not installed. Install with: pip install python-dotenv")

try:
    import mlflow
except ImportError:
    print("Error: 'mlflow' is required. Install it using: pip install mlflow")
    sys.exit(1)

from src.data.pipeline import (
    create_tf_dataset,
    get_dataset_splits,
    set_seed,
)
from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_model import (
    build_transfer_model,
    unfreeze_for_fine_tuning,
)


def run_experiment_comparison(
    data_dir="data/raw",
    train_subsample=300,
    val_subsample=100,
    batch_size=32,
    seed=42
):
    print("=" * 84)
    print("        VisionGuard Task 3.5: Six Experiment Automated Comparison")
    print("=" * 84)

    # 1. Verify DagsHub Remote Connection
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: 'MLFLOW_TRACKING_URI' is not set in environment or .env file.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Task_3_5_Comparison"
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Remote URI : {tracking_uri}")
    print(f"MLflow Experiment : {experiment_name}")
    print(f"Sample Budget     : {train_subsample} Train / {val_subsample} Val (1 epoch each for CPU efficiency)")
    print("-" * 84)

    # 2. Extract Data Split Paths
    set_seed(seed)
    splits = get_dataset_splits(data_dir=data_dir, test_size=0.15, val_size=0.15, seed=seed)

    train_paths = splits["train_paths"][:train_subsample]
    train_labels = splits["train_labels"][:train_subsample]
    val_paths = splits["val_paths"][:val_subsample]
    val_labels = splits["val_labels"][:val_subsample]

    # Pre-generate datasets for 224x224 (Augmented and Non-Augmented) and 128x128
    print("\nPreparing tf.data input streams...")
    ds_224_aug = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, img_size=(224, 224), is_training=True, augment=True)
    ds_224_noaug = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, img_size=(224, 224), is_training=True, augment=False)
    val_224 = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, img_size=(224, 224), is_training=False, augment=False)

    ds_128_aug = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, img_size=(128, 128), is_training=True, augment=True)
    val_128 = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, img_size=(128, 128), is_training=False, augment=False)

    # 3. Define the Six Rubric Configurations
    experiments = [
        {
            "run_name": "Run_1_Baseline_CNN",
            "model_type": "Custom_CNN",
            "learning_rate": 1e-3,
            "augmentation": True,
            "image_size": "224x224",
            "backbone": "None (Scratch)",
            "frozen_state": "N/A",
            "change_description": "Baseline: Custom CNN, LR=1e-3, Aug=True, 224x224",
            "train_ds": ds_224_aug,
            "val_ds": val_224,
            "input_shape": (224, 224, 3)
        },
        {
            "run_name": "Run_2_LR_1e4",
            "model_type": "Custom_CNN",
            "learning_rate": 1e-4,
            "augmentation": True,
            "image_size": "224x224",
            "backbone": "None (Scratch)",
            "frozen_state": "N/A",
            "change_description": "Controlled Change: Lower Learning Rate (1e-4 vs 1e-3)",
            "train_ds": ds_224_aug,
            "val_ds": val_224,
            "input_shape": (224, 224, 3)
        },
        {
            "run_name": "Run_3_No_Augmentation",
            "model_type": "Custom_CNN",
            "learning_rate": 1e-3,
            "augmentation": False,
            "image_size": "224x224",
            "backbone": "None (Scratch)",
            "frozen_state": "N/A",
            "change_description": "Controlled Change: Data Augmentation Disabled",
            "train_ds": ds_224_noaug,
            "val_ds": val_224,
            "input_shape": (224, 224, 3)
        },
        {
            "run_name": "Run_4_Size_128",
            "model_type": "Custom_CNN",
            "learning_rate": 1e-3,
            "augmentation": True,
            "image_size": "128x128",
            "backbone": "None (Scratch)",
            "frozen_state": "N/A",
            "change_description": "Controlled Change: Reduced Image Resolution (128x128 vs 224x224)",
            "train_ds": ds_128_aug,
            "val_ds": val_128,
            "input_shape": (128, 128, 3)
        },
        {
            "run_name": "Run_5_MobileNet_Frozen",
            "model_type": "MobileNetV2_Transfer",
            "learning_rate": 1e-3,
            "augmentation": True,
            "image_size": "224x224",
            "backbone": "MobileNetV2",
            "frozen_state": "All Frozen",
            "change_description": "Controlled Change: Transfer Learning Backbone (Feature Extraction)",
            "train_ds": ds_224_aug,
            "val_ds": val_224,
            "input_shape": (224, 224, 3)
        },
        {
            "run_name": "Run_6_MobileNet_Unfrozen",
            "model_type": "MobileNetV2_Transfer",
            "learning_rate": 1e-4,
            "augmentation": True,
            "image_size": "224x224",
            "backbone": "MobileNetV2",
            "frozen_state": "Top 30 Unfrozen",
            "change_description": "Controlled Change: Fine-Tuning with Top 30 Backbone Layers Unfrozen",
            "train_ds": ds_224_aug,
            "val_ds": val_224,
            "input_shape": (224, 224, 3)
        }
    ]

    summary_records = []

    # 4. Execute the 6 Runs Sequentially
    for idx, cfg in enumerate(experiments, start=1):
        print("\n" + "=" * 84)
        print(f">>> EXECUTING RUN {idx}/6: {cfg['run_name']}")
        print(f"    Hypothesis/Change: {cfg['change_description']}")
        print("=" * 84)

        # Build fresh model according to configuration
        set_seed(seed + idx)

        if cfg["model_type"] == "Custom_CNN":
            model = build_custom_cnn(input_shape=cfg["input_shape"], learning_rate=cfg["learning_rate"])
        elif cfg["run_name"] == "Run_5_MobileNet_Frozen":
            model, _ = build_transfer_model(input_shape=cfg["input_shape"], learning_rate=cfg["learning_rate"])
        elif cfg["run_name"] == "Run_6_MobileNet_Unfrozen":
            base_m, base_backbone = build_transfer_model(input_shape=cfg["input_shape"], learning_rate=1e-3)
            model = unfreeze_for_fine_tuning(
                model=base_m,
                base_model=base_backbone,
                unfreeze_layers=30,
                fine_tune_lr=cfg["learning_rate"]
            )

        # Start MLflow Run
        with mlflow.start_run(run_name=cfg["run_name"]) as run:
            run_id = run.info.run_id
            print(f"  MLflow Run ID: {run_id}")

            # Log Hyperparameters
            mlflow.log_param("run_index", idx)
            mlflow.log_param("model_type", cfg["model_type"])
            mlflow.log_param("learning_rate", cfg["learning_rate"])
            mlflow.log_param("augmentation", cfg["augmentation"])
            mlflow.log_param("image_size", cfg["image_size"])
            mlflow.log_param("backbone", cfg["backbone"])
            mlflow.log_param("frozen_state", cfg["frozen_state"])
            mlflow.log_param("change_description", cfg["change_description"])
            mlflow.log_param("epochs", 1)
            mlflow.log_param("train_samples", train_subsample)
            mlflow.log_param("val_samples", val_subsample)
            mlflow.set_tag("task", "Task_3_5_Experiment_Comparison")

            # Train for 1 epoch
            history = model.fit(
                cfg["train_ds"],
                validation_data=cfg["val_ds"],
                epochs=1,
                verbose=1
            )

            # Extract metrics
            train_loss = float(history.history["loss"][-1])
            train_acc = float(history.history["accuracy"][-1])
            val_loss = float(history.history["val_loss"][-1])
            val_acc = float(history.history["val_accuracy"][-1])

            # Log Final Metrics
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_accuracy", val_acc)

            summary_records.append({
                "Run #": idx,
                "Run Name": cfg["run_name"],
                "Model Type": cfg["model_type"],
                "Controlled Change": cfg["change_description"],
                "LR": f"{cfg['learning_rate']:.0e}",
                "Augmentation": str(cfg["augmentation"]),
                "Image Size": cfg["image_size"],
                "Backbone": cfg["backbone"],
                "Val Loss": f"{val_loss:.4f}",
                "Val Accuracy": f"{val_acc:.4f}",
                "MLflow Run ID": run_id,
            })

    # 5. Export and Display Summary Table
    df_summary = pd.DataFrame(summary_records)
    out_csv = Path("reports/task3_5_six_experiments_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_csv, index=False)

    print("\n" + "=" * 84)
    print("        VisionGuard Task 3.5: Six Experiments Comparison Summary")
    print("=" * 84)
    # Print formatted columns
    display_cols = ["Run #", "Run Name", "LR", "Augmentation", "Image Size", "Backbone", "Val Loss", "Val Accuracy", "MLflow Run ID"]
    print(df_summary[display_cols].to_string(index=False))
    print("-" * 84)
    print(f"Summary table exported to: {out_csv.resolve()}")
    print("-" * 84)
    print("To view and compare all 6 runs on DagsHub:")
    print("  1. Open your DagsHub repository in your browser.")
    print("  2. Go to the 'Experiments' tab and select 'Task_3_5_Comparison'.")
    print("  3. Check all 6 runs and click 'Compare' to view the parallel coordinates chart,")
    print("     scatter plots, and metric comparisons across the runs.")
    print("=" * 84 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Task 3.5: Execute and log six distinct comparison experiments to DagsHub MLflow."
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset.")
    parser.add_argument("--train-subsample", type=int, default=300, help="Train samples per run (default: 300).")
    parser.add_argument("--val-subsample", type=int, default=100, help="Val samples per run (default: 100).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()
    run_experiment_comparison(
        data_dir=args.data_dir,
        train_subsample=args.train_subsample,
        val_subsample=args.val_subsample,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
