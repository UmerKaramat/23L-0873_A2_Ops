"""
VisionGuard - Master Task 2 Execution Script
File: scripts/run_task2.py

Purpose:
Orchestrates the complete Task 2 model development, evaluation, and visual evidence generation:
1. Loads dataset splits & builds tf.data pipelines (Task 2.1).
2. Generates augmented batch grid figure (reports/figures/task2_1_augmented_batch.png).
3. Builds, inspects, exports parameter CSV, and trains Custom CNN (Task 2.2).
4. Generates Custom CNN loss/accuracy curves with best-epoch marker (reports/figures/task2_2_cnn_curves.png).
5. Builds, trains (Feature Extraction + Fine-Tuning) Transfer Learning MobileNetV2 (Task 2.3).
6. Audits and prints trainable layers before and after unfreezing.
7. Generates combined transfer learning curves (reports/figures/task2_3_transfer_curves.png).
8. Evaluates both models on unseen test split; prints comparison table & exports CSV (Task 2.5).
9. Generates side-by-side Confusion Matrices & ROC curves (reports/figures/task2_5_*.png).
10. Generates qualitative Error Analysis grid of 10 misclassified images (reports/figures/task2_6_error_analysis.png).
"""

import argparse
import os
import sys
from pathlib import Path
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import (
    CLASS_NAMES,
    create_tf_dataset,
    get_dataset_splits,
    save_augmented_batch_grid,
    set_seed,
)
from src.evaluation.metrics import (
    evaluate_model_on_dataset,
    generate_error_analysis_grid,
    plot_comparative_roc_curves,
    plot_side_by_side_confusion_matrices,
    print_and_export_comparison_table,
)
from src.models.custom_cnn import (
    build_custom_cnn,
    export_model_summary_to_csv,
)
from src.models.transfer_model import (
    build_transfer_model,
    plot_combined_transfer_curves,
    print_trainable_layer_status,
    unfreeze_for_fine_tuning,
)
from src.utils.training_utils import (
    get_standard_callbacks,
    plot_training_curves_with_best_marker,
    set_reproducible_seeds,
)


def run_task2(
    data_dir="data/raw",
    cnn_epochs=12,
    transfer_init_epochs=8,
    transfer_fine_epochs=6,
    batch_size=32,
    subsample=None,
    seed=42,
):
    print("=" * 80)
    print("     VISIONGUARD: TASK 2 MODEL DEVELOPMENT & EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Project Root     : {PROJECT_ROOT}")
    print(f"Data Directory   : {Path(data_dir).resolve()}")
    print(f"TensorFlow Ver   : {tf.__version__} (CPU Execution)")
    print(f"Random Seed      : {seed}")
    print("-" * 80)

    # Enforce global reproducibility
    set_reproducible_seeds(seed)
    set_seed(seed)

    # --------------------------------------------------------------------------
    # 1. Dataset Discovery & Split (Task 2.1)
    # --------------------------------------------------------------------------
    print("\n>>> STEP 1: Creating Stratified Train / Val / Test Splits (70 / 15 / 15)...")
    splits = get_dataset_splits(data_dir=data_dir, test_size=0.15, val_size=0.15, seed=seed)

    train_paths = splits["train_paths"]
    train_labels = splits["train_labels"]
    val_paths = splits["val_paths"]
    val_labels = splits["val_labels"]
    test_paths = splits["test_paths"]
    test_labels = splits["test_labels"]

    # Optional subsampling for rapid CPU verification
    if subsample and subsample < len(train_paths):
        print(f"\n[NOTE] Subsampling applied: Using {subsample} train samples.")
        train_paths = train_paths[:subsample]
        train_labels = train_labels[:subsample]
        val_paths = val_paths[:subsample // 4]
        val_labels = val_labels[:subsample // 4]
        test_paths = test_paths[:subsample // 4]
        test_labels = test_labels[:subsample // 4]

    print("Building tf.data input pipelines...")
    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True, augment=True)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False, augment=False)
    test_ds = create_tf_dataset(test_paths, test_labels, batch_size=batch_size, is_training=False, augment=False)

    # Evidence Gen 1: Save augmented batch visualization
    aug_batch_path = "reports/figures/task2_1_augmented_batch.png"
    print(f"\n>>> Generating Evidence 2.1: Augmented Batch Grid -> {aug_batch_path}")
    save_augmented_batch_grid(train_ds, output_path=aug_batch_path)

    # --------------------------------------------------------------------------
    # 2. Baseline Custom CNN (Task 2.2)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> STEP 2: Baseline Custom CNN Training")
    print("=" * 80)
    cnn_model = build_custom_cnn(input_shape=(224, 224, 3), learning_rate=1e-3)

    # Evidence Gen 2: Print summary to console & export layer parameters CSV
    cnn_csv_path = "reports/task2_2_cnn_params.csv"
    export_model_summary_to_csv(cnn_model, output_csv=cnn_csv_path)

    cnn_ckpt_path = "models/custom_cnn_best.keras"
    cnn_callbacks = get_standard_callbacks(checkpoint_path=cnn_ckpt_path, patience_early_stopping=4, patience_lr=2)

    print(f"Training Custom CNN for up to {cnn_epochs} epochs with EarlyStopping & ReduceLROnPlateau...")
    cnn_history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cnn_epochs,
        callbacks=cnn_callbacks,
        verbose=1
    )

    # Evidence Gen 2 (Curves): Plot loss and accuracy with best-checkpoint marker
    cnn_curves_path = "reports/figures/task2_2_cnn_curves.png"
    plot_training_curves_with_best_marker(cnn_history, model_name="Custom CNN", output_path=cnn_curves_path)

    # Reload best saved weights
    if Path(cnn_ckpt_path).exists():
        print(f"Loading best checkpoint weights from: {cnn_ckpt_path}")
        cnn_model = tf.keras.models.load_model(cnn_ckpt_path)

    # --------------------------------------------------------------------------
    # 3. Transfer Learning: MobileNetV2 (Task 2.3)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> STEP 3: Transfer Learning with MobileNetV2 (Feature Extraction + Fine-Tuning)")
    print("=" * 80)

    transfer_model, base_model = build_transfer_model(input_shape=(224, 224, 3), learning_rate=1e-3)

    # Evidence Gen 3: Audit trainable layers before unfreezing
    print_trainable_layer_status(transfer_model, stage_title="Phase 1: Feature Extraction (Backbone Frozen)")

    transfer_ckpt_path = "models/transfer_mobilenet_best.keras"
    init_callbacks = get_standard_callbacks(checkpoint_path=transfer_ckpt_path, patience_early_stopping=3, patience_lr=2)

    print(f"\n--- Phase 1: Training Classification Head ({transfer_init_epochs} epochs, LR=1e-3) ---")
    transfer_init_history = transfer_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=transfer_init_epochs,
        callbacks=init_callbacks,
        verbose=1
    )

    # Fine-Tuning: Unfreeze last 30 layers
    print("\n--- Phase 2: Unfreezing Top 30 Layers for Fine-Tuning (LR=1e-5) ---")
    transfer_model = unfreeze_for_fine_tuning(
        model=transfer_model,
        base_model=base_model,
        unfreeze_layers=30,
        fine_tune_lr=1e-5
    )

    # Evidence Gen 3: Audit trainable layers after unfreezing
    print_trainable_layer_status(transfer_model, stage_title="Phase 2: Fine-Tuning (Top 30 Layers Unfrozen)")

    fine_callbacks = get_standard_callbacks(checkpoint_path=transfer_ckpt_path, patience_early_stopping=3, patience_lr=2)

    transfer_fine_history = transfer_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=transfer_fine_epochs,
        callbacks=fine_callbacks,
        verbose=1
    )

    # Evidence Gen 3 (Curves): Combined transfer learning curves
    transfer_curves_path = "reports/figures/task2_3_transfer_curves.png"
    plot_combined_transfer_curves(
        transfer_init_history,
        transfer_fine_history,
        output_path=transfer_curves_path
    )

    # Reload best saved transfer model
    if Path(transfer_ckpt_path).exists():
        print(f"Loading best transfer checkpoint from: {transfer_ckpt_path}")
        transfer_model = tf.keras.models.load_model(transfer_ckpt_path)

    # --------------------------------------------------------------------------
    # 4. Comprehensive Evaluation on Unseen Test Split (Task 2.5)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> STEP 4: Evaluating Models on Unseen Test Set")
    print("=" * 80)

    cnn_eval = evaluate_model_on_dataset(cnn_model, test_ds, model_name="Baseline Custom CNN")
    transfer_eval = evaluate_model_on_dataset(transfer_model, test_ds, model_name="MobileNetV2 Transfer Learning")

    all_evals = [cnn_eval, transfer_eval]

    # Evidence Gen 4: Quantitative Comparison Table in Terminal & CSV
    comparison_csv_path = "reports/task2_model_comparison.csv"
    print_and_export_comparison_table(all_evals, output_csv=comparison_csv_path)

    # Evidence Gen 4: Confusion Matrices
    cm_path = "reports/figures/task2_5_confusion_matrices.png"
    plot_side_by_side_confusion_matrices(all_evals, class_names=CLASS_NAMES, output_path=cm_path)

    # Evidence Gen 4: ROC Curves
    roc_path = "reports/figures/task2_5_roc_curves.png"
    plot_comparative_roc_curves(all_evals, output_path=roc_path)

    # --------------------------------------------------------------------------
    # 5. Qualitative Error Analysis (Task 2.6)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> STEP 5: Generating Qualitative Error Analysis Grid (10 Misclassifications)")
    print("=" * 80)

    # Use the best model (typically transfer learning) for error inspection
    best_eval = transfer_eval if transfer_eval["f1_score"] >= cnn_eval["f1_score"] else cnn_eval
    print(f"Inspecting failure cases for best performing model: {best_eval['model_name']}")

    error_analysis_path = "reports/figures/task2_6_error_analysis.png"
    generate_error_analysis_grid(
        test_paths=test_paths,
        y_true=best_eval["y_true"],
        y_pred_probs=best_eval["y_pred_probs"],
        class_names=CLASS_NAMES,
        output_path=error_analysis_path,
        max_images=10
    )

    print("\n" + "=" * 80)
    print("               TASK 2 EXECUTION COMPLETE")
    print("=" * 80)
    print("All models trained and checkpoints saved:")
    print(f"  - Custom CNN Checkpoint         : {Path(cnn_ckpt_path).resolve()}")
    print(f"  - Transfer Learning Checkpoint  : {Path(transfer_ckpt_path).resolve()}")
    print("\nVisual Evidence Artifacts Generated:")
    print(f"  - Task 2.1 Augmented Batch Grid : {Path(aug_batch_path).resolve()}")
    print(f"  - Task 2.2 Layer Parameter CSV  : {Path(cnn_csv_path).resolve()}")
    print(f"  - Task 2.2 Custom CNN Curves    : {Path(cnn_curves_path).resolve()}")
    print(f"  - Task 2.3 Transfer Curves      : {Path(transfer_curves_path).resolve()}")
    print(f"  - Task 2.5 Model Comparison CSV : {Path(comparison_csv_path).resolve()}")
    print(f"  - Task 2.5 Confusion Matrices   : {Path(cm_path).resolve()}")
    print(f"  - Task 2.5 Comparative ROC Plot : {Path(roc_path).resolve()}")
    print(f"  - Task 2.6 Error Analysis Grid  : {Path(error_analysis_path).resolve()}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Task 2: Model Development, Training, Evaluation, and Evidence Generation."
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset.")
    parser.add_argument("--cnn-epochs", type=int, default=12, help="Max epochs for Custom CNN.")
    parser.add_argument("--transfer-init-epochs", type=int, default=8,
                        help="Max epochs for Transfer Learning feature extraction.")
    parser.add_argument("--transfer-fine-epochs", type=int, default=6,
                        help="Max epochs for Transfer Learning fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training/evaluation.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Optional limit on training samples for rapid CPU execution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    run_task2(
        data_dir=args.data_dir,
        cnn_epochs=args.cnn_epochs,
        transfer_init_epochs=args.transfer_init_epochs,
        transfer_fine_epochs=args.transfer_fine_epochs,
        batch_size=args.batch_size,
        subsample=args.subsample,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

