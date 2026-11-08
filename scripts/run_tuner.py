"""
VisionGuard - Transfer Learning Hyperparameter Tuning Script
File: scripts/run_tuner.py

Purpose:
Implements Evidence Gen 2 for Task 2.7:
- Tunes Learning Rate, Dropout rate, and Number of Unfrozen Layers for VisionGuard.
- Uses Keras Tuner RandomSearch across at least 8 trials.
- Utilizes the real VisionGuard dataset via src/data/pipeline.py.
- Prints the best hyperparameter summary and exports the trial results table to CSV.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import tensorflow as tf

try:
    import keras_tuner as kt
except ImportError:
    print("Error: 'keras-tuner' is required. Install it using: pip install keras-tuner")
    sys.exit(1)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import create_tf_dataset, get_dataset_splits, set_seed


class VisionGuardHyperModel(kt.HyperModel):
    """
    Defines the tunable MobileNetV2 Transfer Learning architecture for Keras Tuner.
    """
    def __init__(self, input_shape=(224, 224, 3)):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        # 1. Hyperparameters to tune
        hp_lr = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4, 1e-5])
        hp_dropout = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
        hp_unfreeze = hp.Choice("unfreeze_layers", values=[10, 20, 30])

        # 2. Base Pre-trained MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = True

        # Freeze all layers except the last hp_unfreeze layers
        freeze_until = max(0, len(base_model.layers) - hp_unfreeze)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        # 3. Model assembly
        inputs = tf.keras.Input(shape=self.input_shape, name="input_tensor")
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        features = base_model(preprocessed, training=False)

        x = tf.keras.layers.GlobalAveragePooling2D(name="tuner_gap")(features)
        x = tf.keras.layers.BatchNormalization(name="tuner_bn")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="tuner_dense")(x)
        x = tf.keras.layers.Dropout(hp_dropout, name="tuner_dropout")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="tuner_output")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Tuned_Transfer_Model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )
        return model


def run_tuning(data_dir="data/raw", max_trials=8, epochs_per_trial=3, batch_size=32, subsample=None):
    set_seed(42)

    print("=" * 80)
    print("      VisionGuard Task 2.7: Transfer Learning Hyperparameter Tuning")
    print("=" * 80)
    print(f"Target Trials   : {max_trials} (Minimum required: 8)")
    print(f"Epochs / Trial  : {epochs_per_trial}")
    print(f"Dataset Dir     : {Path(data_dir).resolve()}")
    print("-" * 80)

    # 1. Prepare datasets
    splits = get_dataset_splits(data_dir=data_dir, test_size=0.15, val_size=0.15)

    train_paths = splits["train_paths"]
    train_labels = splits["train_labels"]
    val_paths = splits["val_paths"]
    val_labels = splits["val_labels"]

    # Optional subsampling for CPU speed
    if subsample and subsample < len(train_paths):
        print(f"Subsampling enabled: using {subsample} training and {subsample // 4} validation samples.")
        train_paths = train_paths[:subsample]
        train_labels = train_labels[:subsample]
        val_paths = val_paths[:subsample // 4]
        val_labels = val_labels[:subsample // 4]

    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True, augment=True)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False, augment=False)

    # 2. Configure Tuner
    tuner_dir = Path("experiments/tuner_visionguard")
    tuner_dir.mkdir(parents=True, exist_ok=True)

    hypermodel = VisionGuardHyperModel(input_shape=(224, 224, 3))

    tuner = kt.RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=1,
        directory=str(tuner_dir),
        project_name="mobilenet_tuning",
        overwrite=True,
        seed=42
    )

    print("\nHyperparameter Search Space:")
    tuner.search_space_summary()
    print("-" * 80)
    print(f"Starting search across {max_trials} trials...")

    # 3. Execute tuning
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1)
    ]

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_per_trial,
        callbacks=callbacks,
        verbose=1
    )

    # 4. Generate results table
    best_trials = tuner.oracle.get_best_trials(num_trials=max_trials)
    table_rows = []

    for rank, trial in enumerate(best_trials, start=1):
        hp_dict = trial.hyperparameters.values
        val_acc = trial.score

        table_rows.append({
            "Rank": rank,
            "Trial_ID": trial.trial_id,
            "Learning_Rate": f"{hp_dict.get('learning_rate'):.0e}",
            "Dropout": hp_dict.get("dropout"),
            "Unfrozen_Layers": hp_dict.get("unfreeze_layers"),
            "Val_Accuracy": f"{val_acc:.4f}" if val_acc is not None else "N/A",
            "Status": trial.status
        })

    df_results = pd.DataFrame(table_rows)

    out_csv = Path("reports/task2_7_tuner_visionguard_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("        Evidence Gen 2: VisionGuard Applied Tuner Results Table")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("-" * 80)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("BEST HYPERPARAMETERS FOUND:")
    print(f"  - Learning Rate   : {best_hp.get('learning_rate')}")
    print(f"  - Dropout Rate    : {best_hp.get('dropout')}")
    print(f"  - Unfrozen Layers : {best_hp.get('unfreeze_layers')}")
    print(f"Full results exported to: {out_csv.resolve()}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Task 2.7: Tune Transfer Learning hyperparameters with Keras Tuner."
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset.")
    parser.add_argument("--max-trials", type=int, default=8, help="Number of trials (min 8).")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Optional limit on sample count for fast CPU runs.")

    args = parser.parse_args()
    run_tuning(
        data_dir=args.data_dir,
        max_trials=args.max_trials,
        epochs_per_trial=args.epochs,
        batch_size=args.batch_size,
        subsample=args.subsample
    )


if __name__ == "__main__":
    main()

