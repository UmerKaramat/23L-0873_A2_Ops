"""
VisionGuard - Keras Tuner Toy Demonstration Script
File: scripts/run_tuner_toy.py

Purpose:
Implements Evidence Gen 1 for Task 2.7:
- Demonstrates hyperparameter tuning on the built-in Fashion-MNIST dataset.
- Tunes dense units (32 to 128) and learning rates (1e-2, 1e-3, 1e-4).
- Runs quickly on CPU using Keras Tuner.
- Formats and displays a toy results comparison table in the terminal and exports to CSV.
"""

import os
from pathlib import Path
import pandas as pd
import tensorflow as tf

try:
    import keras_tuner as kt
except ImportError:
    print("Error: 'keras-tuner' is required. Install it using: pip install keras-tuner")
    sys.exit(1)


def build_toy_model(hp):
    """
    Constructs a simple MLP model with tunable hyperparameters.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Tunable Hyperparameter: Number of dense units
    hp_units = hp.Int("units", min_value=32, max_value=128, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # Tunable Hyperparameter: Learning rate
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("=" * 75)
    print("     VisionGuard Task 2.7: Keras Tuner Demonstration (Fashion-MNIST)")
    print("=" * 75)

    # 1. Load built-in Fashion-MNIST dataset
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # Subsample for rapid execution on CPU (5,000 train samples, 1,000 val samples)
    x_train_sub = x_train[:5000]
    y_train_sub = y_train[:5000]
    x_val_sub = x_val[:1000]
    y_val_sub = y_val[:1000]

    # 2. Configure RandomSearch Tuner
    tuner_dir = Path("experiments/tuner_toy")
    tuner_dir.mkdir(parents=True, exist_ok=True)

    tuner = kt.RandomSearch(
        build_toy_model,
        objective="val_accuracy",
        max_trials=6,
        executions_per_trial=1,
        directory=str(tuner_dir),
        project_name="fashion_mnist_search",
        overwrite=True
    )

    print("\nTuner Search Space:")
    tuner.search_space_summary()
    print("-" * 75)
    print("Beginning hyperparameter search across 6 trials (2 epochs per trial)...")

    # 3. Search
    tuner.search(
        x_train_sub,
        y_train_sub,
        epochs=2,
        validation_data=(x_val_sub, y_val_sub),
        batch_size=64,
        verbose=1
    )

    # 4. Extract and format trial results table
    trials = tuner.oracle.get_best_trials(num_trials=6)
    table_rows = []

    for rank, trial in enumerate(trials, start=1):
        units = trial.hyperparameters.get("units")
        lr = trial.hyperparameters.get("learning_rate")
        val_acc = trial.score

        table_rows.append({
            "Rank": rank,
            "Trial_ID": trial.trial_id,
            "Dense_Units": units,
            "Learning_Rate": f"{lr:.0e}",
            "Val_Accuracy": f"{val_acc:.4f}",
            "Status": trial.status
        })

    df_results = pd.DataFrame(table_rows)

    # Export to CSV
    out_csv = Path("reports/task2_7_tuner_toy_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 75)
    print("           Evidence Gen 1: Toy Keras Tuner Results Table")
    print("=" * 75)
    print(df_results.to_string(index=False))
    print("-" * 75)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Optimal Hyperparameters: Units={best_hp.get('units')}, LR={best_hp.get('learning_rate')}")
    print(f"Results exported to: {out_csv.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

