"""
VisionGuard - Evaluation Metrics & Error Analysis Module
File: src/evaluation/metrics.py

Purpose:
Performs quantitative and qualitative evaluation for Task 2.5 & Task 2.6:
- Computes standard classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Generates clean comparison table in terminal and exports to CSV.
- Plots side-by-side Confusion Matrices and comparative ROC curves.
- Identifies misclassified test cases and renders an error analysis image grid.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model_on_dataset(model, dataset, model_name="Model"):
    """
    Evaluates a trained model on a tf.data.Dataset and calculates metrics.

    Returns:
        dict: Containing scalar metrics, confusion matrix, ROC points, and raw predictions.
    """
    print(f"Generating test set predictions for {model_name}...")
    y_pred_probs = model.predict(dataset, verbose=1).ravel()

    # Extract true labels from tf.data.Dataset
    y_true = []
    for _, batch_labels in dataset:
        y_true.extend(batch_labels.numpy().ravel())
    y_true = np.array(y_true, dtype=int)

    # Binary threshold at 0.5
    y_pred = (y_pred_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_probs)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
    }


def print_and_export_comparison_table(results_list, output_csv="reports/task2_model_comparison.csv"):
    """
    Prints a clean comparison table in the terminal and exports it to CSV.
    Required evidence for Task 2.5 in the report.
    """
    rows = []
    for res in results_list:
        rows.append({
            "Model Name": res["model_name"],
            "Accuracy": f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall": f"{res['recall']:.4f}",
            "F1-Score": f"{res['f1_score']:.4f}",
            "ROC-AUC": f"{res['roc_auc']:.4f}",
        })

    df = pd.DataFrame(rows)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("           VisionGuard Task 2.5: Quantitative Test Set Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"Comparison Table Exported: {out_path.resolve()}")
    print("=" * 80 + "\n")
    return df


def plot_side_by_side_confusion_matrices(results_list, class_names=["Real", "AI"],
                                         output_path="reports/figures/task2_5_confusion_matrices.png"):
    """
    Generates and saves side-by-side confusion matrix heatmaps with counts and normalized percentages.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_models = len(results_list)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        cm = res["confusion_matrix"]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{res['model_name']}\nAccuracy: {res['accuracy']:.2%}", fontsize=12, fontweight="bold")

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=10)

        # Annotate each cell with raw count and percentage
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                        ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        ax.set_ylabel("True Ground Truth Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    plt.suptitle("VisionGuard Task 2.5: Test Set Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual Evidence Generated: {out_path.resolve()}")


def plot_comparative_roc_curves(results_list, output_path="reports/figures/task2_5_roc_curves.png"):
    """
    Plots ROC curves for both models on a single graph with chance reference.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

    for idx, res in enumerate(results_list):
        color = colors[idx % len(colors)]
        ax.plot(
            res["fpr"],
            res["tpr"],
            label=f"{res['model_name']} (AUC = {res['roc_auc']:.4f})",
            color=color,
            linewidth=2.2
        )

    # 45-degree random chance line
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", alpha=0.6)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("VisionGuard Task 2.5: Comparative ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual Evidence Generated: {out_path.resolve()}")


def generate_error_analysis_grid(test_paths, y_true, y_pred_probs, class_names=["Real", "AI"],
                                 output_path="reports/figures/task2_6_error_analysis.png", max_images=10):
    """
    Renders a grid of misclassified test images showing true label, predicted label,
    confidence score, and filename for qualitative inspection.
    Required evidence for Task 2.6 in the report.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = (y_pred_probs >= 0.5).astype(int)
    misclassified_indices = np.where(y_true != y_pred)[0]

    print(f"Error Analysis: Found {len(misclassified_indices)} misclassifications out of {len(y_true)} test samples.")

    if len(misclassified_indices) == 0:
        print("Model achieved 100% accuracy on test set; no errors to plot.")
        return

    # Sort errors by confidence distance from decision boundary (most confident mistakes first)
    confidence_error_scores = np.abs(y_pred_probs[misclassified_indices] - 0.5)
    sorted_order = np.argsort(-confidence_error_scores)
    selected_indices = misclassified_indices[sorted_order][:max_images]

    cols = 5
    rows = int(np.ceil(len(selected_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    axes = axes.flatten()

    for idx, mis_idx in enumerate(selected_indices):
        ax = axes[idx]
        img_path = test_paths[mis_idx]
        true_lbl = class_names[y_true[mis_idx]]
        pred_lbl = class_names[y_pred[mis_idx]]
        prob_ai = y_pred_probs[mis_idx]

        # Calculate prediction confidence
        confidence = prob_ai if pred_lbl == "AI" else (1.0 - prob_ai)

        try:
            with Image.open(img_path) as img:
                ax.imshow(img)
            short_name = Path(img_path).name
            if len(short_name) > 18:
                short_name = short_name[:15] + "..."

            title_color = "#c0392b"  # Red for error
            ax.set_title(
                f"File: {short_name}\nTrue: {true_lbl} | Pred: {pred_lbl}\nConf: {confidence:.1%} (P(AI)={prob_ai:.2f})",
                fontsize=9,
                color=title_color,
                fontweight="bold"
            )
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot open:\n{err}", ha="center", va="center")

        ax.axis("off")

    # Turn off any remaining unused axes
    for unused in range(len(selected_indices), len(axes)):
        axes[unused].axis("off")

    plt.suptitle("VisionGuard Task 2.6: Qualitative Error Analysis (Misclassified Test Samples)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual Evidence Generated: {out_path.resolve()}")

