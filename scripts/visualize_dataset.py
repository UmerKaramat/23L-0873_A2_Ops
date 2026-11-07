"""
VisionGuard - Visual Exploratory Data Analysis (EDA) Script
File: scripts/visualize_dataset.py

Purpose:
Generates exploratory visualizations and actual-data-based EDA summaries for Task 1.3:
- Creates a balanced sample image grid comparing AI-generated vs Real images.
- Generates comparative histograms for image dimensions (width & height).
- Generates file size distribution plots per class.
- Produces a class balance chart comparing total vs valid vs corrupt files.
- Saves all visual plots to 'reports/figures/'.
- Emits a dynamic markdown summary based on computed data statistics.
"""

import argparse
import os
import sys
from pathlib import Path

# Safe imports
try:
    from PIL import Image
except ImportError:
    print("Error: The 'Pillow' library is required. Install it using: pip install pillow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: 'matplotlib', 'pandas', and 'numpy' are required.")
    print("Install them using: pip install matplotlib pandas numpy")
    sys.exit(1)


def plot_sample_grid(df, base_data_dir, output_dir, samples_per_class=4):
    """
    Renders and saves a balanced sample image grid for each detected class.
    """
    # Only sample from valid images
    valid_df = df[df["is_valid_image"]].copy()
    classes = sorted(valid_df["class_label"].unique())
    num_classes = len(classes)

    if num_classes == 0:
        print("Warning: No valid images found to generate sample grid.")
        return

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(4 * samples_per_class, 4.2 * num_classes))
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, c in enumerate(classes):
        class_df = valid_df[valid_df["class_label"] == c]
        if len(class_df) == 0:
            continue

        sample_subset = class_df.sample(min(samples_per_class, len(class_df)), random_state=42)

        for col_idx, (_, row) in enumerate(sample_subset.iterrows()):
            ax = axes[row_idx, col_idx]
            image_path = base_data_dir / row["relative_path"]

            try:
                with Image.open(image_path) as img:
                    ax.imshow(img)
                title_str = (
                    f"[{c}]\n"
                    f"{int(row['width'])}x{int(row['height'])} px | {row['file_size_kb']} KB\n"
                    f"{row['image_format']} ({row['color_mode']})"
                )
                ax.set_title(title_str, fontsize=10, pad=6)
            except Exception as err:
                ax.text(0.5, 0.5, f"Error displaying:\n{err}", ha="center", va="center")

            ax.axis("off")

        # Blank out remaining columns if class has fewer images than samples_per_class
        for extra_col in range(len(sample_subset), samples_per_class):
            axes[row_idx, extra_col].axis("off")

    plt.suptitle("VisionGuard Task 1.3: Visual Comparison of Sample Images", fontsize=16, y=0.98, fontweight="bold")
    plt.tight_layout()

    out_file = output_dir / "task1_3_sample_grid.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved sample grid figure: {out_file.resolve()}")


def plot_class_balance(df, output_dir):
    """
    Renders and saves a bar chart showing class balance and file integrity.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = sorted(df["class_label"].unique())
    counts = [len(df[df["class_label"] == c]) for c in classes]
    valid_counts = [len(df[(df["class_label"] == c) & (df["is_valid_image"])]) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    bar1 = ax.bar(x - width / 2, counts, width, label="Total Files", color="#3498db")
    bar2 = ax.bar(x + width / 2, valid_counts, width, label="Valid Images", color="#2ecc71")

    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Dataset Class Balance and File Integrity", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for b in bar1:
        height = b.get_height()
        ax.annotate(f"{height}", xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    for b in bar2:
        height = b.get_height()
        ax.annotate(f"{height}", xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    out_file = output_dir / "task1_3_class_balance.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved class balance figure: {out_file.resolve()}")


def plot_dimension_distributions(df, output_dir):
    """
    Renders comparative histograms of image widths and heights per class on valid images.
    """
    valid_df = df[df["is_valid_image"]].copy()
    classes = sorted(valid_df["class_label"].unique())
    colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Width distribution
    for i, c in enumerate(classes):
        c_widths = valid_df[valid_df["class_label"] == c]["width"].dropna()
        if len(c_widths) > 0:
            axes[0].hist(c_widths, bins=25, alpha=0.6, label=f"Class: {c}", color=colors[i % len(colors)], edgecolor="black")

    axes[0].set_title("Image Width Distribution by Class", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Width (pixels)", fontsize=11)
    axes[0].set_ylabel("Image Count", fontsize=11)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Height distribution
    for i, c in enumerate(classes):
        c_heights = valid_df[valid_df["class_label"] == c]["height"].dropna()
        if len(c_heights) > 0:
            axes[1].hist(c_heights, bins=25, alpha=0.6, label=f"Class: {c}", color=colors[i % len(colors)], edgecolor="black")

    axes[1].set_title("Image Height Distribution by Class", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Height (pixels)", fontsize=11)
    axes[1].set_ylabel("Image Count", fontsize=11)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("VisionGuard Task 1.3: Image Dimension Distributions", fontsize=15, y=1.02, fontweight="bold")
    plt.tight_layout()

    out_file = output_dir / "task1_3_dimension_distributions.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved dimension distribution figure: {out_file.resolve()}")


def plot_filesize_distribution(df, output_dir):
    """
    Renders file size distribution histograms for each class on valid images.
    """
    valid_df = df[df["is_valid_image"]].copy()
    classes = sorted(valid_df["class_label"].unique())
    colors = ["#9b59b6", "#16a085", "#d35400", "#2c3e50"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, c in enumerate(classes):
        c_sizes = valid_df[valid_df["class_label"] == c]["file_size_kb"].dropna()
        if len(c_sizes) > 0:
            ax.hist(c_sizes, bins=30, alpha=0.6, label=f"Class: {c}", color=colors[i % len(colors)], edgecolor="black")

    ax.set_title("Image File Size Distribution (KB) by Class", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("File Size (KB)", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out_file = output_dir / "task1_3_filesize_distribution.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved file size distribution figure: {out_file.resolve()}")


def export_eda_summary_markdown(df, output_path):
    """
    Generates a dynamic markdown snippet summarizing the visual EDA findings from actual data.
    """
    valid_df = df[df["is_valid_image"]].copy()
    classes = sorted(df["class_label"].unique())
    corrupt_count = int(df["is_corrupt_image"].sum()) if "is_corrupt_image" in df.columns else int((~df["is_valid_image"]).sum())
    non_image_count = int(df["is_non_image_file"].sum()) if "is_non_image_file" in df.columns else 0

    lines = [
        "### Task 1.3 Visual EDA Summary (Generated from Actual Dataset Audit)",
        "",
        f"- **Total Files Audited:** {len(df)}",
        f"- **Valid Images Plotted:** {len(valid_df)}",
        f"- **Corrupt Images Excluded:** {corrupt_count}",
        f"- **Non-Image Files Excluded:** {non_image_count}",
        f"- **Detected Classes:** {', '.join(classes)}",
        "",
        "#### Per-Class Quantitative Breakdown (Valid Images Only):",
        "",
        "| Class | Valid Images | Width Min/Med/Max (px) | Height Min/Med/Max (px) | File Size Min/Med/Max (KB) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for c in classes:
        cdf = valid_df[valid_df["class_label"] == c]
        if len(cdf) == 0:
            continue
        w_min, w_med, w_max = cdf["width"].min(), cdf["width"].median(), cdf["width"].max()
        h_min, h_med, h_max = cdf["height"].min(), cdf["height"].median(), cdf["height"].max()
        s_min, s_med, s_max = cdf["file_size_kb"].min(), cdf["file_size_kb"].median(), cdf["file_size_kb"].max()
        lines.append(
            f"| **{c}** | {len(cdf)} | {w_min:.0f} / {w_med:.0f} / {w_max:.0f} | {h_min:.0f} / {h_med:.0f} / {h_max:.0f} | {s_min:.1f} / {s_med:.1f} / {s_max:.1f} |"
        )

    lines.extend([
        "",
        "#### Key Visual & Structural Observations:",
        "- Image dimension consistency across classes informs input resizing decisions for Task 2 modeling.",
        "- Formats and color channels (e.g. RGB) are verified prior to pipeline preprocessing.",
        "- Generated plots are saved in `reports/figures/` for the final report.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Exported dynamic EDA markdown summary: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Task 1.3: Generate Visual EDA plots and summaries from audited dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to the raw dataset directory (default: 'data/raw')",
    )
    parser.add_argument(
        "--audit-csv",
        type=str,
        default="reports/dataset_audit_details.csv",
        help="Path to audit details CSV from Task 1.2 (default: 'reports/dataset_audit_details.csv')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory to save generated plots (default: 'reports/figures')",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of sample images per class to plot in the grid (default: 4)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    audit_csv = Path(args.audit_csv)
    output_dir = Path(args.output_dir)

    print("=" * 74)
    print("        VisionGuard: Task 1.3 Visual Exploratory Data Analysis")
    print("=" * 74)

    if not audit_csv.exists():
        print(f"Error: Audit CSV not found at '{audit_csv.resolve()}'.")
        print("Please run 'python scripts/audit_dataset.py' first to generate the audit data.")
        sys.exit(1)

    print(f"Loading audit data from : {audit_csv.resolve()}")
    df = pd.read_csv(audit_csv)
    print(f"Loaded {len(df)} records.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample comparison grid
    plot_sample_grid(df, data_dir, output_dir, samples_per_class=args.samples)

    # 2. Class balance
    plot_class_balance(df, output_dir)

    # 3. Dimension distributions
    plot_dimension_distributions(df, output_dir)

    # 4. File size distribution
    plot_filesize_distribution(df, output_dir)

    # 5. Export summary markdown snippet for report integration
    summary_md_path = output_dir.parent / "task1_3_eda_summary.md"
    export_eda_summary_markdown(df, summary_md_path)

    print("=" * 74)
    print("Visual EDA complete. All figures and summaries generated.")
    print("=" * 74)


if __name__ == "__main__":
    main()
