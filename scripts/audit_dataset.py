"""
VisionGuard - Dataset Audit Script
File: scripts/audit_dataset.py

Purpose:
Performs a comprehensive quality audit on the Kaggle dataset
(cashbowman/ai-generated-images-vs-real-images):
- Recursively inspects raw files.
- Accurately identifies class folders ('AI' and 'Real') including nested folder layouts.
- Distinguishes valid images, corrupted images, and non-image files.
- Extracts dimensions (width, height), color mode, format, file size, and SHA-256 hash.
- Identifies exact duplicate files using SHA-256 without double-counting original files.
- Computes min / median / max statistics per class on valid images.
- Exports detailed records to CSV and summary statistics to JSON.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Safe imports of required libraries
try:
    from PIL import Image
    # Prevent decompression bomb warnings on valid large images
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    print("Error: The 'Pillow' library is required. Install it using: pip install pillow")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: 'pandas' and 'numpy' are required. Install them using: pip install pandas numpy")
    sys.exit(1)


# Standard image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def compute_sha256(filepath):
    """
    Computes the SHA-256 checksum of a file in 64 KB chunks.

    Args:
        filepath (Path): Target file path.

    Returns:
        str: Hexadecimal SHA-256 digest, or None if reading failed.
    """
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def determine_class_label(file_path, base_data_dir):
    """
    Accurately identifies the class label ('AI' or 'Real') for cashbowman dataset,
    handling nested directories (e.g. data/raw/AiArtData, data/raw/RealArt,
    or train/val splits) without false substring matches (e.g. 'train' or 'portrait').

    Args:
        file_path (Path): Path of the file.
        base_data_dir (Path): Dataset root path.

    Returns:
        str: 'AI', 'Real', or immediate parent directory name as fallback.
    """
    try:
        relative_parts = file_path.relative_to(base_data_dir).parts
    except ValueError:
        relative_parts = file_path.parts

    # Scan parent folders from immediate parent upwards
    for part in reversed(relative_parts[:-1]):
        part_clean = part.strip().lower()

        # Check explicit AI class folder patterns
        if (
            part_clean in ("ai", "aiartdata", "fake", "synthetic", "synth")
            or "aiart" in part_clean
            or "ai_gen" in part_clean
            or part_clean.startswith("ai_")
            or part_clean.startswith("ai-")
            or part_clean.endswith("_ai")
            or part_clean.endswith("-ai")
        ):
            return "AI"

        # Check explicit Real class folder patterns
        if (
            part_clean in ("real", "realart", "authentic", "original", "orig")
            or "realart" in part_clean
            or "real_image" in part_clean
            or part_clean.startswith("real_")
            or part_clean.startswith("real-")
            or part_clean.endswith("_real")
            or part_clean.endswith("-real")
        ):
            return "Real"

    # Fallback to immediate parent directory
    if len(relative_parts) > 1:
        return relative_parts[-2]
    return "Unknown"


def inspect_single_file(file_path, base_data_dir):
    """
    Inspects a file, distinguishing between valid images, corrupt images, and non-image files.

    Args:
        file_path (Path): Absolute file path.
        base_data_dir (Path): Root dataset path.

    Returns:
        dict: File metadata record.
    """
    file_size_bytes = 0
    try:
        file_size_bytes = os.path.getsize(file_path)
    except Exception:
        pass

    class_label = determine_class_label(file_path, base_data_dir)
    sha256_hash = compute_sha256(file_path)
    ext = file_path.suffix.lower()
    has_image_extension = ext in IMAGE_EXTENSIONS

    record = {
        "filename": file_path.name,
        "extension": ext,
        "relative_path": str(file_path.relative_to(base_data_dir)),
        "class_label": class_label,
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "color_mode": None,
        "image_format": None,
        "sha256_hash": sha256_hash,
        "is_valid_image": False,
        "is_corrupt_image": False,
        "is_non_image_file": False,
        "status": "UNREADABLE",
        "error_message": None,
    }

    # Attempt image decoding via Pillow
    try:
        with Image.open(file_path) as img:
            record["image_format"] = img.format
            record["color_mode"] = img.mode
            record["width"], record["height"] = img.size
            if record["height"] > 0:
                record["aspect_ratio"] = round(record["width"] / record["height"], 3)

            # verify() verifies header and structural consistency
            img.verify()

        # Two-pass verification: reload to confirm pixel decompression succeeds
        with Image.open(file_path) as img:
            img.load()

        record["is_valid_image"] = True
        record["status"] = "VALID"

    except Exception as err:
        record["is_valid_image"] = False
        record["error_message"] = f"{type(err).__name__}: {str(err)}"

        if has_image_extension:
            # Expected to be an image, but decompression or header failed
            record["is_corrupt_image"] = True
            record["status"] = "CORRUPT_IMAGE"
        else:
            # Not an image file (e.g. metadata.json, notes.txt, .gitkeep)
            record["is_non_image_file"] = True
            record["status"] = "NON_IMAGE_FILE"

    return record


def compute_statistics(values):
    """Computes min, median, and max for a list of numerical values."""
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    if not clean_vals:
        return {"min": 0, "median": 0, "max": 0}
    return {
        "min": float(np.min(clean_vals)),
        "median": float(np.median(clean_vals)),
        "max": float(np.max(clean_vals)),
    }


def run_audit(data_dir, output_dir):
    """
    Executes dataset audit, calculates metrics, and exports reports.
    """
    print("=" * 74)
    print("           VisionGuard: Task 1.2 Dataset Audit")
    print("=" * 74)
    print(f"Target Dataset Directory : {data_dir.resolve()}")
    print(f"Reports Output Directory : {output_dir.resolve()}")
    print("-" * 74)

    if not data_dir.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}")
        print("Please extract the Kaggle dataset to 'data/raw/' before running this audit.")
        sys.exit(1)

    # Collect candidate files recursively
    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            # Skip hidden markers (.gitkeep, .DS_Store)
            if f.startswith(".") or f == "Thumbs.db":
                continue
            all_files.append(Path(root) / f)

    total_files_found = len(all_files)
    print(f"Total files discovered   : {total_files_found}")

    if total_files_found == 0:
        print("Warning: No files found in the dataset directory.")
        print("Ensure dataset images are extracted into 'data/raw/'.")
        return

    print("Auditing image integrity and calculating SHA-256 hashes...")
    records = []
    for i, file_path in enumerate(all_files, start=1):
        record = inspect_single_file(file_path, data_dir)
        records.append(record)
        if i % 1000 == 0 or i == total_files_found:
            print(f"  Processed {i}/{total_files_found} files...")

    # Build DataFrame
    df = pd.DataFrame(records)

    # Accurate duplicate detection using SHA-256:
    # keep='first' marks only subsequent duplicate copies as True (the original stays False)
    df["is_duplicate"] = df.duplicated(subset=["sha256_hash"], keep="first")
    df["in_duplicate_set"] = df.duplicated(subset=["sha256_hash"], keep=False)

    total_duplicates_redundant = int(df["is_duplicate"].sum())
    valid_count = int(df["is_valid_image"].sum())
    corrupt_count = int(df["is_corrupt_image"].sum())
    non_image_count = int(df["is_non_image_file"].sum())
    unique_hashes = int(df["sha256_hash"].nunique())

    # Group statistics by class label (only computing dimensional stats on valid images)
    class_summaries = {}
    classes = sorted(df["class_label"].unique())

    for c in classes:
        cdf = df[df["class_label"] == c]
        valid_cdf = cdf[cdf["is_valid_image"]]

        format_counts = valid_cdf["image_format"].value_counts().to_dict()
        color_mode_counts = valid_cdf["color_mode"].value_counts().to_dict()

        class_summaries[c] = {
            "total_files": int(len(cdf)),
            "valid_images": int(len(valid_cdf)),
            "corrupt_images": int(cdf["is_corrupt_image"].sum()),
            "non_image_files": int(cdf["is_non_image_file"].sum()),
            "duplicate_extra_copies": int(cdf["is_duplicate"].sum()),
            "width_px": compute_statistics(valid_cdf["width"].tolist()),
            "height_px": compute_statistics(valid_cdf["height"].tolist()),
            "file_size_kb": compute_statistics(valid_cdf["file_size_kb"].tolist()),
            "formats_detected": {str(k): int(v) for k, v in format_counts.items()},
            "color_modes_detected": {str(k): int(v) for k, v in color_mode_counts.items()},
        }

    # Overall dataset summary
    summary_data = {
        "dataset_path": str(data_dir.resolve()),
        "total_files_audited": total_files_found,
        "valid_images": valid_count,
        "corrupt_images": corrupt_count,
        "non_image_files": non_image_count,
        "duplicate_extra_copies": total_duplicates_redundant,
        "unique_sha256_hashes": unique_hashes,
        "classes_identified": classes,
        "per_class_summary": class_summaries,
    }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dataset_audit_details.csv"
    json_path = output_dir / "dataset_audit_summary.json"

    # Export records
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # Console output
    print("\n" + "=" * 74)
    print("                    DATASET AUDIT SUMMARY")
    print("=" * 74)
    print(f"Total Files Audited     : {total_files_found}")
    print(f"Valid Readable Images   : {valid_count}")
    print(f"Corrupt Image Files     : {corrupt_count}")
    print(f"Non-Image Files Excluded: {non_image_count}")
    print(f"Duplicate Extra Copies  : {total_duplicates_redundant}")
    print(f"Unique SHA-256 Hashes   : {unique_hashes}")
    print(f"Classes Detected        : {', '.join(classes)}")
    print("-" * 74)

    for c, stats in class_summaries.items():
        print(f"\n[Class: {c}]")
        print(f"  Total Files Found   : {stats['total_files']}")
        print(f"  Valid Images        : {stats['valid_images']}")
        print(f"  Corrupt Images      : {stats['corrupt_images']}")
        print(f"  Non-Image Files     : {stats['non_image_files']}")
        print(f"  Duplicate Copies    : {stats['duplicate_extra_copies']}")
        print(f"  Width (px)          : min={stats['width_px']['min']}, median={stats['width_px']['median']}, max={stats['width_px']['max']}")
        print(f"  Height (px)         : min={stats['height_px']['min']}, median={stats['height_px']['median']}, max={stats['height_px']['max']}")
        print(f"  File Size (KB)      : min={stats['file_size_kb']['min']}, median={stats['file_size_kb']['median']}, max={stats['file_size_kb']['max']}")
        print(f"  Formats             : {stats['formats_detected']}")
        print(f"  Color Modes         : {stats['color_modes_detected']}")

    print("\n" + "-" * 74)
    print(f"Detailed CSV exported to : {csv_path.resolve()}")
    print(f"Summary JSON exported to : {json_path.resolve()}")
    print("=" * 74)


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard Task 1.2: Audit raw image dataset integrity, formats, and duplicates."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to the raw dataset directory (default: 'data/raw')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save audit CSV and JSON results (default: 'reports')",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    run_audit(data_dir, output_dir)


if __name__ == "__main__":
    main()
