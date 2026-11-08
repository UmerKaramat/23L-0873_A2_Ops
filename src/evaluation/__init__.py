"""
VisionGuard Evaluation Package
"""

from src.evaluation.metrics import (
    evaluate_model_on_dataset,
    generate_error_analysis_grid,
    plot_comparative_roc_curves,
    plot_side_by_side_confusion_matrices,
    print_and_export_comparison_table,
)

__all__ = [
    "evaluate_model_on_dataset",
    "print_and_export_comparison_table",
    "plot_side_by_side_confusion_matrices",
    "plot_comparative_roc_curves",
    "generate_error_analysis_grid",
]
