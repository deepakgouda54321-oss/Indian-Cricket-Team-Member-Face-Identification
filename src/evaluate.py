"""
Model Evaluation and Visualization Module:
- Confusion Matrix Heatmap
- Multi-Model Benchmark Comparison Plots
- Detailed Classification Reports
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str = "artifacts/confusion_matrix.png",
    model_name: str = "Champion Model"
):
    """Generates and saves a publication-quality confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6), dpi=200)
    sns.set_theme(style="white")

    # Indian cricket jersey blue palette
    cmap = sns.light_palette("#0f4c81", as_cmap=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.0,
        linecolor="#f0f2f6",
        square=True
    )

    plt.title(f"Confusion Matrix — {model_name}\nIndian Cricket Team Face Identification", fontsize=12, pad=15, weight="bold")
    plt.xlabel("Predicted Player", fontsize=10, labelpad=10, weight="bold")
    plt.ylabel("Actual Player", fontsize=10, labelpad=10, weight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Confusion matrix saved to: {output_path}")


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: str = "artifacts/model_comparison.png"
):
    """Plots comparative grouped bar chart across all evaluated algorithms."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model_names = list(metrics_dict.keys())
    cv_accs = [metrics_dict[m]["cv_mean_acc"] * 100 for m in model_names]
    test_accs = [metrics_dict[m]["test_acc"] * 100 for m in model_names]
    f1_scores = [metrics_dict[m]["f1_score"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(10, 5.5), dpi=200)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Colors: Indian Navy Blue, Saffron/Orange, Teal
    plt.bar(x - width, cv_accs, width, label="5-Fold CV Accuracy (%)", color="#1f4e78", edgecolor="none")
    plt.bar(x, test_accs, width, label="Test Set Accuracy (%)", color="#f26522", edgecolor="none")
    plt.bar(x + width, f1_scores, width, label="Weighted F1-Score (%)", color="#2e8b57", edgecolor="none")

    plt.title("Model Benchmark Comparison — Indian Cricketer Face Identification", fontsize=12, pad=15, weight="bold")
    plt.xlabel("Machine Learning Algorithm", fontsize=10, labelpad=10, weight="bold")
    plt.ylabel("Score (%)", fontsize=10, labelpad=10, weight="bold")
    plt.xticks(x, model_names, rotation=20, ha="right", fontsize=9)
    plt.ylim(0, 105)
    plt.legend(frameon=True, facecolor="white", edgecolor="#d0d0d0", loc="lower right")

    # Add score labels on test accuracy bars
    for i in range(len(model_names)):
        val = test_accs[i]
        plt.text(x[i], val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8, weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Model comparison chart saved to: {output_path}")


def print_detailed_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> str:
    """Computes and formats a scikit-learn classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print("\n" + "=" * 65)
    print("           DETAILED CLASSIFICATION REPORT")
    print("=" * 65)
    print(report)
    print("=" * 65)
    return report

