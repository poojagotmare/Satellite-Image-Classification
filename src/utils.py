"""
Utility functions for plotting metrics, confusion matrices, and saving visual artifacts.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, save_path="reports/training_curves.png"):
    """
    Plots training and validation accuracy & loss across epochs and saves the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2563EB", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10B981", linewidth=2, linestyle="--")
    ax1.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Loss Plot
    ax2.plot(epochs_range, loss, label="Training Loss", color="#DC2626", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#F59E0B", linewidth=2, linestyle="--")
    ax2.set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Utils] Training curves saved to: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path="reports/confusion_matrix.png"):
    """
    Plots a labeled confusion matrix heatmap and saves the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix - Test Set",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate counts inside heatmap
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Utils] Confusion matrix saved to: {save_path}")
