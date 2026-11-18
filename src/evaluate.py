"""
Module 4: Model Evaluation & Validation
Performs unbiased testing on the holdout test set:
- Calculates Accuracy, Precision, Recall, and F1-score.
- Computes and plots the Confusion Matrix.
- Executes Error Analysis by identifying and saving top misclassified samples.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.utils import plot_confusion_matrix

def evaluate_model_performance(
    model_path="models/best_satellite_model.h5",
    test_ds=None,
    class_names=None,
    report_dir="reports"
):
    """
    Evaluates saved model on the test dataset and outputs metrics, reports, and error analysis.
    """
    os.makedirs(report_dir, exist_ok=True)

    print(f"\n--- [Module 4] Loading Model from {model_path} ---")
    model = tf.keras.models.load_model(model_path)

    # Accumulate all test samples and ground truth labels
    print("[Module 4] Running inference across holdout test dataset...")
    y_true_list = []
    y_pred_list = []
    all_images = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        
        y_true_list.extend(labels.numpy())
        y_pred_list.extend(pred_labels)
        all_images.append(images.numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    all_images = np.concatenate(all_images, axis=0)

    # 1. Compute Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n[Test Set Accuracy]: {accuracy * 100:.2f}%")

    # 2. Classification Report (Precision, Recall, F1-Score)
    clf_report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    clf_report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n--- Detailed Classification Report ---")
    print(clf_report_text)

    # Save classification report as text and json
    with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Set Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(clf_report_text)

    with open(os.path.join(report_dir, "metrics_summary.json"), "w") as f:
        json.dump(clf_report_dict, f, indent=4)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=cm_path)

    # 4. Error Analysis: Top Misclassifications
    error_indices = np.where(y_true != y_pred)[0]
    total_errors = len(error_indices)
    print(f"\n[Error Analysis] Total misclassifications: {total_errors} / {len(y_true)} ({total_errors/len(y_true)*100:.2f}%)")

    if total_errors > 0:
        num_to_plot = min(8, total_errors)
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()

        for idx, err_idx in enumerate(error_indices[:num_to_plot]):
            img = all_images[err_idx].astype("uint8")
            true_label = class_names[y_true[err_idx]]
            pred_label = class_names[y_pred[err_idx]]

            axes[idx].imshow(img)
            axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}", color="red", fontsize=10, fontweight="bold")
            axes[idx].axis("off")

        # Hide any unused subplots
        for idx in range(num_to_plot, len(axes)):
            axes[idx].axis("off")

        err_plot_path = os.path.join(report_dir, "error_analysis_samples.png")
        plt.tight_layout()
        plt.savefig(err_plot_path, dpi=300)
        plt.close()
        print(f"[Error Analysis] Misclassified sample visualizations saved to: {err_plot_path}")

    return {
        "accuracy": accuracy,
        "total_test_samples": len(y_true),
        "total_errors": total_errors,
        "report_dict": clf_report_dict
    }
