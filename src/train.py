"""
Module 3: Model Training & Optimization
Handles optimizer configuration, loss function, learning rate scheduling,
callbacks, checkpointing, and training execution.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from src.data_loader import load_dataset_splits
from src.model import build_custom_cnn
from src.utils import plot_training_history

def train_classification_model(
    data_dir="data",
    epochs=12,
    batch_size=32,
    learning_rate=1e-3,
    model_save_path="models/best_satellite_model.h5",
    report_dir="reports"
):
    """
    Executes end-to-end model training with callbacks and evaluation metrics tracking.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print("--- [Module 3] Loading & Preparing Datasets ---")
    train_ds, val_ds, test_ds, split_info = load_dataset_splits(
        data_dir=data_dir,
        batch_size=batch_size
    )
    num_classes = split_info["num_classes"]
    class_names = split_info["class_names"]
    print(f"Detected {num_classes} classes: {class_names}")

    # Build model
    print("\n--- [Module 3] Initializing Neural Network Architecture ---")
    model = build_custom_cnn(
        input_shape=(128, 128, 3),
        num_classes=num_classes,
        dropout_rate=0.3
    )

    # Compile with Adam optimizer & CrossEntropy loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks configuration
    csv_log_path = os.path.join(report_dir, "training_history.csv")
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(csv_log_path)
    ]

    print("\n--- [Module 3] Starting Training Execution ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Plot & save accuracy and loss curves
    curve_path = os.path.join(report_dir, "training_curves.png")
    plot_training_history(history, save_path=curve_path)

    print(f"\n[Module 3 Complete] Best model saved to: {model_save_path}")
    print(f"Training history saved to: {csv_log_path}")
    return model, history, test_ds, split_info

if __name__ == "__main__":
    train_classification_model(epochs=10)
