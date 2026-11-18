"""
Module 2: Model Architecture & Design
Defines:
1. Baseline Custom CNN with Conv2D, Batch Normalization, Max Pooling, and Dropout.
2. Transfer Learning Model (MobileNetV2) with fine-tuning capability.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(
    input_shape=(128, 128, 3),
    num_classes=4,
    dropout_rate=0.3
):
    """
    Constructs a 3-block Convolutional Neural Network.
    Designed for fast convergence and high spatial feature extraction.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),
        
        # Scaling pixel range [0, 255] -> [0, 1]
        layers.Rescaling(1.0 / 255.0, name="rescaling"),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(dropout_rate * 0.5, name="drop1"),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(dropout_rate, name="drop2"),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        layers.Dropout(dropout_rate, name="drop3"),

        # Dense Classifier Head
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.BatchNormalization(name="bn4"),
        layers.Dropout(dropout_rate * 1.5, name="drop4"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="Custom_Satellite_CNN")

    return model


def build_transfer_learning_model(
    input_shape=(128, 128, 3),
    num_classes=4,
    weights=None,
    fine_tune_at=None
):
    """
    Constructs a transfer learning model based on MobileNetV2.
    """
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    except Exception as e:
        print(f"[Warning] Could not load weights '{weights}' ({e}). Falling back to weights=None.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    if fine_tune_at is None and weights is not None:
        base_model.trainable = False  # Feature extraction only
    else:
        base_model.trainable = True

    inputs = layers.Input(shape=input_shape, name="input_image")
    # MobileNetV2 expects [-1, 1] normalized inputs
    x = layers.Rescaling(scale=1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_Satellite")
    return model


if __name__ == "__main__":
    print("--- Testing Model Architectures ---")
    custom_model = build_custom_cnn(input_shape=(128, 128, 3), num_classes=4)
    custom_model.summary()

    transfer_model = build_transfer_learning_model(input_shape=(128, 128, 3), num_classes=4)
    print(f"\nTransfer Model Layers: {len(transfer_model.layers)}")
    print(f"Transfer Model Trainable Weights: {len(transfer_model.trainable_weights)}")
