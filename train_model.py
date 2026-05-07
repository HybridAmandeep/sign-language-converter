"""
=============================================================================
  STEP 3: MLP MODEL TRAINING SCRIPT
  Sign Language to Text/Speech Converter
=============================================================================
  Trains a Multi-Layer Perceptron (MLP) neural network to recognize
  hand gestures from landmark data.

  ARCHITECTURE:
  ┌─────────────────────────────────────────────────┐
  │  Input Layer:   63 features (21 landmarks × 3)  │
  │  Hidden 1:      256 neurons + BatchNorm + Drop  │
  │  Hidden 2:      128 neurons + BatchNorm + Drop  │
  │  Hidden 3:       64 neurons + BatchNorm + Drop  │
  │  Output:         N neurons (softmax)            │
  └─────────────────────────────────────────────────┘

  HOW TO USE:
  1. Run generate_data.py first (or collect_data.py + preprocess.py)
  2. Run: python train_model.py
  3. Output: model/gesture_model.h5 and model/gesture_model.tflite
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "landmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "gesture_model.h5")
MODEL_PATH_TFLITE = os.path.join(MODEL_DIR, "gesture_model.tflite")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.npy")

EPOCHS = 80              # Max epochs (early stopping will likely trigger sooner)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001


def load_data():
    """Load and prepare the landmark data from CSV."""
    print("[STEP 1] Loading data from landmarks.csv...")

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File not found: {CSV_PATH}")
        print("  → Run generate_data.py (or preprocess.py) first!")
        return None, None, None, None, None

    df = pd.read_csv(CSV_PATH)
    print(f"  → Loaded {len(df)} samples with {len(df.columns) - 1} features")
    print(f"  → Gesture classes: {sorted(df['label'].unique())}")
    print(f"  → Samples per class:")
    for label, count in df['label'].value_counts().sort_index().items():
        print(f"     {label}: {count} samples")

    X = df.drop('label', axis=1).values.astype(np.float32)
    y_text = df['label'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    class_names = label_encoder.classes_
    num_classes = len(class_names)
    print(f"  → Number of classes: {num_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    print(f"  → Training samples: {len(X_train)}")
    print(f"  → Testing samples:  {len(X_test)}")
    print()

    return X_train, X_test, y_train, y_test, class_names


def build_model(input_shape, num_classes):
    """
    Build the MLP neural network model.

    Architecture:
    - Input:     63 features (hand landmark coordinates)
    - Dense 256: First hidden layer — learns basic spatial patterns
    - Dense 128: Second hidden layer — learns finger relationships
    - Dense 64:  Third hidden layer — learns gesture-level features
    - Output:    N neurons (one per gesture class, softmax)

    BatchNormalization and Dropout are used to prevent overfitting.
    """
    print("[STEP 2] Building MLP neural network...")

    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),

        # Layer 1: 256 neurons
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Layer 2: 128 neurons
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Layer 3: 64 neurons
        layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print()
    model.summary()
    print()

    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with callbacks."""
    print("[STEP 3] Training the MLP model...")
    print(f"  → Epochs: {EPOCHS} (with early stopping)")
    print(f"  → Batch size: {BATCH_SIZE}")
    print()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate and print results."""
    print()
    print("[STEP 4] Evaluating model performance...")
    print()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✅ Test Accuracy: {accuracy * 100:.2f}%")
    print(f"  📉 Test Loss:     {loss:.4f}")
    print()

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("  CLASSIFICATION REPORT:")
    print("  " + "-" * 55)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("  CONFUSION MATRIX:")
    header = "     " + "  ".join([f"{c:>4}" for c in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:>4}" for v in row])
        print(f"  {class_names[i]:>3}  {row_str}")
    print()

    return accuracy


def save_model(model, class_names):
    """Save model in Keras and TFLite formats."""
    print("[STEP 5] Saving model...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH_H5)
    print(f"  → Saved Keras model:  {MODEL_PATH_H5}")

    np.save(LABEL_MAP_PATH, class_names)
    print(f"  → Saved label map:    {LABEL_MAP_PATH}")

    print("  → Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_PATH_TFLITE, 'wb') as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(MODEL_PATH_TFLITE) / 1024
    h5_size = os.path.getsize(MODEL_PATH_H5) / 1024
    print(f"  → Saved TFLite model: {MODEL_PATH_TFLITE}")
    print(f"  → Keras model size:   {h5_size:.1f} KB")
    print(f"  → TFLite model size:  {tflite_size:.1f} KB")
    print(f"  → Size reduction:     {(1 - tflite_size / h5_size) * 100:.0f}%")
    print()


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  SIGN LANGUAGE MLP MODEL TRAINING")
    print("=" * 60)
    print()

    # Step 1: Load data
    result = load_data()
    if result[0] is None:
        return

    X_train, X_test, y_train, y_test, class_names = result

    # Step 2: Build model
    input_shape = X_train.shape[1]  # 63 features
    num_classes = len(class_names)
    model = build_model(input_shape, num_classes)

    # Step 3: Train
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Step 4: Evaluate
    accuracy = evaluate_model(model, X_test, y_test, class_names)

    # Step 5: Save
    save_model(model, class_names)

    # Final summary
    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Model:          Multi-Layer Perceptron (MLP)")
    print(f"  Architecture:   Input(63) → 256 → 128 → 64 → {num_classes}")
    print(f"  Final Accuracy: {accuracy * 100:.2f}%")
    print(f"  Model saved to: {MODEL_DIR}")
    print()
    if accuracy >= 0.90:
        print("  🎉 Excellent! Your model is ready for deployment!")
    elif accuracy >= 0.80:
        print("  👍 Good accuracy! Consider collecting more data to improve.")
    else:
        print("  ⚠️  Accuracy is low. Try generating more samples.")
    print()
    print("  ✅ You can now run: python web/server.py")
    print()


if __name__ == "__main__":
    main()
