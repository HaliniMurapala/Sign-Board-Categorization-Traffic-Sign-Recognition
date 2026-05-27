# ─── train.py ──────────────────────────────────────────────
# Train CNN model on GTSRB dataset

import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from labels import LABELS

# ─── Settings ──────────────────────────────────────────────
IMAGE_SIZE   = (32, 32)
NUM_CLASSES  = 43
BATCH_SIZE   = 32
EPOCHS       = 30
DATASET_PATH = "archive/Train"

print("=" * 50)
print("  Traffic Sign Classification — Training")
print("=" * 50)

# ─── Load Dataset ───────────────────────────────────────────
def load_dataset(dataset_path):
    images = []
    labels = []

    print("Loading dataset...")
    for class_id in range(NUM_CLASSES):
        class_path = os.path.join(dataset_path, str(class_id))

        if not os.path.exists(class_path):
            continue

        for img_file in os.listdir(class_path):
            if img_file.endswith(('.png', '.jpg', '.ppm')):
                img_path = os.path.join(class_path, img_file)
                img      = cv2.imread(img_path)

                if img is None:
                    continue

                # Resize to standard size
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(class_id)

    print(f"✅ Loaded {len(images)} images across {NUM_CLASSES} classes")
    return np.array(images), np.array(labels)

# ─── Load and Preprocess ────────────────────────────────────
X, y = load_dataset(DATASET_PATH)

# Normalize pixel values to 0-1
X = X.astype("float32") / 255.0

# Convert labels to one hot encoding
y = to_categorical(y, NUM_CLASSES)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = 0.2,
    random_state= 42
)

print(f"Training samples  : {len(X_train)}")
print(f"Testing samples   : {len(X_test)}")

# ─── Build CNN Model ────────────────────────────────────────
def build_model():
    model = Sequential([

        # Block 1
        Conv2D(32, (3,3), activation='relu',
               input_shape=(32, 32, 3), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_model()
model.summary()

# ─── Compile Model ──────────────────────────────────────────
model.compile(
    optimizer = 'adam',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

# ─── Callbacks ──────────────────────────────────────────────
callbacks = [
    # Stop training if no improvement
    EarlyStopping(
        monitor  = 'val_accuracy',
        patience = 5,
        restore_best_weights = True,
        verbose  = 1
    ),
    # Save best model automatically
    ModelCheckpoint(
        filepath = 'model/traffic_sign_model.h5',
        monitor  = 'val_accuracy',
        save_best_only = True,
        verbose  = 1
    )
]

# ─── Train Model ────────────────────────────────────────────
print("\n🚀 Training started...")

os.makedirs("model", exist_ok=True)

history = model.fit(
    X_train, y_train,
    batch_size      = BATCH_SIZE,
    epochs          = EPOCHS,
    validation_data = (X_test, y_test),
    callbacks       = callbacks,
    verbose         = 1
)

# ─── Evaluate Model ─────────────────────────────────────────
print("\n📊 Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy : {accuracy * 100:.2f}%")
print(f"✅ Test Loss     : {loss:.4f}")
print("\n✅ Model saved to model/traffic_sign_model.h5")
