"""
cnn.py

This script implements a CNN model for lung cancer classification with advanced
features including residual connections, attention mechanisms, mixup augmentation,
and test-time augmentation.
"""

import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# For evaluation and plotting
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def mixup_data(x, y, alpha=0.2):
    """Performs mixup on the input data and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = len(x)
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=16, alpha=0.2, data_gen=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.data_gen = data_gen
        self.indexes = np.arange(len(x))
        
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_indexes]
        batch_y = self.y[batch_indexes]
        
        if self.data_gen:
            batch_x = next(self.data_gen.flow(batch_x, batch_size=len(batch_x)))
            
        mixed_x, mixed_y = mixup_data(batch_x, batch_y, self.alpha)
        return mixed_x, mixed_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def load_and_preprocess_images(directory, label, img_size=224):
    """Load and preprocess images from a directory."""
    images = []
    labels = []
    print(f"\nProcessing directory: {directory}")
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return images, labels

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                if img.mean() > 10:  # Skip very dark images
                    images.append(img)
                    labels.append(label)
                else:
                    print(f"Skipping too dark image: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    print(f"Loaded {len(images)} images with label {label}")
    return images, labels

def create_improved_model(input_shape=(224, 224, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)
    
    # Simpler initial convolution
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Simpler progression
    for filters in [64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def create_training_callbacks(model_name='lung_cancer_model'):
    """Create callbacks for training"""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'{model_name}_best.weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]

def train_improved_model(X_train, y_train, X_valid, y_valid, batch_size=16):
    """Train the improved model with mixup and augmentation"""
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train)
    y_valid_cat = tf.keras.utils.to_categorical(y_valid)
    
    # Create and compile model
    model = create_improved_model()
    model.compile(
        optimizer=Adam(
            learning_rate=1e-3,  # Higher initial learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            weight_decay=1e-5  # Add weight decay
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Modified data augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  # Reduced rotation
        width_shift_range=0.1,  # Reduced shift
        height_shift_range=0.1,  # Reduced shift
        brightness_range=[0.9, 1.1],  # Added brightness variation
        zoom_range=0.1,  # Reduced zoom
        horizontal_flip=True,
        vertical_flip=False,  # Removed vertical flip
        fill_mode='nearest'
    )

    # Create generator
    train_generator = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=batch_size
    )

    # Modified callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience
            restore_best_weights=True,
            min_delta=0.01  # Added minimum improvement threshold
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Less aggressive reduction
            patience=10,  # More patience before reducing
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=100,
        validation_data=(X_valid, y_valid_cat),
        callbacks=callbacks
    )
    
    return model, history

def test_time_augmentation(model, image, num_augmentations=10):
    """Perform test-time augmentation for more robust predictions"""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    predictions = []
    image_batch = np.repeat(image[np.newaxis], num_augmentations, axis=0)
    for x in datagen.flow(image_batch, batch_size=num_augmentations, shuffle=False):
        pred = model.predict(x)
        predictions.append(pred)
        if len(predictions) >= num_augmentations:
            break
    
    return np.mean(predictions, axis=0)

def predict_with_tta(model, image):
    """Predict with test-time augmentation"""
    if image.max() > 1:
        image = image / 255.0
    predictions = test_time_augmentation(model, image)
    return predictions

def main():
    # Define dataset directories
    base_dir = "/Users/andresfelipecastellanos/LungCancerAI/datasets"
    benign_dir = os.path.join(base_dir, "BenignCases", "BenignCases")
    malignant_dir = os.path.join(base_dir, "MalignantCases", "MalignantCases")
    normal_dir = os.path.join(base_dir, "NormalCases")

    # Load and process images
    print("Loading images...")
    benign_images, benign_labels = load_and_preprocess_images(benign_dir, label=0)
    malignant_images, malignant_labels = load_and_preprocess_images(malignant_dir, label=1)
    normal_images, normal_labels = load_and_preprocess_images(normal_dir, label=2)

    # Combine data
    X = np.array(benign_images + malignant_images + normal_images)
    y = np.array(benign_labels + malignant_labels + normal_labels)

    if X.size == 0:
        print("No images were loaded. Please check your dataset paths and image files.")
        return None, None, None, None

    # Normalize pixel values
    X = X / 255.0

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_valid)}")

    # Apply SMOTE
    print("\nApplying SMOTE for class balancing...")
    # Reshape maintaining aspect ratio / flatens images 
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    X_train_resampled = X_train_resampled.reshape(-1, 224, 224, 3)
    print("Class distribution after SMOTE:", dict(zip(*np.unique(y_train_resampled, return_counts=True))))

    # Train model
    print("\nStarting improved training process...")
    model, history = train_improved_model(
        X_train_resampled, 
        y_train_resampled, 
        X_valid, 
        y_valid, 
        batch_size=8
    )

    # Define model directory
    model_dir = "/Users/andresfelipecastellanos/LungCancerAI/models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model with full path
    model_path = os.path.join(model_dir, 'lung_cancer_improved2_cnn_complete.h5')
    model.save(model_path)
    print(f"Model saved as {model_path}")

    return model, history, X_valid, y_valid

if __name__ == "__main__":
    model, history, X_valid, y_valid = main()

    if model is None:
        print("Model training was not completed due to earlier errors.")
        exit()

    # Evaluate model
    print("\nEvaluating model...")
    val_predictions = model.predict(X_valid, batch_size=32)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    
    # Convert one-hot encoded labels back to class indices if needed
    if len(y_valid.shape) > 1:  # If y_valid is one-hot encoded
        y_valid = np.argmax(y_valid, axis=1)

    # Print classification report
    target_names = ['Benign', 'Malignant', 'Normal']
    print("\nClassification Report:")
    print(classification_report(y_valid, val_pred_classes, target_names=target_names))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_valid, val_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_improved_cnn.png')
    plt.close()

    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history_improved_cnn.png')