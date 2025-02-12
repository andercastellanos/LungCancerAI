import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# For evaluation and plotting later
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def load_and_preprocess_images(directory, label, img_size=256):
    """
    Load and preprocess images from a directory.
    Prints debugging messages if files cannot be loaded.
    """
    images = []
    labels = []
    print(f"\nProcessing directory: {directory}")
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return images, labels

    for filename in os.listdir(directory):
        # Check for common image file extensions (adjust as needed)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                # Convert from BGR to RGB and resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                # Optionally add basic quality check (e.g., ignore too-dark images)
                if img.mean() > 10:  # adjust threshold if needed
                    images.append(img)
                    labels.append(label)
                else:
                    print(f"Skipping too dark image: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    print(f"Loaded {len(images)} images with label {label}")
    return images, labels


def create_model(input_shape=(256, 256, 3), num_classes=3):
    """
    Create and return an EfficientNetB7 model with additional layers.
    Fine-tunes only the last layers of the base model.
    """
    base_model = EfficientNetB7(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Fine-tuning: freeze layers up to a specified index
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    # Define dataset directories (update these paths to your dataset location)
    base_dir = "/Users/andresfelipecastellanos/LungCancerAI/datasets"
    benign_dir = os.path.join(base_dir, "BenignCases", "BenignCases")
    malignant_dir = os.path.join(base_dir, "MalignantCases", "MalignantCases")
    normal_dir = os.path.join(base_dir, "NormalCases")

    # Load and process images from each category
    print("Loading images...")
    benign_images, benign_labels = load_and_preprocess_images(benign_dir, label=0)
    malignant_images, malignant_labels = load_and_preprocess_images(malignant_dir, label=1)
    normal_images, normal_labels = load_and_preprocess_images(normal_dir, label=2)

    # Combine data from all categories
    X = np.array(benign_images + malignant_images + normal_images)
    y = np.array(benign_labels + malignant_labels + normal_labels)

    if X.size == 0:
        print("No images were loaded. Please check your dataset paths and image files.")
        return None, None, None, None

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_valid)}")

    # Apply SMOTE to the training data
    print("\nApplying SMOTE for class balancing...")
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    X_train_resampled = X_train_resampled.reshape(-1, 256, 256, 3)
    print("Class distribution after SMOTE:", dict(zip(*np.unique(y_train_resampled, return_counts=True))))

    # Data Augmentation for training data
    train_datagen = ImageDataGenerator(
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        fill_mode='constant',
        cval=0
    )

    # For validation, only rescale (if not already normalized)
    val_datagen = ImageDataGenerator()

    # Create data generators; using a smaller batch size to avoid memory issues
    batch_size = 8
    train_generator = train_datagen.flow(X_train_resampled, y_train_resampled, batch_size=batch_size)
    val_generator = val_datagen.flow(X_valid, y_valid, batch_size=batch_size, shuffle=False)

    # Calculate steps per epoch
    steps_per_epoch = len(X_train_resampled) // batch_size
    validation_steps = len(X_valid) // batch_size

    # Create and compile the model
    print("\nCreating and compiling model...")
    model = create_model()
    model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    model.summary()

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            'lung_cancer_model.weights.h5',  # save only weights
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train the model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Save the complete model (architecture + weights)
    model.save('lung_cancer_model_complete.h5')
    print("Model saved as lung_cancer_model_complete.h5")

    return model, history, X_valid, y_valid


if __name__ == "__main__":
    model, history, X_valid, y_valid = main()

    if model is None:
        print("Model training was not completed due to earlier errors.")
        exit()

    # Evaluate the model on the validation set
    print("\nEvaluating model...")
    y_pred = model.predict(X_valid)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    target_names = ['Benign', 'Malignant', 'Normal']
    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred_classes, target_names=target_names))

    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_valid, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

    # Plot and save the training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Training Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as training_history.png")
