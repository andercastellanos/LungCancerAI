"""
cnn.py

This script loads image file paths and labels from a dataset,
performs exploratory data analysis (EDA), splits the data into
training/validation/test sets, creates image generators with
data augmentation and a preprocessing function, builds a custom
CNN model, and trains the model using the training data.

Note: This script is adapted from a Jupyter Notebook example.
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

# For evaluation and plotting later
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
            # Apply data augmentation first
            batch_x = next(self.data_gen.flow(batch_x, batch_size=len(batch_x)))
            
        # Then apply mixup
        mixed_x, mixed_y = mixup_data(batch_x, batch_y, self.alpha)
        return mixed_x, mixed_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


def load_and_preprocess_images(directory, label, img_size=224):
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

def create_custom_cnn(input_shape=(256, 256, 3), num_classes=3):
    """
    Create and return a custom CNN model.
    This model uses a few convolutional blocks with BatchNormalization,
    MaxPooling, and Dropout, followed by dense layers.
    """
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    
    return model


def create_improved_model(input_shape=(224, 224, 3), num_classes=3):
    """Create improved CNN with residual connections and attention mechanisms"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution with larger kernel for better feature capture
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('selu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Progressive blocks with increasing filters and residual connections
    filter_sizes = [64, 128, 256, 512]
    for filters in filter_sizes:
        # First residual block
        identity = x
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('selu')(x)
        x = layers.SpatialDropout2D(0.1)(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add skip connection with projection if needed
        if identity.shape[-1] != filters:
            identity = layers.Conv2D(filters, 1, padding='same')(identity)
        x = layers.Add()([x, identity])
        x = layers.Activation('selu')(x)
        
        # Second residual block
        identity = x
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('selu')(x)
        x = layers.SpatialDropout2D(0.1)(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, identity])
        x = layers.Activation('selu')(x)
        
        # Add attention mechanism
        channels = x.shape[-1]
        attention = layers.GlobalAveragePooling2D()(x)
        attention = layers.Dense(channels // 16, activation='selu')(attention)
        attention = layers.Dense(channels, activation='sigmoid')(attention)
        attention = layers.Reshape((1, 1, channels))(attention)
        x = layers.Multiply()([x, attention])
        
        # Pool except for last block
        if filters != filter_sizes[-1]:
            x = layers.MaxPooling2D(2)(x)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='selu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def create_training_callbacks(model_name='lung_cancer_model'):
    """Create callbacks for training"""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'{model_name}_best.weights.h5',  # Fixed file extension
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
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train)
    y_valid_cat = tf.keras.utils.to_categorical(y_valid)
    
    # Create and compile model
    model = create_improved_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=batch_size
    )

    # Train the model - removed workers and use_multiprocessing
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=100,
        validation_data=(X_valid, y_valid_cat),
        callbacks=create_training_callbacks()
    )
    
    return model, history

def predict_with_tta(model, image):
    """Predict with test-time augmentation"""
    # Ensure image is preprocessed (normalized, etc.)
    if image.max() > 1:
        image = image / 255.0
        
    # Get predictions with TTA
    predictions = test_time_augmentation(model, image)
    
    return predictions

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
    
    # Generate augmented versions and predict
    predictions = []
    image_batch = np.repeat(image[np.newaxis], num_augmentations, axis=0)
    for x in datagen.flow(image_batch, batch_size=num_augmentations, shuffle=False):
        pred = model.predict(x)
        predictions.append(pred)
        if len(predictions) >= num_augmentations:
            break
    
    # Average predictions
    return np.mean(predictions, axis=0)

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
    X_train_resampled = X_train_resampled.reshape(-1, 224, 224, 3)
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

 # Create and compile model using the improved training function
    print("\nStarting improved training process...")
    model, history = train_improved_model(
        X_train_resampled, 
        y_train_resampled, 
        X_valid, 
        y_valid, 
        batch_size=8  # Keep your smaller batch size
    )

   

    # Save the complete model (architecture + weights)
    model.save('lung_cancer_improvedtrain_cnn_complete.h5')
    print("Model saved as lung_cancer_improvedtrain_cnn_complete.h5")

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
    plt.savefig('confusion_matrix_improved_cnn.png')
    plt.close()
    print("Confusion matrix saved as confusion_matrix_improved_cnn.png")

    # Plot and save the training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Improved CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Training Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Improved CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history_custom_cnn.png')
    plt.close()
    print("Training history plot saved as training_history_custom_cnn.png")
