#Define the Directories for the dataset

benign_dir = "/Users/andresfelipecastellanos/LungCancerAI/datasets/BenignCases"
malignant_dir = "/Users/andresfelipecastellanos/LungCancerAI/datasets/MalignantCases"
normal_dir = "/Users/andresfelipecastellanos/LungCancerAI/datasets/NormalCases"

#Load the Images Using a Helper Function 

import os
import cv2
import numpy as np
import random
import tensorflow as tf

#Reduce variability to make the results reproducible

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)




def load_images_from_dir(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
                images.append(img)
                labels.append(label)
    return images, labels


# Load images and labels for each category
benign_images, benign_labels = load_images_from_dir(benign_dir, label=0)      # 0 for Benign
malignant_images, malignant_labels = load_images_from_dir(malignant_dir, label=1)  # 1 for Malignant
normal_images, normal_labels = load_images_from_dir(normal_dir, label=2)        # 2 for Normal

# Combine all images and labels into single arrays
images = benign_images + malignant_images + normal_images
labels = benign_labels + malignant_labels + normal_labels

images = np.array(images)
labels = np.array(labels)

# Print out information to verify everything loaded correctly
print("Total images loaded:", len(images))
print("Total labels loaded:", len(labels))

#Data Splitting and Training:

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Complete the Training Script 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Benign, Malignant, Normal
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to categorical format
y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train_encoded:", y_train_encoded.shape)

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)
