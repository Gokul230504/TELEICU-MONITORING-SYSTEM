# ICU Monitoring System using Video Processing and Deep Learning

# This script provides a framework for developing an ICU monitoring system using video processing and deep learning. 
# We will:
# 1. Load and preprocess video data.
# 2. Train a deep learning model to identify different individuals and activities.
# 3. Implement real-time video processing and predictions.

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define paths
dataset_path = 'healthcare_dataset.csv'

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Load and preprocess data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Save the model
model.save('icu_monitoring_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Function to predict the class of the frame
def predict_frame(frame, model):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    return np.argmax(prediction, axis=1)[0]

# Load pre-trained model
model = tf.keras.models.load_model('icu_monitoring_model.h5')

# Capture real-time video from the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict the class of the frame
    prediction = predict_frame(frame, model)
    
    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ICU Monitoring', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Function to evaluate the model on a test dataset
def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")

# Load and preprocess test data
test_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Evaluate the model
evaluate_model(model, test_generator)
