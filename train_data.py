import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
# Define your model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Binary classification: Drowsy or not
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load your dataset
# Assuming you have a directory structure like:
# dataset/
#   train/
#     drowsy/
#     non_drowsy/
#   validation/
#     drowsy/
#     non_drowsy/

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'newdata/train',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'newdata/validation',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Save the model
model.save('model/conCat2.keras')
loss,accuracy=model.evaluate(validation_generator)
print(f"validation Accuracy: {accuracy* 100:.2f}%")