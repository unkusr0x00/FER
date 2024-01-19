import tensorflow as tf
import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# Define paths to your training and testing directories
train_dir = 'Datasets/FER2013/train'
test_dir = 'Datasets/FER2013/test'

# Set the image size and batch size
image_size = (48, 48)
batch_size = 32

# Create an ImageDataGenerator for data augmentation (optional)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    horizontal_flip=True,  # Augmentation: horizontal flip
    zoom_range=0.0,  # Augmentation: zoom (with current dataset not needed, since faces are centered)
    rotation_range=10,  # Augmentation: rotation
    width_shift_range=0.05,  # Augmentation: width shift (only 5% since faces are centered)
    height_shift_range=0.05  # Augmentation: height shift (only 5% since faces are centered)
)

test_datagen = ImageDataGenerator(rescale=1. / 255)  # Only rescaling for test data

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,  # Resize images to (48, 48) for uniformity
    batch_size=batch_size,  # Batch size for training
    color_mode='grayscale',  # FER-2013 images are grayscale
    class_mode='categorical'  # For multi-class classification
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# train_generator and test_generator are now ready to be used in model training and evaluation



# Define the model
model = Sequential()

# First Conv Block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Second Conv Block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Third Conv Block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # Assuming 7 emotions

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()



# Number of epochs to train for
epochs = 10

# Steps per epoch (usually the number of samples in the training set divided by the batch size)
steps_per_epoch = train_generator.samples // train_generator.batch_size

# Validation steps (usually the number of samples in the validation set divided by the batch size)
validation_steps = test_generator.samples // test_generator.batch_size

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps
)



import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print("Test accuracy: ", test_accuracy)

# Predictions on the test set
test_generator.reset()  # Resetting generator to avoid shuffling
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
print("Classification Report")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Conducting error analysis
# This can be done by examining misclassified examples, which can provide insights into what types of errors the model is making
