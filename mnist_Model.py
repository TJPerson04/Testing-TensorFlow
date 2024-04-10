# Libraries
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import tensorflow_datasets as tfds

import pathlib

### Load the dataset
img_height = 28
img_width = 28

ds_split, info = tfds.load("mnist", split=['train[:20%]', 'train[20%:]'], as_supervised=True, with_info=True)

train_ds = ds_split[0]
val_ds = ds_split[1]
assert isinstance(val_ds, tf.data.Dataset)  # Throws an error if the test data is not of the tf.data.Dataset type

ds_train_batch = train_ds.batch(32)

features, labels = next(iter(ds_train_batch))

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Manually creating the class_names bc idk how to get it from this data

print(class_names)

### Visualize the data
plt.figure(figsize=(10, 10))
for image in features:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(features[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()

print(features.shape)
print(labels.shape)

### Change the data from tf.uint8 to tf.float32
train_ds = tf.data.Dataset.map(train_ds, lambda x: x + 1)
# val_ds = tf.data.Dataset.map(val_ds, lambda x: float(x))
list(train_ds.as_numpy_iterator())

### Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

### Standardize the data
# The RGB channel values are from 0 - 255, we want them from 0 - 1
# normalization_layer = layers.Rescaling(1./255)

### Data augmentation
# Without this the model has a high accuracy when training,
# but a very low accuracy when validating, which is a sign of overfitting.
# To fix this one solution is to randomly change the training data
# in a believable way, exposing the model to more aspects of the data, allowing it to generalize better.

# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal",
#                           input_shape=(img_height,
#                                        img_width,
#                                        1)),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.1)

#     ]
# )

# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")

# plt.show()


### Create the model
num_classes = len(class_names)

model = Sequential([
    # data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    # layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


### Train the model
# Train the model for 10 epochs
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize the training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label = "Validation Accuracy")
plt.legend(loc = "lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = "Training Loss")
plt.plot(epochs_range, val_loss, label = "Validation Loss")
plt.legend(loc = "upper right")
plt.title("Training and Validation Loss")
plt.show()


### Save the model as a .keras zip file
model.save('my_model.keras')