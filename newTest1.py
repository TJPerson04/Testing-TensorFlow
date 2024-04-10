### All of this code was taken from a guide at https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

print("TesnorFlow version: {}".format(tf.__version__))
print("TesnorFlow Datasets version: ", tfds.__version__)

# Load a preview of the dataset
ds_preview, info = tfds.load('penguins/simple', split='train', with_info=True)
df = tfds.as_dataframe(ds_preview.take(5), info)
print(df)
print(info.features)

class_names = ['Adelie', 'Chinstrap', 'Gentoo']  # This is just to convert the integers 0, 1, and 2 to a penguin species name

# Load the full dataset, split into 80:20 split between training and testing data
ds_split, info = tfds.load("penguins/processed", split=['train[:20%]', 'train[20%:]'], as_supervised=True, with_info=True)

ds_test = ds_split[0]
ds_train = ds_split[1]
assert isinstance(ds_test, tf.data.Dataset)  # Throws an error if the test data is not of the tf.data.Dataset type

# Prints samples of the training and the testing datasets
print(info.features)
df_test = tfds.as_dataframe(ds_test.take(5), info)
print("Test dataset sample: ")
print(df_test)

df_train = tfds.as_dataframe(ds_train.take(5), info)
print("Train dataset sample: ")
print(df_train)

ds_train_batch = ds_train.batch(32)

features, labels = next(iter(ds_train_batch))

print(features)
print(labels)

# Plots... something? tbh idk what "Culmen Length" is
plt.scatter(features[:,0],
            features[:,2],
            c=labels,
            cmap='viridis')

plt.xlabel("Body Mass")
plt.ylabel("Culmen Length")
plt.show()


### USING KERAS ###
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)  # Output layers, represents label predictions
])

predictions = model(features)
tf.nn.softmax(predictions[:5])  # "Converts logits to a probability for each class," I have no clue what that means

### Training the model ###
# Calculating the loss
# Loss is how off a model's predictions are from the desired label

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))

# Calculates gradients used to optimize the model
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},        Loss: {}".format(optimizer.iterations.numpy(),
                                         loss(model, features, labels, training=True).numpy()))


### Training Loop
# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201  # This is the number of times to loop over the dataset collection. Setting this too low or too high could cause inaccuracies

# Note: an epoch is a single pass through the dataset
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in ds_train_batch:  # x is the dataset's features and y is the label (I think basically x is the info given and y is what it is trying to guess)
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout)
        epoch_accuracy.update_state(y, model(x, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))
        
### Visualize the loss function over time
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()


### Set up the test set
test_accuracy = tf.keras.metrics.Accuracy()
ds_test_batch = ds_test.batch(10)

for (x, y) in ds_test_batch:
    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout)
    logits = model(x, training=False)
    prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


### Test with an unlabeled manual test
# Manually creating features for imaginary penguins
predict_dataset = tf.convert_to_tensor([
    [0.3, 0.8, 0.4, 0.5],
    [0.4, 0.1, 0.8, 0.5],
    [0.7, 0.9, 0.8, 0.4]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout)
predicitions = model(predict_dataset, training=False)

for i, logits in enumerate(predicitions):
    class_idx = tf.math.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))