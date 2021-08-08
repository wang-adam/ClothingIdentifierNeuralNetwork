import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ConvertImage

# Image to test Neural Network on. It gets compressed into a 28x28 matrix
this_image = ConvertImage.compress("shoe.jpg")

# Dataset to train the network on.
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# We want all the values to be between 0 and 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# Category Names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Model consists of a layer to flatten the matrix, 3 Dense layers with 128 neurons each, and then the output layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


# Load the model from disk or train the model if none:
try:
    model.load_weights('model.weights')
    print("Model Weights Loaded")
except:
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    print("Training Model...")
    # Fit the model with the training data
    model.fit(train_images, train_labels, epochs=20)
    # Save the model to disk.
    model.save_weights('model.weights')


# Plots the Image
def plot_image(predictions_array, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue'

    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                    100*np.max(predictions_array)),
               color=color)


# Plots the predictions
def plot_value_array(predictions_array):
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('blue')


# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(this_image)

# Plot the first X test images and their predicted labels.


plt.subplot(1, 2, 1)
plot_image(predictions[0], this_image[0])
plt.subplot(1, 2, 2)
plot_value_array(predictions[0])
plt.tight_layout()
plt.show()
