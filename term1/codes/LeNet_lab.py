import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load Data for training. Mnist data comes form pre-loaded with TensorFlow.
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)

# Assign training, validation and test data. x for features, y for labels
x_train, y_train           = mnist.train.images, mnist.train.labels
x_validation, y_validation = mnist.validation.images, mnist.validation.labels
x_test, y_test             = mnist.test.images, mnist.test.labels

assert (len(x_train))      == len(y_train)
assert (len(x_validation)) == len(y_validation)
assert (len(x_test))       == len(y_test)


print()
print("Image Shape: {}" .format(x_train[0].shape))  # Print the dimensions of the image, 28*28*1
print()
print("Training Set: {} sample" .format(len(x_train))) # The number of the training images is 55000
print("Validation Set: {} sample" .format(len(x_validation))) # The number of the validation images is 5000
print("Test Set: {} sample" .format(len(x_test)))   # The number of the test images is 10000

"""
The MNIST image is 28*28*1. THe LeNet architecture only accepts 32*32*C, C is the number of color channels.
We need to pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left
and right (28+2+2)=32.
"""

# Pad images wit 0s, transform 28*28*1 to 32*32*1 to be processed in LeNet
x_train      = np.pad(x_train,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_validation = np.pad(x_validation,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test       = np.pad(x_test,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# Print the updated dimension of the image
print("Update Image Shape: {}".format(x_train[0].shape))

# Visualize data
index = random.randint(0,len(x_train)) # Randomly Generate a number from 0~55000. Normal Distribution
image = x_train[index].squeeze() # Delete the 1D to plot the figure

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
# plt.show()
print(y_train[index])

# Pre-process Data, shuffle the training data
x_train, y_train = shuffle(x_train, y_train)

# Define the training epochs and sample size
EPOCHS     = 10
BATCH_SIZE = 128

def LeNet(x):
    # Define hyperparameters
    mu    = 0
    sigma = 0.1

    # Layer 1: Convolution. Input 32x32x1. Output = 28x28x6. 5*5*1 is the size of the filter
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6),  mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation，Transform the linear model to the non-linear model
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6. ksize/strides=[batch, height,width, channels]
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolution. Output = 10x10x6
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean = mu, stddev= sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400. Unfold the 3D matrix to 1D vector
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_w) + fc1_b

    # Activation
    fc1   = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2   = tf.matmul(fc1, fc2_w) + fc2_b

    # Activation
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84.  Output = 10.
    fc3_w  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

"""
Train LeNet to classify MNIST data. x is a placeholder for a batch of input image. 
y is a placeholder for a batch of output labels.
"""
x = tf.placeholder(tf.float32, (None, 32, 32, 1)) # None means it accepts any size of batch later.
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

"""
Training Pipeline. Create a training pipeline that uses the model to classify MNIST data.
"""
rate = 0.001

logits             = LeNet(x)
cross_entropy      = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
loss_operation     = tf.reduce_mean(cross_entropy)
optimizer          = tf.train.AdamOptimizer(learning_rate = rate) # Similar to stochastic gradient descent function
training_operation = optimizer.minimize(loss_operation)

"""
Model Evaluation. Evaluate how well the loss and accuracy of the model for a given dataset.
"""
# Compare the maximum of logits and one_hot_y to determine whether they are equal. Output boolean.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
# Averaging the individual prediction accuracies
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(x_data, y_data):
    num_examples   = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = x_data[offset:end], y_data[offset:end]
        accuracy = sess.run(accuracy_operation, feed_dict = {x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

"""
Train the model
Run the training data through the training pipeline to train the model.
Before each epoch, shuffle the training set.
After each epoch, measure the loss and accuracy of the validation set.
save the model after training.
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y})

        validation_accuracy = evaluate(x_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:3f}".format(validation_accuracy))
        print()

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, './LeNet.ckpt')
    print("Model saved")
