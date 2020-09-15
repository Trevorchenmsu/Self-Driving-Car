import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from sklearn.utils import shuffle

from keras.datasets import cifar10

nb_classes = 10
rate = 0.001
EPOCHS     = 10
BATCH_SIZE = 100
saver = tf.train.Saver()
save_file = './TrafficSign_Cifar10/Cifar10'

# Load traffic signs data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Split data into training and validation sets.
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                test_size=0.3, random_state=42, stratify = y_train)
# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))

keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, nb_classes)


# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the traffic signs
# model.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes) # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8W) + fc8b

# Define loss, training, accuracy operations.
cross_entropy      = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
loss_operation     = tf.reduce_mean(cross_entropy)
optimizer          = tf.train.AdamOptimizer() # Similar to SGD function
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

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
        accuracy = sess.run(accuracy_operation, feed_dict = {x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train and evaluate the feature extraction model.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    num_examples = len(x_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(x_valid, y_valid)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:3f}".format(validation_accuracy))
        print()

    saver.save(sess, save_file)
    print("Model saved")



