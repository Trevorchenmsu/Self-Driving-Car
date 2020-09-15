import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split
import glob
import matplotlib.image as mpimg

"""
STEP 0: Load the training, validation, and test data
"""
training_file   = 'traffic-signs-data/train.p'
test_file       = 'traffic-signs-data/test.p'

with open(training_file, mode = 'rb') as f:
    train = pickle.load(f)
with open(test_file, mode = 'rb') as f:
    test  = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_test, y_test   = test['features'], test['labels']

save_file = './TrafficSignModel/LeNet_Modified'

"""
Step 1: Dataset Summary & Exploration
"""
# The number of training, validation, test sets, and classes in a label
n_train = len(x_train)
n_test  = len(x_test)

# The shape of the input images.The size of the input images is 32*32*3.
# Thus, we don't need to zero padding the raw images if using LeNet
image_shape = x_train[0].shape

# There are 34799 labels (y_train). Most of them are repeated.
# So using unique function can only count those non-repeated numbers.
n_classes = len(np.unique(y_train))

# print("Image Shape: {}" .format(image_shape))
# print("The number of training Set: {} sample" .format(n_train)) # 34799
# print("The number of test Set: {} sample" .format(n_test))   # 12630
# print("The number of classes: {} sample" .format(n_classes))

# # Visualize data. show image of 10 random data points
# fig, axs = plt.subplots(2,5, figsize = (15, 6))
# fig.subplots_adjust(hspace = .2, wspace=.001)
# axs = axs.ravel()
# for i in range(10):
#     # Randomly Generate a number from 0~34799. Normal Distribution
#     index = random.randint(0, len(x_train))
#     image = x_train[index] # Delete the 1D to plot the figure
#     axs[i].axis('off')
#     axs[i].imshow(image)
#     axs[i].set_title(y_train[index])
# plt.show()
#
# # histogram of label frequency
# hist, bins = np.histogram(y_train, bins = n_classes)
# width  = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align = 'center', width = width)
# plt.show()

"""
Step 2: Design and Test a Model Architecture
Design and implement a deep learning model that learns to recognize traffic signs.
Train and test the model on the German Traffic Sign Dataset.
# """
# Convert the RGB image to grayscale image
x_train_rgb = x_train
x_train_gry = np.sum(x_train/3, axis = 3, keepdims = True)

x_test_rgb = x_test
x_test_gry = np.sum(x_test/3, axis = 3, keepdims = True)

# print('RGB shape:', x_train_rgb.shape)
# print('Grayscale shape:', x_train_gry.shape)

# Since LeNet model uses grayscale image, we use grayscale image to replace the RGB image.
x_train = x_train_gry
x_test  = x_test_gry

# print("RGB->Grayscale, done")

# Visualize RGB VS. Grayscale
# n_rows = 8
# n_cols = 10
# offset = 6000
# fig, axs = plt.subplots(n_rows, n_cols, figsize = (18,14))
# fig.subplots_adjust(hspace = .1, wspace = .001)
# axs = axs.ravel()
#
# for j in range(0,n_rows,2):
#     for i in range(n_cols):
#         index = i + j*n_cols
#         image = x_train_rgb[index + offset]
#         axs[index].axis('off')
#         axs[index].imshow(image)
#     for i in range(n_cols):
#         index = i + j*n_cols + n_cols
#         image = x_train[index + offset - n_cols].squeeze()
#         axs[index].axis('off')
#         axs[index].imshow(image, cmap='gray')
# plt.show()

# A way to see how the images are distributed. If they are evenly distributed, we need to shuffle.
# print(y_train[0:500])

# Calculate the average value of the whole data.
# print("The average value of the training data before normalization is: {}".format(np.mean(x_train)))
# print("The average value of the test data before normalization is: {}".format(np.mean((x_test))))

# Normalize the train and test datasets to (-1,1), why normalize?
x_train_normalized  = (x_train - 128) / 128
x_test_normalized   = (x_test - 128) / 128

# # Print the value after normalization
# print("The average value of the training data after normalization is: {}".format(np.mean(x_train_normalized)))
# print("The average value of the test data after normalization is: {}".format(np.mean(x_test_normalized)))

# Print the image shape before and after normalization
# print("Original shape:", x_train.shape)
# print("Normalized shape:", x_train_normalized.shape)

# Visualize the image before and after normalization
# fig, axs = plt.subplots(1,2, figsize=(10, 3))
# axs = axs.ravel()
#
# axs[0].axis('off')
# axs[0].set_title('normalized')
# axs[0].imshow(x_train_normalized[0].squeeze(), cmap='gray')
#
# axs[1].axis('off')
# axs[1].set_title('original')
# axs[1].imshow(x_train[0].squeeze(), cmap='gray')
# plt.show()

# Generate additional data and split them into training/validation/testing sets
# Four functions for augmenting the dataset:
# random_translate, random_scale, random_warp, and random_brightness

# Define random_translate function
# def random_translate(img):
#     rows,cols,_ = img.shape # Because we don't need to use the third dimension, then set it _
#     # allow translation up to px pixels in x and y directions
#     px = 2
#     dx,dy = np.random.randint(-px,px,2)
#     M = np.float32([[1,0,dx],[0,1,dy]])
#     dst = cv2.warpAffine(img,M,(cols,rows))
#     dst = dst[:,:,np.newaxis]
#     return dst

# Select an image to see how the translate function works
# test_img = x_train_normalized[20002]
# test_dst = random_translate(test_img)
# fig, axs = plt.subplots(1,2, figsize=(10, 3))

# Visualize the image before and after translation
# axs[0].axis('off')
# axs[0].imshow(test_img.squeeze(), cmap='gray')
# axs[0].set_title('original')
#
# axs[1].axis('off')
# axs[1].imshow(test_dst.squeeze(), cmap='gray')
# axs[1].set_title('translated')
# plt.show()

# print('Translate shape in/out:', test_img.shape, test_dst.shape)

# Define a random scaling function
# def random_scaling(img):
#     rows,cols,_ = img.shape
#     # transform limits
#     px = np.random.randint(-2,2)
#     # ending locations
#     pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
#     # starting locations (4 corners)
#     pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
#     M = cv2.getPerspectiveTransform(pts1,pts2)
#     dst = cv2.warpPerspective(img,M,(rows,cols))
#     dst = dst[:,:,np.newaxis]
#     return dst

# Test the scaling function
# test_dst = random_scaling(test_img)
# fig, axs = plt.subplots(1,2, figsize=(10, 3))
#
# axs[0].axis('off')
# axs[0].imshow(test_img.squeeze(), cmap='gray')
# axs[0].set_title('original')
#
# axs[1].axis('off')
# axs[1].imshow(test_dst.squeeze(), cmap='gray')
# axs[1].set_title('scaled')
# plt.show()

# print('Scaling shape in/out:', test_img.shape, test_dst.shape)

# Define the random warp function
# def random_warp(img):
#     rows,cols,_ = img.shape
#
#     # random scaling coefficients
#     rndx = np.random.rand(3) - 0.5
#     rndx *= cols * 0.06   # this coefficient determines the degree of warping
#     rndy = np.random.rand(3) - 0.5
#     rndy *= rows * 0.06
#
#     # 3 starting points for transform, 1/4 way from edges
#     x1 = cols/4
#     x2 = 3*cols/4
#     y1 = rows/4
#     y2 = 3*rows/4
#
#     pts1 = np.float32([[y1,x1],
#                        [y2,x1],
#                        [y1,x2]])
#     pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
#                        [y2+rndy[1],x1+rndx[1]],
#                        [y1+rndy[2],x2+rndx[2]]])
#
#     M = cv2.getAffineTransform(pts1,pts2)
#     dst = cv2.warpAffine(img,M,(cols,rows))
#     dst = dst[:,:,np.newaxis]
#     return dst

# Test the warp function
# test_dst = random_warp(test_img)
# fig, axs = plt.subplots(1,2, figsize=(10, 3))
#
# axs[0].axis('off')
# axs[0].imshow(test_img.squeeze(), cmap='gray')
# axs[0].set_title('original')
#
# axs[1].axis('off')
# axs[1].imshow(test_dst.squeeze(), cmap='gray')
# axs[1].set_title('warped')
# plt.show()
#
# print('Warp shape in/out:', test_img.shape, test_dst.shape)

# Define the brightness function
# def random_brightness(img):
#     shifted = img + 1.0   # shift to (0,2) range, why shift the range?
#     img_max_value = max(shifted.flatten())
#     max_coef = 2.0/img_max_value
#     min_coef = max_coef - 0.1
#     coef = np.random.uniform(min_coef, max_coef)
#     dst = shifted * coef - 1.0
#     return dst

# Test the brightness function
# test_dst = random_brightness(test_img)
# fig, axs = plt.subplots(1,2, figsize=(10, 3))
#
# axs[0].axis('off')
# axs[0].imshow(test_img.squeeze(), cmap='gray')
# axs[0].set_title('original')
#
# axs[1].axis('off')
# axs[1].imshow(test_dst.squeeze(), cmap='gray')
# axs[1].set_title('brightness adjusted')
# plt.show()
# #
# print('Brightness shape in/out:', test_img.shape, test_dst.shape)

# Histogram of label frequency (once again, before data augmentation)
# The purpose to show the histogram twice is compare their change after augmentation
# hist, bins = np.histogram(y_train, bins=n_classes)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.show()
# #
# Print the sample number of every label and the minimum samples
# print(np.bincount(y_train))
# print("minimum samples for any label:", min(np.bincount(y_train)))
# print("maximum samples for any label:", max(np.bincount(y_train)))

# Data augmentation process
# Print the data shape before the augmentation
# print('The shape of training data and labels before data augmentation:', x_train_normalized.shape, y_train.shape)

# input_indices = []
# output_indices = []
#
# for class_n in range(n_classes): # from 0 to 43
#     print(class_n, ': ', end='')
#
#     # Return a 1*N array of the indices (position)that labels in those indices equals to class_n
#     # N is dependent on the repeating times of the label in y_train.
#     class_indices = np.where(y_train == class_n)
#
#     # Knowing the number of samples in every label is to enlarge those labels with less samples
#     # Since class_indices only has one row, we just consider class_indices[0]
#     n_samples = len(class_indices[0])
#
#     if n_samples < 1000:
#         # Augment all the labels with samples less than 800 to 800
#         for i in range(1000 - n_samples):
#             # List append adds every value to the end of the list
#             # 'i%n_samples' repeats several times of n_samples. So the array enlarges util i reaches limit
#             input_indices.append(class_indices[0][i%n_samples])
#
#             # Only add the number of x_train_normalized to the output_indices
#             output_indices.append(x_train_normalized.shape[0])
#
#             # 'class_indices[0][i%n_samples]' represents the position/index in x_train
#             new_img = x_train_normalized[class_indices[0][i%n_samples]]
#             # Every new added image needs to be modified to generate a totally different image
#             new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
#
#             # Add the new image to training image set.
#             x_train_normalized = np.concatenate((x_train_normalized, [new_img]), axis=0)
#             # Since training data is enlarged, training label needs to be enlarged correspondingly
#             y_train = np.concatenate((y_train, [class_n]), axis=0)
#
#            # The below code is used to visualize the number of new augmented samples
#             if i % 50 == 0:
#                 print('|', end='')
#             elif i % 10 == 0:
#                 print('-',end='')
#     print('')

# Print the data shape after the augmentation
# print('The shape of training data and labels after data augmentation::', x_train_normalized.shape, y_train.shape)

# Show comparisons of 5 random augmented data points
# choices = list(range(len(input_indices)))
# picks = []
# for i in range(5):
#     rnd_index = np.random.randint(low=0,high=len(choices))
#     picks.append(choices.pop(rnd_index))
# fig, axs = plt.subplots(2,5, figsize=(15, 6))
# fig.subplots_adjust(hspace = .2, wspace=.001)
# axs = axs.ravel()
# for i in range(5):
#     image = x_train_normalized[input_indices[picks[i]]].squeeze()
#     axs[i].axis('off')
#     axs[i].imshow(image, cmap = 'gray')
#     axs[i].set_title(y_train[input_indices[picks[i]]])
# for i in range(5):
#     image = x_train_normalized[output_indices[picks[i]]].squeeze()
#     axs[i+5].axis('off')
#     axs[i+5].imshow(image, cmap = 'gray')
#     axs[i+5].set_title(y_train[output_indices[picks[i]]])
# plt.show()

# histogram of label frequency
# hist, bins = np.histogram(y_train, bins=n_classes)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.show()

# # # Shuffle the data
x_train_normalized, y_train = shuffle(x_train_normalized, y_train)
# print('Shuffle done')
#
# # Split validation dataset off from training dataset
x_train, x_valid, y_train, y_valid = train_test_split(x_train_normalized, y_train,
                                                test_size=0.25, random_state=42)
# # print("Old x_train size:",len(x_train_normalized))
# # print("New x_train size:",len(x_train))
# # print("x_validation size:",len(x_valid))
#
#
# Define the training epochs and sample size
EPOCHS     = 60
BATCH_SIZE = 100
# #
# #
# # # Original LeNet model architecture
# # # def LeNet(x):
# # #     # Define hyperParameters
# # #     mu    = 0
# # #     sigma = 0.1
# # #
# # #     # Layer 1: Convolution. Input 32x32x1. Output = 28x28x6. 5*5*1, filter size
# # #     w1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6),  mean = mu, stddev = sigma))
# # #     x  = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
# # #     b1 = tf.Variable(tf.zeros(6))
# # #     x  = tf.nn.bias_add(x, b1)
# # #     print("Layer 1 shape:", x.get_shape())
# # #
# # #     # Activation，Transform the linear model to the non-linear model
# # #     x = tf.nn.relu(x)
# # #
# # #     # Pooling. Input = 28x28x6. Output = 14x14x6. ksize/strides=[batch, height,width, channels]
# # #     x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# # #
# # #     # Layer 2: Convolution. Output = 10x10x6
# # #     w2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean = mu, stddev= sigma))
# # #     x  = tf.nn.conv2d(x, w2, strides=[1, 1, 1, 1], padding='VALID')
# # #     b2 = tf.Variable(tf.zeros(16))
# # #     x  = tf.nn.bias_add(x, b2)
# # #
# # #     # Activation
# # #     x = tf.nn.relu(x)
# # #
# # #     # Pooling. Input = 10x10x16. Output = 5x5x16
# # #     x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# # #
# # #     # Flatten. Input = 5x5x16. Output = 400. Unfold the 3D matrix to 1D vector
# # #     x = flatten(x)
# # #
# # #     # Layer 3: Fully Connected. Input = 400. Output = 120.
# # #     w3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
# # #     b3 = tf.Variable(tf.zeros(120))
# # #     x  = tf.add(tf.matmul(x, w3), b3)
# # #
# # #     # Activation
# # #     x  = tf.nn.relu(x)
# # #
# # #     # Dropout. Used to avoid overfitting
# # #     x  = tf.nn.dropout(x, keep_prob)
# # #
# # #     # Layer 4: Fully Connected. Input = 120. Output = 84.
# # #     w4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
# # #     b4 = tf.Variable(tf.zeros(84))
# # #     x  = tf.add(tf.matmul(x, w4), b4)
# # #
# # #     # Activation
# # #     x  = tf.nn.relu(x)
# # #
# # #     # Dropout.
# # #     x  = tf.nn.dropout(x, keep_prob)
# # #
# # #     # Layer 5: Fully Connected. Input = 84.  Output = 43.
# # #     w5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
# # #     b5 = tf.Variable(tf.zeros(43))
# # #     logits = tf.add(tf.matmul(x, w5), b5)
# # #
# # #     return logits
# # #
# # # print('done')
# #
# # Modified LeNet model architecture from the paper
def LeNet_Modified(x):
    # Define hyperParameters
    mu    = 0
    sigma = 0.1

    # Layer 1: Convolution. Input 32x32x1. Output = 28x28x6. 5*5*1, filter size
    w1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6),  mean = mu, stddev = sigma), name="w1")
    x  = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x  = tf.nn.bias_add(x, b1)
    # print("Layer 1 shape:", x.get_shape())

    # Activation，Transform the linear model to the non-linear model
    x = tf.nn.relu(x)

    # Pooling. Input = 28x28x6. Output = 14x14x6. ksize/strides=[batch, height,width, channels]
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = x

    # Layer 2: Convolution. Output = 10x10x6
    w2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean = mu, stddev= sigma), name="w2")
    x  = tf.nn.conv2d(x, w2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x  = tf.nn.bias_add(x, b2)

    # Activation
    x = tf.nn.relu(x)

    # Pooling. Input = 10x10x16. Output = 5x5x16
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = x

    # Layer 3: Convolution. Output = 1x1x400.
    W3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma), name="w3")
    x = tf.nn.conv2d(x, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x, b3)

    # Activation.
    x = tf.nn.relu(x)
    layer3 = x

    # Flatten. Input = 5x5x16. Output = 400. Unfold the 3D matrix to 1D vector
    layer2flat = flatten(layer2)
    # print("layer2flat shape", layer2flat.get_shape())

    # Flatten x.Input = 1*1*400. Output = 400.
    xflat = flatten(x)  # Actually, this is the flatten for layer3.

    # Concatenate layer2flat and xflat. Input = 400+400. Output = 800.
    x = tf.concat([xflat, layer2flat],1)
    # print("x shape:", x.get_shape())

    # Dropout.
    x = tf.nn.dropout(x, keep_prob)

    # Layer 4: Fully Connected. Input = 800. Output = 43.
    w4 = tf.Variable(tf.truncated_normal(shape=(800, 43), mean = mu, stddev = sigma), name="w4")
    b4 = tf.Variable(tf.zeros(43), name="b4")
    logits = tf.add(tf.matmul(x, w4), b4)

    return logits
#
# # print('done')
#
tf.reset_default_graph()
#
x = tf.placeholder(tf.float32, (None, 32, 32, 1)) # None means it accepts any size of batch later.
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, 43)
#
# """
# Training Pipeline. Create a training pipeline that uses the model to classify German traffic signs.
# """
rate = 0.0009
#
logits             = LeNet_Modified(x)
cross_entropy      = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
loss_operation     = tf.reduce_mean(cross_entropy)
optimizer          = tf.train.AdamOptimizer(learning_rate=rate) # Similar to stochastic gradient descent function
training_operation = optimizer.minimize(loss_operation)
#
# """
# Model Evaluation. Evaluate how well the loss and accuracy of the model for a given dataset.
# """
# # Compare the maximum of logits and one_hot_y to determine whether they are equal. Output boolean.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#
# # Averaging the individual prediction accuracies
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
saver = tf.train.Saver()
#
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

# # print('done')
#
# # """
# # Train the model
# # Run the training data through the training pipeline to train the model.
# # Before each epoch, shuffle the training set.
# # After each epoch, measure the loss and accuracy of the validation set.
# # Save the model after training.
# # """
# #
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(x_train)
#
#     print("Training...")
#     print()
#     for i in range(EPOCHS):
#         x_train, y_train = shuffle(x_train, y_train)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = x_train[offset:end], y_train[offset:end]
#             sess.run(training_operation, feed_dict={x:batch_x, y:batch_y, keep_prob: 0.5})
#
#         validation_accuracy = evaluate(x_valid, y_valid)
#         print("EPOCH {} ...".format(i+1))
#         print("Validation Accuracy = {:3f}".format(validation_accuracy))
#         print()
#
#     saver.save(sess, save_file)
#     print("Model saved")
# #
# Now (drumroll) evaluate the accuracy of the model on the test dataset
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver2 = tf.train.import_meta_graph('./TrafficSignModel/LeNet_Modified.meta')
#     saver2.restore(sess, tf.train.latest_checkpoint('./TrafficSignModel'))
#     test_accuracy = evaluate(x_test_normalized, y_test)
#     print("Test Set Accuracy = {:.3f}".format(test_accuracy))

# """
# Step 3: Test the model on new images.
# Take several pictures of traffic signs that are found on the web.
# Run them through the classifier to produce example results.
# """
#
# Load the images and plot them
# Reading in an image
fig, axs = plt.subplots(2, 4, figsize = (4, 2))
fig.subplots_adjust(hspace = .2, wspace = .001)
axs = axs.ravel()

my_images = []

for i, img in enumerate(glob.glob('./TrafficSignsFoundForTest/*x.png')):
    image = cv2.imread(img)
    axs[i].axis('off')
    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    my_images.append(image)
plt.show()

my_images = np.asarray(my_images)
my_images_gry = np.sum(my_images/3, axis = 3, keepdims = True)
my_images_normalized = (my_images_gry - 128)/128
#
# # # Run the predictions
my_labels = [3, 11, 1, 12, 38, 34, 18, 25]
# #
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver3 = tf.train.import_meta_graph('./TrafficSignModel/LeNet_Modified.meta')
#     saver3.restore(sess, tf.train.latest_checkpoint('./TrafficSignModel'))
#     my_accuracy = evaluate(my_images_normalized, my_labels)
#     print("Online Image Test Set Accuracy = {:.3f}".format(my_accuracy))
#
# Visualize the softmax probabilities
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k = 3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./TrafficSignModel/LeNet_Modified.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./TrafficSignModel'))
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 1.0})

    fig, axs = plt.subplots(len(my_images),4, figsize=(12, 14))
    fig.subplots_adjust(hspace = .4, wspace=.2)
    axs = axs.ravel()

# for i, image in enumerate(my_images):
#     axs[4*i].axis('off')
#     axs[4*i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     axs[4*i].set_title('input')
#     guess1 = my_top_k[1][i][0]
#     index1 = np.argwhere(y_valid == guess1)[0]
#     axs[4*i+1].axis('off')
#     axs[4*i+1].imshow(x_valid[index1].squeeze(), cmap='gray')
#     axs[4*i+1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100*my_top_k[0][i][0]))
#     guess2 = my_top_k[1][i][1]
#     index2 = np.argwhere(y_valid == guess2)[0]
#     axs[4*i+2].axis('off')
#     axs[4*i+2].imshow(x_valid[index2].squeeze(), cmap='gray')
#     axs[4*i+2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100*my_top_k[0][i][1]))
#     guess3 = my_top_k[1][i][2]
#     index3 = np.argwhere(y_valid == guess3)[0]
#     axs[4*i+3].axis('off')
#     axs[4*i+3].imshow(x_valid[index3].squeeze(), cmap='gray')
#     axs[4*i+3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100*my_top_k[0][i][2]))
# plt.show()
#
fig, axs = plt.subplots(8,2, figsize=(9, 19))
axs = axs.ravel()

for i in range(len(my_softmax_logits)*2):
    if i%2 == 0:
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(my_images[i//2], cv2.COLOR_BGR2RGB))
    else:
        axs[i].bar(np.arange(n_classes), my_softmax_logits[(i-1)//2])
        axs[i].set_ylabel('Probability')
plt.show()
