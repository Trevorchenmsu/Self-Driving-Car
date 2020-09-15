import cv2
import csv
import numpy as np
import os
from os import getcwd
import sklearn

def getLinesFromDrivingLogs(dataPath, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def findImages(dataPath):
    """
    Finds all the images needed for training on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []

    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))

    for directory in dataDirectories:
            lines = getLinesFromDrivingLogs(directory)
            center = []
            left = []
            right = []
            measurements = []
            for line in lines:
                measurements.append(float(line[3]))
                # center.append(directory + '/' + line[0].strip())
                # left.append(directory + '/' + line[1].strip())
                # right.append(directory + '/' + line[2].strip())
                center.append(line[0].strip())
                left.append(line[1].strip())
                right.append(line[2].strip())
            centerTotal.extend(center)
            leftTotal.extend(left)
            rightTotal.extend(right)
            measurementTotal.extend(measurements)

    # directories2 = [x[0] for x in os.walk(dataPath2)]
    # dataDirectories2 = list(filter(lambda directory2: os.path.isfile(directory2 + '/driving_log.csv'), directories2))
    # for directory2 in dataDirectories2:
    #     lines2 = getLinesFromDrivingLogs(directory2)
    #     center = []
    #     left = []
    #     right = []
    #     measurements = []
    #     for line2 in lines2:
    #         measurements.append(float(line2[3]))
    #         center.append(directory2 + '/' + line2[0].strip())
    #         left.append(directory2 + '/' + line2[1].strip())
    #         right.append(directory2 + '/' + line2[2].strip())
    #     centerTotal.extend(center)
    #     leftTotal.extend(left)
    #     rightTotal.extend(right)
    #     measurementTotal.extend(measurements)

    return (centerTotal, leftTotal, rightTotal, measurementTotal)

def combineImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)

def generator(samples, batch_size=32):
    """
    Generate the required images and measurements for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # Change samples into array
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, ELU
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

# # Resize the input data.  CNN models takes in 200*66*3
# def resize_img(input_shape):
#     from keras.backend import tf as ktf
#     return ktf.image.resize_images(input_shape, (66, 200))

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    # model.add(Lambda(resize_img)) # Resize the image
    return model

def nVidiaModel():
    """
    Creates nVidia Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    # model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    # model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    # model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    # model.add(Convolution2D(64,3,3, activation='relu'))
    # model.add(Convolution2D(64,3,3, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Dense(50))
    # model.add(Dense(10))
    # model.add(Dense(1))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='elu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.5))

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Conv2D(64, (3, 3), padding='valid', activation='elu', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='elu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.5))

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), ELU activation (and dropouts)
    model.add(Dense(100, activation='elu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.50))
    model.add(Dense(50, activation='elu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.50))
    model.add(Dense(10, activation='elu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))

    return model


# Reading images locations
centerPaths, leftPaths, rightPaths, measurements = findImages( './BehaviorCloningData/UdacityData')
imagePaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
print('Total Images: {}'.format( len(imagePaths)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_samples2, test_samples = train_test_split(samples, test_size=0.05, random_state=42)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))



batch_size=32
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)

# Model creation
model = nVidiaModel()
# model.summary()

# Compiling and training the model

model.compile(loss='mse', optimizer='adam')
nb_steps = int(len(train_samples)/batch_size)
print(nb_steps)
history_object = model.fit_generator(train_generator, steps_per_epoch=nb_steps,validation_data=validation_generator,
                 validation_steps=int(len(validation_samples)/batch_size), verbose=1, initial_epoch=5)

model.save('model.h5')

# print('Test Loss:', model.evaluate_generator(test_generator, 64))
# print('Test Loss:', model.evaluate_generator(validation_generator, 64))

print(history_object.history.keys())
# print('Loss')
# print(history_object.history['loss'])
# print('Validation Loss')
# print(history_object.history['val_loss'])

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

# python drive.py model.h5 run1
