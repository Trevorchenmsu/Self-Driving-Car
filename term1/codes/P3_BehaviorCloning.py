import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
import math


"""
STEP 0: Load the data and split it into training data and validation data
"""

samples = []

with open('./BehaviorCloningData/UdacityData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        samples.append(line)

with open('./BehaviorCloningData/MyData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split the data set into training data and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


"""
Step 1: Define Data pre-processing functions: 
random_flip, random_brightness, random_translate, random_scaling, random_warp, random_shadow
"""

# define random flip function
def random_flip(car_image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        car_image = cv2.flip(car_image, 1)
        steering_angle = -steering_angle
    return car_image, steering_angle

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Define random_translate function
def random_translate(image):
    rows,cols,_ = image.shape # Because we don't need to use the third dimension, then set it _
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(image,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    return dst

# Define a random scaling function
def random_scaling(image):
    rows,cols,_ = image.shape
    # transform limits
    px = np.random.randint(-2,2)
    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(image,M,(rows,cols))
    dst = dst[:,:,np.newaxis]
    return dst

# Define the random warp function
def random_warp(image):
    rows,cols,_ = image.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(image,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    return dst

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = image.shape[1] * np.random.rand(), 0
    x2, y2 = image.shape[1] * np.random.rand(), image.shape[0]
    xm, ym = np.mgrid[0: image.shape[0], 0: image.shape[1]]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

"""
Step 2: Design a Model Architecture based on NVIDIA model
"""

# Resize the input data.  CNN models takes in 200*66*3
def resize_img(input_shape):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input_shape, (66, 200))

model = Sequential()
model.add(Cropping2D(cropping = ((65, 20), (2,2)), input_shape = (160, 320, 3))) # Trim the image
model.add(Lambda(resize_img)) # Resize the image
model.add(Lambda(lambda x: x/127.5 - 1)) # Normalize the image

# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.5))

# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.5))

# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10), ELU activation (and dropouts)
model.add(Dense(100, kernel_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))
model.add(Dense(50, kernel_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))
model.add(Dense(10, kernel_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))

# Add a fully connected output layer
model.add(Dense(1))

# Output the information of the model
# model.summary()


"""
Step 3: Compile and train the Model
"""

# Define the generators since the memory of the computer is not enough
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                 # Read center camera images
                 # for i in range(3):
                    name = './BehaviorCloningData/MyData/IMG' + batch_sample[0].split('/')[-1]
                    # Read BGR format image
                    center_image = cv2.imread(name)
                    # center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

                    # Add random_shadow data into training samples
                    image_shadow = random_shadow(center_image)
                    images.append(image_shadow)
                    angles.append(center_angle)

                    # Add random_brightness data into training samples
                    image_brightness = random_brightness(center_image)
                    images.append(image_brightness)
                    angles.append(center_angle)

                    # Add random_flip data into training samples
                    image_flip, angle_flip = random_flip(center_image, center_angle)
                    images.append(image_flip)
                    angles.append(angle_flip)

                    # Add random_translate data into training samples
                    image_translate = random_translate(center_image)
                    images.append(image_translate)
                    angles.append(center_angle)

                    # Add random_scaling data into training samples
                    image_scaling = random_scaling(center_image)
                    images.append(image_scaling)
                    angles.append(center_angle)

                    # Add random_scaling data into training samples
                    image_warp = random_warp(center_image)
                    images.append(image_warp)
                    angles.append(center_angle)

            #change samples into array
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# Define the generator function for model training
batch_size=32
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Compile the model and train the model with generator
nb_steps = math.ceil(len(train_samples)/batch_size)
print(nb_steps)
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=nb_steps,
                    validation_data=validation_generator,
                    validation_steps=math.ceil(len(validation_samples)/batch_size), verbose=1, initial_epoch=5)
# Save model
model.save('model.h5')

# Loss evaluation
print(history_object.history.keys())
# print('Loss')
# print(history_object.history['loss'])
# print('Validation Loss')
# print(history_object.history['val_loss'])

