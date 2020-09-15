# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from os import getcwd
import csv

"""
Step1: Image processing and visualization
"""

#显示图片，按任意键显示图片，输入格式为BGR
def displayCV2(img):
    '''
    Utility method to display a CV2 Image
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图片预处理,返回处理后的图片，输入格式为YUV,输出格式为BGR。
# （1）YUV转回BGR。因为用于训练的数据已经被处理成YUV，所以需要转回去。
# （2）图片放大三倍。
# （3）在图片上显示帧数和角度值。
# （4）画绿色实时角度线。如果有预定角，画红色预定角度线。
def process_img_for_visualization(image, angle, pred_angle, frame):
    '''
    Used by visualize_dataset method to format image prior to displaying.
    Converts colorspace YUV back to original BGR.
    Apply text to display steering angle and frame number (within batch to be visualized).
    Apply lines representing steering angle and model-predicted steering angle (if available) to image.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    # Apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=.5,
                color=(200,100,100), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=.5,
                color=(200,100,100), thickness=1)
    # Apply a line representing the steering angle
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0),thickness=4)
    if pred_angle is not None:
        cv2.line(img,(int(w/2),int(h)),(int(w/2+pred_angle*w/4),int(h/2)),(0,0,255),thickness=4)
    return img

# 可视化数据集，图片输入格式YUV。调用上述两个函数，预处理并显示图片。
# 每一帧都预处理图片：显示角度值和帧数文本，角度线，图片。只显示图片，不返回值。
def visualize_dataset(images,angles,pred_angle=None):
    '''
    format the data from the dataset (image, steering angle) and display
    '''
    for frame in range(len(images)):
        if pred_angle is not None:
            img = process_img_for_visualization(images[frame], angles[frame], pred_angle[frame], frame)
        else:
            img = process_img_for_visualization(images[frame], angles[frame], None, frame)
        displayCV2(img)

# 模型训练前的图片预处理。图片输入格式为BGR，输出格式为YUV。预处理包括：
# 1.裁剪图片顶部60*底部20；
# 2.光顺（高斯模糊）；
# 3.修改图片尺寸为200*66*3；
# 4.颜色格式转化：BGR->YUV.
def preprocess_image(img):
    '''
    Method for pre-processing images: this method is the same used in drive.py, except this version uses BGR to YUV
    and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are received in RGB)。
    Original shape: 160x320x3, input shape for neural net: 66x200x3
    '''
    # crop to 80x320x3
    new_img = img[60:140,:,:]
    # Apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # Convert BGR to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)

    return new_img

# 数据集随机曲化处理：1.随机调整亮度；2.随机垂向偏移；3.随机调整清晰度
# 返回处理后的图片和角度。
def random_distort(img, angle):
    '''
    Method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)

    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor

    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)

    return (new_img.astype(np.uint8), angle)

"""
Step2: Data Generator Definition
"""

# 数据训练发生器。功能：
# 1.导入数据；
# 2.预处理数据；
# 3.曲化数据；
# 4.图片和角度反转；
# 5. 填充数据集至batch后逐批提交模型训练。Flag为真时图片无曲化处理。
def generate_training_data(image_paths, angles, batch_size=128, validation_flag=False):
    '''
    Method for the model training data generator to load, process, and distort images,
    then yield them to the model. if 'validation_flag' is true the image is not distorted.
    Flips images with turning angle magnitudes of greater than 0.33,
    as to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    image_paths, angles = shuffle(image_paths, angles)
    train_image, train_angle = ([], [])
    while True:
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = preprocess_image(img)
            if not validation_flag:
                img, angle = random_distort(img, angle)
            train_image.append(img)
            train_angle.append(angle)

            if len(train_image) == batch_size:
                yield shuffle((np.array(train_image), np.array(train_angle)))
                # Clean the former batch content for next loop
                image_paths, angles = shuffle(image_paths, angles)
                train_image, train_angle = ([], [])

            # Flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                train_image.append(img)
                train_angle.append(angle)
                if len(train_image) == batch_size:
                    yield shuffle(np.array(train_image), np.array(train_angle))
                    image_paths, angles = shuffle(image_paths, angles)
                    train_image, train_angle = ([],[])

# 用于显示的数据训练发生器。功能类似上述函数，减少某些功能：
# 1. 反转；
# 2.达到batch大小后无初始化清零和乱序。
def generate_training_data_for_visualization(image_paths, angles, batch_size=20, validation_flag=False):
    '''
    method for loading, processing, and distorting images
    if 'validation_flag' is true the image is not distorted
    '''
    train_image, train_angle = ([], [])
    image_paths, angles = shuffle(image_paths, angles)
    for i in range(batch_size):
        img = cv2.imread(image_paths[i])
        angle = angles[i]
        img = preprocess_image(img)
        if not validation_flag:
            img, angle = random_distort(img, angle)
        train_image.append(img)
        train_angle.append(angle)
    return (np.array(train_image), np.array(train_angle))

"""
Step3: Main program
"""

# Select data sources
MyData = True
UdacityData = True

data_to_use = [MyData, UdacityData]
# When saving my own data, it saves the detailed file path, so the first element is not required.
img_path_prepend = ['', getcwd() + '/BehaviorCloningData/UdacityData/']
csv_path = ['./BehaviorCloningData/MyData/driving_log.csv', './BehaviorCloningData/UdacityData/driving_log.csv']

# Load the training  data
image_paths = []
angles = []

# 联结MyData和UdacityData，全部存储进paths和angles的列表。
for i in range(2):
    if not data_to_use[i]:
        # 0 = MyData, 1 = UdacityData
        print('not using dataset ', i)
        continue
    # Import driving data from csv
    with open(csv_path[i], newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    # Gather data - image paths and angles for center, left, right cameras in each row
    for row in driving_data[0:]:
        # skip it if ~0 speed - not representative of driving behavior
        if float(row[6]) < 0.1 :
            continue

        # get center image path and angle
        image_paths.append(img_path_prepend[i] + row[0])
        angles.append(float(row[3]))

        # get left image path and angle
        image_paths.append(img_path_prepend[i] + row[1])
        angles.append(float(row[3])+0.25)

        # get left image path and angle
        image_paths.append(img_path_prepend[i] + row[2])
        angles.append(float(row[3])-0.25)

# Convert the list format to array format
image_paths = np.array(image_paths)
angles = np.array(angles)

# 这里的paths是路径，不是图片，所以显示出多少个路径，即多少张图片。angle同理。
print('Before the subsampling:', image_paths.shape, angles.shape)

# 由于在simulator中很可能在某些角度下运行很久或者左右变化幅度不大时，就可能重复之前的角度，导致某些角度的数据很多，分布不均匀。
# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 23
avg_samples_per_bin = len(angles)/num_bins
hist, bins = np.histogram(angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
# plt.show()

# 如果数据低于平均数的一半，保持原来的所有数据，即保存概率取1。如果数据量多于平均数的一半，超出的部分去除。
# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
keep_probs = []
target = avg_samples_per_bin * .7
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))

# 删除多余的无效训练数据
remove_list = []
for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            # delete from train_image and train_angle with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(angles, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
# plt.show()

print('After the subsampling:', image_paths.shape, angles.shape)

# visualize a single batch of the data
train_image, train_angle = generate_training_data_for_visualization(image_paths, angles)
# visualize_dataset(train_image, train_angle)

# Make a video to show the training image and angle line
# fps = 2
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter('./Traindata_visual/jitter.avi', fourcc, fps, (600,198))
# for i in range(len(train_image)):
#     img = process_img_for_visualization(train_image[i], train_angle[i], None, i)
#     cv2.imwrite('./Traindata_visual/train' + str(i) +'.jpg', img)
#     img12 = cv2.imread('./Traindata_visual/train' + str(i) +'.jpg')
#     videoWriter.write(img12)
# videoWriter.release()


# split into train/test sets
image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
                                                                                  test_size=0.10, random_state=42)
print('Train Data Shape:', image_paths_train.shape, angles_train.shape)
print('Test Data Shape:', image_paths_test.shape, angles_test.shape)

"""
Convolution Neural Network Definition
"""

# For debugging purposes - don't want to mess with the model if just check in the data
just_checkin_the_data = False

if not just_checkin_the_data:

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ELU())
    # Add a flatten layer
    model.add(Flatten())
    # Add three fully connected layers (depth 100, 50, 10), ELU activation (and dropouts)
    model.add(Dense(100, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(50, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(ELU())
    # Add a fully connected output layer
    model.add(Dense(1))
    # Compile the model and train the model with generator
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    # initialize generators
    batch_size = 64
    train_gen = generate_training_data(image_paths_train, angles_train, validation_flag=False, batch_size=64)
    valid_gen = generate_training_data(image_paths_train, angles_train, validation_flag=True, batch_size=64)
    test_gen  = generate_training_data(image_paths_test, angles_test, validation_flag=True, batch_size=64)

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    nb_steps = int(len(image_paths_train)/batch_size)
    print(nb_steps)

    history_object = model.fit_generator(train_gen, steps_per_epoch=nb_steps,  validation_data=valid_gen,
                          validation_steps=nb_steps, initial_epoch=5, verbose=1, callbacks=[checkpoint] )


    # Save model data
    # model.save('model.h5')
    model.save_weights('model.h5')
    json_string = model.to_json()
    with open('model.json', 'w') as f:
        f.write(json_string)

    # print(model.summary())

    # visualize some predictions
    n = 12
    X_test,y_test = generate_training_data_for_visualization(image_paths_test[:n], angles_test[:n],
                                                             batch_size=n,validation_flag=True)
    y_pred = model.predict(X_test, n, verbose=2)
    # visualize_dataset(X_test, y_test, y_pred)

    # Make a video to show the test image and angle line
    # fps = 2
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # videoWriter = cv2.VideoWriter('./Traindata_visual/test.avi', fourcc, fps, (600,198))
    # for i in range(len(X_test)):
    #     img = process_img_for_visualization(X_test[i], y_test[i], y_pred[i], i)
    #     cv2.imwrite('./Traindata_visual/test' + str(i) +'.jpg', img)
    #     img12 = cv2.imread('./Traindata_visual/test' + str(i) +'.jpg')
    #     videoWriter.write(img12)
    # videoWriter.release()



    # Loss evaluation
    # print(history_object.history.keys())
    # print('Loss')
    # print(history_object.history['loss'])
    # print('Validation Loss')
    # print(history_object.history['val_loss'])

    # plot the training and validation loss for each epoch
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()
    # print('Test Loss:', model.evaluate_generator(test_gen, 40))
    # Output the information of the model

# python drive.py model.h5 run1
# python drive.py model.json







