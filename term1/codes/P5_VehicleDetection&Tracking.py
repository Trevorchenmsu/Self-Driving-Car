import glob
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import P4_AdvancedLineFinding_V2 as laneline
from scipy.ndimage.measurements import label
import pickle
from moviepy.editor import VideoFileClip

"""
Step0: Loading the training data
"""

# Loading the training data of vehicles and non-vehicles
car_images = glob.glob('./P5_VehicleDetection_Data/Vehicle/**/*.png')
noncar_images = glob.glob('./P5_VehicleDetection_Data/NonVehicle/**/*.png')
# Print the number of vehicles and non-vehicles
# print(len(car_images), len(noncar_images))
# As we can see, there are about the same number of objects of both classes,
# so we do not need to balance number of images.

# Visualize some the data
# fig, axs = plt.subplots(8, 8, figsize=(16, 16))
# fig.subplots_adjust(hspace=.3, wspace=.001)
# axs = axs.ravel()
# for i in np.arange(32):
#     img = cv2.imread(car_images[np.random.randint(0, len(car_images))])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     axs[i].axis('off')
#     axs[i].set_title('vehicle', fontsize=10)
#     axs[i].imshow(img)
# for i in np.arange(32, 64):
#     img = cv2.imread(noncar_images[np.random.randint(0, len(noncar_images))])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     axs[i].axis('off')
#     axs[i].set_title('Non-vehicle', fontsize=10)
#     axs[i].imshow(img)
# plt.show()

"""
Step1: Features Extraction:  Histogram of Oriented Gradients(HOG), Binned Color, Color Histogram
"""

# Define HOG function. If feature_vec=False, return a feature which is not a vector.
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True to visualize the HOG
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features
# Visualize HOG on example image
# car_example = cv2.imread(car_images[np.random.randint(0, len(car_images))])
# noncar_example = cv2.imread(noncar_images[np.random.randint(0, len(noncar_images))])
# _, car_hog = get_hog_features(car_example, 9, 8, 2, vis=True, feature_vec=True)
# _, noncar_hog = get_hog_features(noncar_example, 9, 8, 2, vis=True, feature_vec=True)
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
# fig.subplots_adjust(hspace=.3, wspace=.001)
# ax1.imshow(car_example)
# ax1.axis('off')
# ax1.set_title('Vehicle Image', fontsize=13)
# ax2.imshow(car_hog, cmap='gray')
# ax2.axis('off')
# ax2.set_title('Vehicle_HOG Image', fontsize=13)
# ax3.imshow(noncar_example, cmap='gray')
# ax3.axis('off')
# ax3.set_title('Non-Vehicle Image', fontsize=13)
# ax4.imshow(noncar_hog,cmap='gray')
# ax4.axis('off')
# ax4.set_title('Non-Vehicle_HOG Image', fontsize=13)
# plt.show()


# Define a function to compute binned color features

 # Define a spatial binned function. Return a feature vector after ravel()

def bin_spatial(img, size=(16, 16)):
    return cv2.resize(img, size).ravel()
# Visualize binned color features on example image
# car_binned = bin_spatial(car_example)
# noncar_binned = bin_spatial((noncar_example))
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
# fig.subplots_adjust(hspace=.3, wspace=.001)
# ax1.imshow(car_example)
# ax1.axis('off')
# ax1.set_title('Vehicle Image', fontsize=13)
# ax2.plot(car_binned)
# ax2.set_title('Vehicle)_Spatially Binned Features', fontsize=13)
# ax3.imshow(noncar_example, cmap='gray')
# ax3.axis('off')
# ax3.set_title('Non-Vehicle Image', fontsize=13)
# ax4.plot(noncar_binned)
# ax4.set_title('Non-Vehicle_Spatially Binned Features', fontsize=13)
# plt.show()

# Define a function for histogram of color. Return a feature vector

def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately.We only need the histogram, not the edges.
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))[0]
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))[0]
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))[0]
    # Concatenate the histograms into a single feature vector.
    hist_features = np.hstack((channel1_hist, channel2_hist, channel3_hist))
    # Return the histograms feature vector
    return hist_features

################## Combine Above Three Features for classifier training ##############################
# Define a function to extract features from a single image window,
# just for a single image rather than a list of images. No concatenate
def img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel):
    file_features = []
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # print('Spatial Features Shape:', spatial_features.shape)
        file_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # print('Histogram of Color Shape:', hist_features.shape)
        file_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell,
                                     cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            # feature_image = cv2.cvtColor(feature_image, cv2.COLOR_LUV2BGR)
            # feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2GRAY)
            # hog_features = get_hog_features(feature_image[:,:], orient,
            #                 pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # print ('HOG Shape', hog_features.shape)
            # Append the new feature vector to the features list
        file_features.append(hog_features)
    return file_features

# Define a function to extract features from a list of images, which can be used for video stream
# extract_features returns a list of feature vectors. Every element is a vector.
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)
        # Achieve the features for one image. img_features return a list of three feature vectors.
        file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel)
        # Concatenate the three features into a feature vector, add to the feature list
        features.append(np.concatenate(file_features))
        # Augment the dataset with flipped images
        feature_image = cv2.flip(feature_image,1)
        file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

"""
Step2: Training a classifier
"""

###########################################################################
# The following code does:
# (1) creates a feature list for the training data.
# (2) normalize the feature list by the StandardScaler() method from sklearn.
# (3) separate the data into training and testing subsets (80% and 20%).
# (4) train the classifier (Linear SVM).
###########################################################################

# Define feature extraction parameters
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16,16)
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
#
# # t1 = time.time()
# car_features = extract_features(car_images, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
#                         orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
#
# # print('Vehicle samples:', len(car_features))
# notcar_features = extract_features(noncar_images, color_space=color_space, spatial_size=spatial_size,
#                         hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
# # print('Non-Vehicle samples:', len(notcar_features))
# # t2 = time.time()
# # print(round(t2-t1, 2), "Seconds to extract HOG features.")
#
# # Create an array stack of feature vectors
# X = np.vstack((car_features, notcar_features)).astype(np.float64)
#
# # Fit a per-column scaler - this will be necessary if combining different types of features:
# # HOG + color_hist/bin_spatial
# X_scaler = StandardScaler().fit(X)
# # Apply the scaler to X
# scaled_X = X_scaler.transform(X)
#
# # Define the labels vector
# y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
#
# # Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
# # print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
# # print('Feature vector length:', len(X_train[0]))
#
# # Use a linear SVC
# svc = LinearSVC(loss='hinge')
# # Check the training time for the SVC
# # t1 = time.time()
# svc.fit(X_train, y_train)
# # t2 = time.time()
# # print(round(t2-t1, 2), 'Seconds to train SVC...')
# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
# # t1 = time.time()
# # n_predict = 10
# # print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
# # print('For these', n_predict, 'labels: ', y_test[0:n_predict])
# # t2 = time.time()
# # print(round(t2-t1, 5), 'Seconds to predict', n_predict, 'labels with SVC')
#
# # Save the classifier
# with open('svc_pickle.p', 'wb') as f:
#   pickle.dump(X_scaler, f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump(svc, f, pickle.HIGHEST_PROTOCOL)

with open('svc_pickle.p', 'rb') as f:
  X_scaler = pickle.load(f)
  svc = pickle.load(f)

"""
# Step3: Sliding window
# """

################################################
# Here we define a sliding window function slide_window to generate a list of boxes
# with predefined parameters  and a draw_boxes to draw the list of boxes on an image.
################################################

# Define a function that takes an image, start and stop positions in both x and y,
# window size (x and y dimensions), and overlap fraction (for both x and y). Returns all the window in the image.
# Every window contains x and y coordinates of staring point and ending point.
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions. Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your classifier, so looping makes sense.
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes on an image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Visualize all the slide window in an image and draw the boxes.
# test_image = './P5_VehicleDetection_Data/test_images/test3.jpg'
# test_image = mpimg.imread(test_image)
# plt.imshow(draw_boxes(image, slide_window(image)))
# plt.show()

# Define a function to extract features in an image. Return a concatenated array of features
def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)

    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image and the list of windows to be searched. Return positive window list.
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=8, pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image. It is a small patch image used for feature extraction
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,  cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# A function to show an image
def show_img(img):
    if len(img.shape)==3: #Color BGR image
        plt.figure()
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else: # Grayscale image
        plt.figure()
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()

# Test the classifier on the test images
# t1 = time.time() # Start time
# test_images = glob.glob('./P5_VehicleDetection_Data/test_images/test*.jpg')
# for image in test_images:
#     image = cv2.imread(image)
#     draw_image = np.copy(image)
#     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 640],
#                     xy_window=(128, 128), xy_overlap=(0.85, 0.85))
#     hot_windows = []
#     hot_windows += (search_windows(image, windows, svc, X_scaler, color_space=color_space,
#                         spatial_size=spatial_size, orient=orient, pix_per_cell=pix_per_cell,
#                         cell_per_block=cell_per_block, hog_channel=hog_channel))
#
#     window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
#     show_img(window_img)
# t2 = time.time()
# print(round(t2-t1, 2), 'Seconds to process test images')

"""
# Step4: Advanced Sliding Windows
# """

###################################################################################
# To improve performance we need to analyze the smallest possible number of windows.
# That is why, we will scan with a search window not across the whole image,
# but only areas where a new car can appear and also we are going to scan areas
# where a car was detected (track cars).
###################################################################################

# Refine detected car position For every detected car
# we are doing to scan with a sliding window the ROI around the previous known position.
# We use multiple scales of windows in order to detect the car and its position more accurate and reliable.

# The following code chunk find windows with a car in a given range with windows of a given scale.
def convert_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

def find_cars(img, ystart, ystop, xstart, xstop, scale, step):
    boxes = []
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) -1
    cells_per_step = step  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract HOG for this patch
            hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            # Extract the image patch
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)+xstart
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((int(xbox_left), int(ytop_draw+ystart)),
                              (int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart))))
    return boxes

# Visulaize the find_cars function to see how efficient it is.
# test_img = cv2.imread('./P5_VehicleDetection_Data/test_images/test4.jpg')
# new_boxes = find_cars(test_img, 400, 600, 100, 1280, 1.5, 2)
# # test4Boxes = open('test4Boxes.p', 'wb')
# # pickle.dump(new_boxes, test4Boxes)
# drawBoxes = draw_boxes(test_img, new_boxes)
# show_img(drawBoxes)
new_boxes = pickle.load( open( "test4Boxes.p", "rb" ))

"""
Step5: Frames Processing
"""

############################################################
# Here we process individual images or videos. To increase performance we skip
# every 2nd frame because we do not expect very fast moving of the detected cars.
# We filter all found windows by a heat map approach (with THRES threshold).
# In order to reduce jitter a function filter applies a simple low-pass filter on the new and the
# previous cars boxes coordinates and sizes.
############################################################
THRES = 3 # Minimal overlapping boxes
THRES_LEN = 32
ALPHA = 0.75 # Filter parameter, weight of the previous measurements
Y_MIN = 440 # The minimum y coordinate to search
n_count = 0 # Frame counter

track_list = [] #[np.array([880, 440, 76, 76])]
#track_list += [np.array([1200, 480, 124, 124])]
heat_p = np.zeros((720, 1280)) # Store prev heat image
boxes_p = [] # Store prev car boxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap # Return updated heatmap

# Zero out pixels below the threshold in the heatmap
def apply_threshold(heatmap, threshold):
    heatmap[heatmap < threshold] = 0
    return heatmap

# Visualize heat and thresholded image
# image = cv2.imread('./P5_VehicleDetection_Data/test_images/test4.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# heatMap = np.zeros_like(image[:,:,0]).astype(np.float)
# heatMap = add_heat(heatMap, new_boxes)
# heatMap = apply_threshold(heatMap,2.0)
# heatMap = np.clip(heatMap, 0, 255)
# plt.imshow(heatMap,cmap='hot')
# plt.axis('off')
# plt.show()

# Smooth the car boxes
def filt(a,b,alpha):
    return a*alpha+(1.0-alpha)*b

# Distance beetween two points
def len_points(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# Create box coordinates out of its center and span
def track_to_box(p):
    return ((int(p[0]-p[2]),int(p[1]-p[3])),(int(p[0]+p[2]), int(p[1]+p[3])))

def draw_labeled_bboxes(img, labels):
    global track_list
    track_list_l = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # img = draw_boxes(np.copy(img), [bbox], color=(255,0,255), thick=3)
        #Size of the found box
        size_x = (bbox[1][0]-bbox[0][0])/2.0
        size_y = (bbox[1][1]-bbox[0][1])/2.0
        asp_d = size_x / size_y
        size_m = (size_x + size_y)/2
        x = size_x+bbox[0][0]
        y = size_y+bbox[0][1]

        # Best rectangle aspect ratio for the box (coefficients from perspective measurements and experiments)
        asp = (y-Y_MIN)/130.0+1.2
        if x > 1050 or x < 230:
            asp*=1.4
        asp = max(asp, asp_d) # for several cars chunk
        size_ya = np.sqrt(size_x*size_y/asp)
        size_xa = int(size_ya*asp)
        size_ya = int(size_ya)

        #If the rectangle on the road, coordinates estimated from a test image
        if x > (-3.049 * y + 1809):
            track_list_l.append(np.array([x, y, size_xa, size_ya]))
            if len(track_list) > 0:
                track_l = track_list_l[-1]
                dist = []
                for track in track_list:
                    dist.append(len_points(track, track_l))
                min_d = min(dist)
                if min_d < THRES_LEN:
                    ind = dist.index(min_d)
                    track_list_l[-1] = filt(track_list[ind], track_list_l[-1], ALPHA)
    track_list = track_list_l
    boxes = []
    for track in track_list_l:
        #print(track_to_box(track))
        # Draw the box on the image
        boxes.append(track_to_box(track))

    return boxes

# Visualize the labeled boxes
# labels = label(heatMap)
# labelbox = draw_labeled_bboxes(np.copy(image), labels)
# # drawBoxes = draw_boxes(image, labelbox)
# heatMap = np.zeros_like(image[:,:,0]).astype(np.float)
# heatMap = add_heat(heatMap, labelbox)
# heatMap = apply_threshold(heatMap,1)
# heatMap = np.clip(heatMap, 0, 255)
# # plt.imshow(drawBoxes)
# plt.imshow(heatMap,cmap='hot')
# plt.axis('off')
# plt.show()

def frame_proc(img, lane = False, video = False, vis = False):
    global heat_p, boxes_p, n_count
    # Skip every second video frame
    if (video and n_count%2==0) or not video:
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        boxes = []
        boxes = find_cars(img, 400, 650, 950, 1280, 2.0, 2)
        boxes += find_cars(img, 400, 500, 950, 1280, 1.5, 2)
        boxes += find_cars(img, 400, 650, 0, 330, 2.0, 2)
        boxes += find_cars(img, 400, 500, 0, 330, 1.5, 2)
        boxes += find_cars(img, 400, 460, 330, 950, 0.75, 3)
        for track in track_list:
            y_loc = track[1]+track[3]
            lane_w = (y_loc*2.841-1170.0)/3.0
            if lane_w < 96:
                lane_w = 96
            lane_h = lane_w/1.2
            lane_w = max(lane_w, track[2])
            xs = int(track[0]-lane_w)
            xf = int(track[0]+lane_w)
            if track[1] < Y_MIN:
                track[1] = Y_MIN
            ys = int(track[1]-lane_h)
            yf = int(track[1]+lane_h)
            if xs < 0: xs=0
            if xf > 1280: xf=1280
            if ys < Y_MIN - 40: ys=Y_MIN - 40
            if yf > 720: yf=720
            size_sq = lane_w / (0.015*lane_w+0.3)
            scale = size_sq / 64.0

            # Apply multi scale image windows
            boxes += find_cars(img, ys, yf, xs, xf, scale, 2)
            boxes += find_cars(img, ys, yf, xs, xf, scale*1.25, 2)
            boxes += find_cars(img, ys, yf, xs, xf, scale*1.5, 2)
            boxes += find_cars(img, ys, yf, xs, xf, scale*1.75, 2)
            if vis:
                cv2.rectangle(img, (int(xs), int(ys)), (int(xf), int(yf)), color=(0,255,0), thickness=3)

        heat = add_heat(heat, boxes)
        heat_l = heat_p + heat
        heat_p = heat
        heat_l = apply_threshold(heat_l,THRES) # Apply threshold to help remove false positives
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat_l, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        #print((labels[0]))
        cars_boxes = draw_labeled_bboxes(img, labels)
        boxes_p = cars_boxes

    else:
        cars_boxes = boxes_p
    #If we was asked to draw the lane line, do it
    if lane:
        if video:
            img = laneline.draw_lane(img, True)
        else:
            img = laneline.draw_lane(img, False)

    imp = draw_boxes(np.copy(img), cars_boxes, color=(0, 0, 255), thick=6)
    if vis:
        imp = draw_boxes(imp, boxes, color=(0, 255, 255), thick=2)
        for track in track_list:
            cv2.circle(imp, (int(track[0]), int(track[1])), 5, color=(255, 0, 255), thickness=4)
    n_count += 1
    return imp

# image = cv2.imread('./P5_VehicleDetection_Data/test_images/test6.jpg')
# show_img(frame_proc(image, lane=True, vis=True))
#
# test_images = glob.glob('./P5_VehicleDetection_Data/test_images/test*.jpg')
# for image in test_images:
#     image = cv2.imread(image)
#     show_img(frame_proc(image, lane=True, vis=True))

# test_images = glob.glob('./P5_VehicleDetection_Data/test_images/test*.jpg')
# for image in test_images:
#     rectangles = []
#     image = cv2.imread(image)
#
#     rectangles.append(find_cars(image, 400, 600, 100, 1280, 1.0, 2))
#     rectangles.append(find_cars(image, 400, 600, 100, 1280, 1.5, 2))
#     rectangles.append(find_cars(image, 400, 600, 100, 1280, 2.0, 2))
#     rectangles.append(find_cars(image, 400, 600, 100, 1280, 3.0, 2))
#     rectangles.append(find_cars(image, 400, 600, 100, 1280, 3.5, 2))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     heatMap = np.zeros_like(image[:,:,0]).astype(np.float)
#     rectangles = [item for sublist in rectangles for item in sublist]
#     heatMap = add_heat(heatMap, rectangles)
#     heatMap = apply_threshold(heatMap,1)
#     labels = label(heatMap)
#     labelbox = draw_labeled_bboxes(np.copy(image), labels)
#     lane_image = laneline.draw_lane(image, False)
#     drawBoxes = draw_boxes(lane_image, labelbox)
#     plt.imshow(drawBoxes)
#     plt.axis('off')
#     plt.show()

"""
Step6: Video processing
"""

from moviepy.editor import VideoFileClip
n_count = 0
laneline.init_params(0.0)

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame_proc(image, lane=True, video=True, vis=False), cv2.COLOR_BGR2RGB)

output_v = './P5_VehicleDetection_Data/project_video_proc.mp4'
clip1 = VideoFileClip('./P5_VehicleDetection_Data/project_video.mp4')
# output_v = './P5_VehicleDetection_Data/challenge_video_pros.mp4'
# clip1 = VideoFileClip('./P5_VehicleDetection_Data/challenge_video.mp4')
clip = clip1.fl_image(process_image)
clip.write_videofile(output_v, audio=False)

