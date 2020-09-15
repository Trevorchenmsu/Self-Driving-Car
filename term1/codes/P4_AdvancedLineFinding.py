import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML

"""
Step1: Camera Calibration
"""

#Number of corners to find
x_cor = 9
y_cor = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((y_cor*x_cor,3), np.float32)
objp[:,:2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./P4_AdvanceLindFind/camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (x_cor,y_cor), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (x_cor,y_cor), corners, ret)
#         cv2.imshow('img', img)
#         cv2.waitKey(500)
# cv2.destroyAllWindows()

"""
Step2: Distortion Correction
"""

# Output the camera matrix and distortion coefficient
img = cv2.imread('./P4_AdvanceLindFind/camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration.p", "wb" ) )

# Visualize distortion correction
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(dst)
# ax2.set_title('Undistorted Image', fontsize=30)
# plt.show()

"""
Step3: Threshold binary image
"""

# Choose an image from which to build and demonstrate each step of the pipeline
exampleImg = cv2.imread('./P4_AdvanceLindFind/test_images/test3.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
# plt.imshow(exampleImg)
# plt.show()
# print('...')

# 1. Undistort the image
# undistort image using camera calibration matrix from above
def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

exampleImg_undistort = undistort(exampleImg)
# Visualize undistortion
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(exampleImg_undistort)
# ax2.set_title('Undistorted Image', fontsize=30)
# plt.show()

# 2. Perspective Transform
def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# Visualize perspective transform on example image

h,w = exampleImg_undistort.shape[:2]

# define source and destination points for transform
src = np.float32([(600,465),
                  (735,465),
                  (280,685),
                  (1080,685)]) # order:left top, right top,left bottom, right bottom
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

exampleImg_unwarp, M, Minv = unwarp(exampleImg_undistort, src, dst)

# Visualize unwarp
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_undistort)
# x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
# y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
# ax1.plot(x, y, color='#33cc99', alpha=0.4, linewidth=4, solid_capstyle='round', zorder=2)
# ax1.set_ylim([h,0])
# ax1.set_xlim([0,w])
# ax1.set_title('Original Image', fontsize=30)
# ax1.axis('off')
# ax2.imshow(exampleImg_unwarp)
# ax2.set_title('Transformed Image Before Processing', fontsize=30)
# ax2.axis('off')
# plt.show()

# 3 Color spaces exploration
# (1) RGB, HSV, HLS
# Visualize multiple colorspace channels
exampleImg_unwarp_R = exampleImg_unwarp[:,:,0]
exampleImg_unwarp_G = exampleImg_unwarp[:,:,1]
exampleImg_unwarp_B = exampleImg_unwarp[:,:,2]

exampleImg_unwarp_HSV = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2HSV)

exampleImg_unwarp_H = exampleImg_unwarp_HSV[:,:,0]
exampleImg_unwarp_S = exampleImg_unwarp_HSV[:,:,1]
exampleImg_unwarp_V = exampleImg_unwarp_HSV[:,:,2]

exampleImg_unwarp_HLS = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2HLS)

exampleImg_unwarp_H1 = exampleImg_unwarp_HLS[:,:,0]
exampleImg_unwarp_L = exampleImg_unwarp_HLS[:,:,1]
exampleImg_unwarp_S1 = exampleImg_unwarp_HLS[:,:,2]

# fig, axs = plt.subplots(3,3, figsize=(16, 12))
# fig.subplots_adjust(hspace = .2, wspace=.001)
# axs = axs.ravel()
# axs[0].imshow(exampleImg_unwarp_R, cmap='gray')
# axs[0].set_title('RGB R-channel', fontsize=30)
# axs[1].imshow(exampleImg_unwarp_G, cmap='gray')
# axs[1].set_title('RGB G-Channel', fontsize=30)
# axs[2].imshow(exampleImg_unwarp_B, cmap='gray')
# axs[2].set_title('RGB B-channel', fontsize=30)
#
# axs[3].imshow(exampleImg_unwarp_H, cmap='gray')
# axs[3].set_title('HSV H-Channel', fontsize=30)
# axs[4].imshow(exampleImg_unwarp_S, cmap='gray')
# axs[4].set_title('HSV S-channel', fontsize=30)
# axs[5].imshow(exampleImg_unwarp_V, cmap='gray')
# axs[5].set_title('HSV V-Channel', fontsize=30)
#
# axs[6].imshow(exampleImg_unwarp_H1, cmap='gray')
# axs[6].set_title('HLS H-channel', fontsize=30)
# axs[7].imshow(exampleImg_unwarp_L, cmap='gray')
# axs[7].set_title('HLS L-Channel', fontsize=30)
# axs[8].imshow(exampleImg_unwarp_S1, cmap='gray')
# axs[8].set_title('HLS S-Channel', fontsize=30)
# plt.show()

# (2) Sobel Absolute Threshold
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    binary_output = np.zeros_like(scaled_sobel) # make sure the input and the output same size
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # plt.imshow(binary_output, cmap='gray')
    return binary_output

# Visualize Sobel absolute threshold on example image
exampleImg_sobelAbsx = abs_sobel_thresh(exampleImg_unwarp, 'x', thresh=(20,100))
exampleImg_sobelAbsy = abs_sobel_thresh(exampleImg_unwarp, 'y', thresh=(20,100))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_unwarp)
# ax1.set_title('Unwarped Image', fontsize=30)
# ax2.imshow(exampleImg_sobelAbs, cmap='gray')
# ax2.set_title('Sobel Absolute', fontsize=30)
# plt.show()

# (3) magnitude of gradient
def mag_thresh(img, sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel) # make sure the input and the output same size
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output
# Visualize Sobel absolute threshold on example image
exampleImg_sobelMag = mag_thresh(exampleImg_unwarp, sobel_kernel=3, thresh=(20,100))
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_unwarp)
# ax1.set_title('Unwarped Image', fontsize=30)
# ax2.imshow(exampleImg_sobelMag, cmap='gray')
# ax2.set_title('Sobel Magnitude', fontsize=30)
# plt.show()

# (4) direction of gradient
def dir_threshold(img, sobel_kernel=3, thresh=(0,np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(dir_grad) # make sure the input and the output same size
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return binary_output
# Visualize Sobel absolute threshold on example image
exampleImg_sobelDir = dir_threshold(exampleImg_unwarp, sobel_kernel=3, thresh=(0.5,1.5))
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_unwarp)
# ax1.set_title('Unwarped Image', fontsize=30)
# ax2.imshow(exampleImg_sobelDir, cmap='gray')
# ax2.set_title('Sobel Direction', fontsize=30)
# plt.show()

exampleImg_combined = np.zeros_like(exampleImg_sobelDir)
exampleImg_combined[((exampleImg_sobelAbsx == 1) & (exampleImg_sobelAbsy == 0))
             | ((exampleImg_sobelMag == 1) & (exampleImg_sobelDir == 0))] = 1
# Visualize the combined results
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_unwarp)
# ax1.set_title('Unwarped Image', fontsize=30)
# ax2.imshow(exampleImg_combined, cmap='gray')
# ax2.set_title('Magnitude&sobel', fontsize=30)
# plt.show()

# (5) Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    binary_img = np.zeros_like(s)
    binary_img[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_img

exampleImg_SThresh = hls_select(exampleImg_unwarp,thresh=(100, 255))
# Visualize hls s-channel threshold
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_unwarp)
# ax1.set_title('Unwarped Image', fontsize=30)
# ax2.imshow(exampleImg_SThresh, cmap='gray')
# ax2.set_title('HLS S-Channel', fontsize=30)
# plt.show()

# (6) Define a function that thresholds the L-channel of HLS
def hls_selectL(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    binary_img = np.zeros_like(s)
    binary_img[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_img

exampleImg_LThresh = hls_selectL(exampleImg_unwarp,thresh=(220, 255))
# Visualize hls s-channel threshold
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_unwarp)
# ax1.set_title('Unwarped Image', fontsize=30)
# ax2.imshow(exampleImg_SThresh, cmap='gray')
# ax2.set_title('HLS L-Channel', fontsize=30)
# plt.show()

# Define the complete image processing pipeline, reads raw image and returns binary image with lane lines identified

# (7) LAB color space

# Define a function that thresholds the B-channel of LAB
# Use exclusive lower bound (>) and inclusive upper (<=), OR the results of the thresholds (B channel should capture
# yellows)
def lab_bthresh(img, thresh=(190,255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

exampleImg_LBThresh = lab_bthresh(exampleImg_unwarp, thresh=(190, 255))
# Visualize LAB B threshold
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(exampleImg_unwarp)
# ax1.set_title('Unwarped Image', fontsize=30)
# ax2.imshow(exampleImg_LBThresh, cmap='gray')
# ax2.set_title('LAB B-channel', fontsize=30)
# plt.show()

# Define the complete image processing pipeline, reads raw image and returns binary image with lane lines identified


def pipeline(img):
    # Undistort
    img_undistort = undistort(img)

    # Perspective Transform
    img_unwarp, M, Minv = unwarp(img_undistort, src, dst)

    # Sobel Absolute (using default parameters)
    # img_sobelAbs = abs_sobel_thresh(img_unwarp, 'x', thresh=(10,80))

    # Sobel Magnitude (using default parameters)
    # img_sobelMag = mag_thresh(img_unwarp, sobel_kernel=3, thresh=(50,180))

    # Sobel Direction (using default parameters)
    # img_sobelDir = dir_threshold(img_unwarp, sobel_kernel=3, thresh=(0.5,1.5))

    # HLS S-channel Threshold (using default parameters)
    img_SThresh = hls_select(img_unwarp, thresh=(180, 210))

    # LAB B-channel Threshold (using default parameters)
    img_BThresh = lab_bthresh(img_unwarp, thresh=(210, 230))

    # LAB L-channel Threshold (using default parameters)
    img_LThresh = hls_selectL(img, thresh=(220, 255))


    # HLS S-channel Threshold and
    combined = np.zeros_like(img_SThresh)
    combined[(img_SThresh == 1) | (img_BThresh == 1) & (img_LThresh==1)] = 1

    return combined, Minv

# Make a list of example images
images = glob.glob('./P4_AdvanceLindFind/test_images/*.jpg')

# fig, axs = plt.subplots(len(images),2, figsize=(10, 20))
# fig.subplots_adjust(hspace = .2, wspace=.001)
# axs = axs.ravel()
#
# i = 0
# for image in images:
#     img = cv2.imread(image)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_bin, Minv = pipeline(img)
#     axs[i].imshow(img)
#     axs[i].axis('off')
#     i += 1
#     axs[i].imshow(img_bin, cmap='gray')
#     axs[i].axis('off')
#     i += 1
#
# plt.show()

"""
Step4: Sliding window polyfit
"""

# Define method to fit polynomial to binary image with lines extracted, using sliding window
def sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low,
                               win_xleft_high, win_xright_low, win_xright_high))

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data

"""
Visualize sliding window polyfit on example image
"""

binary_warped, Minv = pipeline(exampleImg)
left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(binary_warped)

h = binary_warped.shape[0]
rectangles = visualization_data[0]
histogram = visualization_data[1]

# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Draw the windows on the visualization image

for rect in rectangles:
# Draw the windows on the visualization image
    cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)
    cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2)

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()

# Print histogram from sliding window polyfit for example image
# plt.plot(histogram)
# plt.xlim(0, 1280)

"""
Polyfit Using Fit from Previous Frame
The test images test6, test4, and test5 appear, respectively, 
to be chronologically ordered frames from a single video capture
"""

# Define method to fit polynomial to binary image based upon a previous fit (chronologically speaking);
# this assumes that the fit will not change significantly from one video frame to the next
def polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy +
    left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) +
    left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy +
    right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) +
    right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds

"""
# visualize the result on example image
"""
exampleImg2 = cv2.imread('./P4_AdvanceLindFind/test_images/test5.jpg')
exampleImg2 = cv2.cvtColor(exampleImg2, cv2.COLOR_BGR2RGB)
exampleImg2_bin, Minv = pipeline(exampleImg2)
margin = 100

left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = polyfit_using_prev_fit(exampleImg2_bin,
                                                           left_fit, right_fit)
# Generate x and y values for plotting
ploty = np.linspace(0, exampleImg2_bin.shape[0]-1, exampleImg2_bin.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
left_fitx2 = left_fit2[0]*ploty**2 + left_fit2[1]*ploty + left_fit2[2]
right_fitx2 = right_fit2[0]*ploty**2 + right_fit2[1]*ploty + right_fit2[2]

# Create an image to draw on and an image to show the selection window
out_img = np.dstack((exampleImg2_bin, exampleImg2_bin, exampleImg2_bin))*255
window_img = np.zeros_like(out_img)

# Color in left and right line pixels
nonzero = exampleImg2_bin.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                              ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                              ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
# cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
# cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
# result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
# plt.imshow(result)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()


# Method to determine radius of curvature and distance from lane center
# based on binary image, polynomial fit, and L and R lane pixel indices
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                         / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
                         / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist

rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(binary_warped, left_fit, right_fit,
                                                       left_lane_inds, right_lane_inds)

print('Radius of curvature for example:', rad_l, 'm,', rad_r, 'm')
print('Distance from lane center for example:', d_center, 'm')


# Draw the Detected Lane Back onto the Original Image
def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h,w = binary_img.shape[0:2]
    ploty = np.linspace(0, h-1, h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result

exampleImg_out1 = draw_lane(exampleImg, binary_warped, left_fit, right_fit, Minv)
plt.imshow(exampleImg_out1)
# plt.show()

# Draw Curvature Radius and Distance from Center Data onto the Original Image
def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img

exampleImg_out2 = draw_data(exampleImg_out1, (rad_l+rad_r)/2, d_center)
plt.imshow(exampleImg_out2)
# plt.show()

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #number of detected pixels
        self.px_count = None
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)

# Define Complete Image Processing Pipeline
def process_image(img):
    new_img = np.copy(img)
    img_bin, Minv = pipeline(new_img)

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not l_line.detected or not r_line.detected:
        l_fit, r_fit, l_lane_inds, r_lane_inds, _ = sliding_window_polyfit(img_bin)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)

    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
    if l_fit is not None and r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img.shape[0]
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        x_int_diff = abs(r_fit_x_int-l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None

    l_line.add_fit(l_fit, l_lane_inds)
    r_line.add_fit(r_fit, r_lane_inds)

    # draw the current best fit if it exists
    if l_line.best_fit is not None and r_line.best_fit is not None:
        img_out1 = draw_lane(new_img, img_bin, l_line.best_fit, r_line.best_fit, Minv)
        rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                               l_lane_inds, r_lane_inds)
        img_out = draw_data(img_out1, (rad_l+rad_r)/2, d_center)
    else:
        img_out = new_img

    diagnostic_output = False
    if diagnostic_output:
        # put together multi-view output
        diag_img = np.zeros((720,1280,3), dtype=np.uint8)

        # original output (top left)
        diag_img[0:360,0:640,:] = cv2.resize(img_out,(640,360))

        # binary overhead view (top right)
        img_bin = np.dstack((img_bin*255, img_bin*255, img_bin*255))
        resized_img_bin = cv2.resize(img_bin,(640,360))
        diag_img[0:360,640:1280, :] = resized_img_bin

        # overhead with all fits added (bottom right)
        img_bin_fit = np.copy(img_bin)
        for i, fit in enumerate(l_line.current_fit):
            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (20*i+100,0,20*i+100))
        for i, fit in enumerate(r_line.current_fit):
            img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (0,20*i+100,20*i+100))
        img_bin_fit = plot_fit_onto_img(img_bin_fit, l_line.best_fit, (255,255,0))
        img_bin_fit = plot_fit_onto_img(img_bin_fit, r_line.best_fit, (255,255,0))
        diag_img[360:720,640:1280,:] = cv2.resize(img_bin_fit,(640,360))

        # diagnostic data (bottom left)
        color_ok = (200,255,155)
        color_bad = (255,155,155)
        font = cv2.FONT_HERSHEY_DUPLEX
        if l_fit is not None:
            text = 'This fit L: ' + ' {:0.6f}'.format(l_fit[0]) + \
                                    ' {:0.6f}'.format(l_fit[1]) + \
                                    ' {:0.6f}'.format(l_fit[2])
        else:
            text = 'This fit L: None'
        cv2.putText(diag_img, text, (40,380), font, .5, color_ok, 1, cv2.LINE_AA)
        if r_fit is not None:
            text = 'This fit R: ' + ' {:0.6f}'.format(r_fit[0]) + \
                                    ' {:0.6f}'.format(r_fit[1]) + \
                                    ' {:0.6f}'.format(r_fit[2])
        else:
            text = 'This fit R: None'
        cv2.putText(diag_img, text, (40,400), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Best fit L: ' + ' {:0.6f}'.format(l_line.best_fit[0]) + \
                                ' {:0.6f}'.format(l_line.best_fit[1]) + \
                                ' {:0.6f}'.format(l_line.best_fit[2])
        cv2.putText(diag_img, text, (40,440), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Best fit R: ' + ' {:0.6f}'.format(r_line.best_fit[0]) + \
                                ' {:0.6f}'.format(r_line.best_fit[1]) + \
                                ' {:0.6f}'.format(r_line.best_fit[2])
        cv2.putText(diag_img, text, (40,460), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Diffs L: ' + ' {:0.6f}'.format(l_line.diffs[0]) + \
                             ' {:0.6f}'.format(l_line.diffs[1]) + \
                             ' {:0.6f}'.format(l_line.diffs[2])
        if l_line.diffs[0] > 0.001 or \
           l_line.diffs[1] > 1.0 or \
           l_line.diffs[2] > 100.:
            diffs_color = color_bad
        else:
            diffs_color = color_ok
        cv2.putText(diag_img, text, (40,500), font, .5, diffs_color, 1, cv2.LINE_AA)
        text = 'Diffs R: ' + ' {:0.6f}'.format(r_line.diffs[0]) + \
                             ' {:0.6f}'.format(r_line.diffs[1]) + \
                             ' {:0.6f}'.format(r_line.diffs[2])
        if r_line.diffs[0] > 0.001 or \
           r_line.diffs[1] > 1.0 or \
           r_line.diffs[2] > 100.:
            diffs_color = color_bad
        else:
            diffs_color = color_ok
        cv2.putText(diag_img, text, (40,520), font, .5, diffs_color, 1, cv2.LINE_AA)
        text = 'Good fit count L:' + str(len(l_line.current_fit))
        cv2.putText(diag_img, text, (40,560), font, .5, color_ok, 1, cv2.LINE_AA)
        text = 'Good fit count R:' + str(len(r_line.current_fit))
        cv2.putText(diag_img, text, (40,580), font, .5, color_ok, 1, cv2.LINE_AA)

        img_out = diag_img
    return img_out

# Method for plotting a fit on a binary image - diagnostic purposes

def plot_fit_onto_img(img, fit, plot_color):
    if fit is None:
        return img
    new_img = np.copy(img)
    h = new_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(new_img, np.int32([pts]), isClosed=False, color=plot_color, thickness=8)
    return new_img

# Process Project Video
# l_line = Line()
# r_line = Line()
# #my_clip.write_gif('test.gif', fps=12)
# video_output1 = 'project_video_output.mp4'
# video_input1 = VideoFileClip('./P4_AdvanceLindFind/project_video.mp4')#.subclip(22,26)
# processed_video = video_input1.fl_image(process_image)
# processed_video.write_videofile(video_output1, audio=False)

# # Process Challenge Video
# l_line = Line()
# r_line = Line()
# video_output2 = 'challenge_video_output.mp4'
# video_input2 = VideoFileClip('./P4_AdvanceLindFind/challenge_video.mp4')#.subclip(10,12)
# #video_input2.save_frame("challenge01.jpeg", t=0.25) # saves the frame at time = 0.25s
# processed_video = video_input2.fl_image(process_image)
# processed_video.write_videofile(video_output2, audio=False)
#
# # Process Harder Challenge Video
# l_line = Line()
# r_line = Line()
# video_output3 = 'harder_challenge_video_output.mp4'
# video_input3 = VideoFileClip('./P4_AdvanceLindFind/harder_challenge_video.mp4')#.subclip(0,3)
# #video_input3.save_frame("hard_challenge01.jpeg") # saves the first frame
# processed_video = video_input3.fl_image(process_image)
# # %time processed_video.write_videofile(video_output3, audio=False)
