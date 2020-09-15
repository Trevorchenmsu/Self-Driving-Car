import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( './CameraCali/calibration_wide/wide_dist_pickle.p', "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


# Read in an image
img = cv2.imread('./CameraCali/calibration_wide/test_image2.jpg')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):

    undistort = cv2.undistort(img, mtx, dist,None, mtx)
    gray = cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # print(corners)
    if ret == True:
        corner_img = cv2.drawChessboardCorners(undistort, (nx, ny), corners, ret)
        offset  = 100
        img_size = (gray.shape[1], gray.shape[0])

        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(corner_img, M, gray.shape)
        # cv2.imshow('img', undistort)
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
