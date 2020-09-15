import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read and display the original image
img = mpimg.imread('/road_image_viedo/stopsign.jpg')
plt.imshow(img)

# Source image points

plt.imshow(img)
plt.plot(850, 320, '.') # top right
plt.plot(865, 450, '.') # bottom right
plt.plot(533, 350, '.') # bottom left
plt.plot(535, 210, '.') # top left

# Define perspective transform function
def warp(img):
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])

    # Four source coordinate
    src = np.float32(
        [[850, 320],
         [865, 450],
         [533, 350],
         [535, 210]])

    dst = np.float32(
        [[870, 240],
         [870, 370],
         [520, 370],
         [520, 240]])

    # Compute the perspective transform
    M = cv2.getPerspectiveTransform(src, dst)

    # Compute the inverse also by swapping the input parameters
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

# Get perspective transform
warped_img = warp(img)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.set_title('Source Image', fontsize=30)
ax1.imshow(img)
ax2.set_title('Warped Image', fontsize=30)
ax2.imshow(warped_img)
# plt.show()
