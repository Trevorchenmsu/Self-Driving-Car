import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

templates = glob.glob('cutouts/cutout*.jpg')
cutouts = []

img_detected = cv2.imread('cutouts/bbox-example-image.jpg')

for template in templates:
    cutouts.append(template)

# Define a function to search for template matches
# and return a list of bounding boxes
def find_matches(img, template_list):
    # Define an empty list to take bound_box coordinates.
    bound_boxes = []
    # Define matching method
    methods = [cv2.TM_CCORR, cv2.TM_SQDIFF]
    # Iterate through template list
    for template in template_list:
        # Read in templates one by one
        template = cv2.imread(template)
        # Use cv2.matchTemplate() to search the image
        res = cv2.matchTemplate(img,template,methods[1])
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # Determine a bounding box for the match
        h, w = template.shape[0:2]
        top_left = min_loc  # minimum location for square difference
        # top_left = max_loc # maximum location for correlation
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bound_boxes.append((top_left, bottom_right))
        img2 = cv2.rectangle(img,top_left, bottom_right, 255, 2)
        cv2.imshow('img2', img2)
        cv2.waitKey(50000)
        plt.show()
    # Return the list of bounding boxes
    return bound_boxes

find_matches(img_detected, cutouts)
