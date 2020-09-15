import cv2
import numpy as np
import matplotlib.pyplot as plt

carDetection = cv2.imread('carDetection.jpg')

# Define a function that takes an image, a list of bounding boxes,
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output
def draw_box(img, bound_box, color=(255, 0, 0), thick=6):
    img = np.copy(img)
    for box in bound_box:
        img = cv2.rectangle(img, box[0], box[1], color, thick)
        cv2.imshow('img', img)
        cv2.waitKey(50000)
        plt.show()
    return img

bound_box = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)),
          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]

draw_box(carDetection, bound_box)

