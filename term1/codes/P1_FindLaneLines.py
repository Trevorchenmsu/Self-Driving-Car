import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from P1_HelperFunC import *

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import display, HTML


# Set up the image files path
os.listdir("P1_FindLaneLines/")

kernel_size = 3  # Gaussian Smoothing size
low_threshold = 75  # threshold for canny edge detector
high_threshold = 150    # threshold for canny edge detect

bottom_shift = 75 # use for creating area of interest region
top_shift = 40 # use for creating area of interest region

rho = 2  # distance resolution in pixels of the Hough grid
theta = np.pi/180   # angular resolution in radians of the Hough grid
threshold = 100   # minimum number of votes (intersections in Hough grid cell)
min_line_len = 25  # minimum number of pixels making up a line
max_line_gap = 25  # maximum gap in pixels between connectable line segments

# Building the image processing pipeline
def build_pipeline(image):
    # Gray scaling: convert RGB image to gray scale image.
    gray = grayscale(image)

    # Gaussian Smoothing: smooth / blurring the gray scale image.
    blur_gray = gaussian_blur(gray, kernel_size)

    # Canny Edge Detection
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Define the height and the width of the images
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Define a polygon vertices array
    vertices = np.array([[(bottom_shift, image_height),
                        (image_width / 2 - top_shift, image_height / 2 + top_shift),
                        (image_width /2 + top_shift, image_height / 2 + top_shift),
                        (image_width - bottom_shift, image_height)]], dtype=np.int32)

    # Region Selection: define a four sided polygon to mask.
    region_selection = region_of_interest(edges, vertices)

    # Hough Transform Line Detection
    hough_line = hough_lines(region_selection, rho, theta, threshold,
                                       min_line_len, max_line_gap)

    # Draw the lines on the edge image.
    return weighted_img(hough_line, image)

for test_image in os.listdir("P1_FindLaneLines/test_images/"):
    image = mpimg.imread('P1_FindLaneLines/test_images/' + test_image)
    plt.imshow(build_pipeline(image))
    # x = [bottom_shift, 480 - top_shift, 480 + top_shift, 960 - bottom_shift]
    # y = [539, 270 + top_shift, 270 + top_shift, 539]
    # plt.plot(x,y,'b--', lw=4)
    # plt.show()


def process_image(image):
    return build_pipeline(image)

def processVideo(videoFileName, inputVideoDir, outputVideoDir):
    """
    Applys the process_image pipeline to the video `videoFileName` on the directory `inputVideoDir`.
    The video is displayed and also saved with the same name on the directory `outputVideoDir`.
    """
    if not os.path.exists(outputVideoDir):
        os.makedirs(outputVideoDir)
    clip = VideoFileClip(inputVideoDir + '/' + videoFileName)
    outputClip = clip.fl_image(process_image)
    outVideoFile = outputVideoDir + '/' + videoFileName
    outputClip.write_videofile(outVideoFile, audio=False)
    display(
        HTML("""
        <video width="960" height="540" controls>
          <source src="{0}">
        </video>
        """.format(outVideoFile))
    )

testVideosOutputDir = 'P1_FindLaneLines/test_videos_output'
testVideoInputDir = 'P1_FindLaneLines/test_videos'
processVideo('solidWhiteRight.mp4', testVideoInputDir, testVideosOutputDir)
processVideo('solidYellowLeft.mp4', testVideoInputDir, testVideosOutputDir)
processVideo('challenge.mp4', testVideoInputDir, testVideosOutputDir)
