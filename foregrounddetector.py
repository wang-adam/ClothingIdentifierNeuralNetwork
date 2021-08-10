# Code adapted from https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/

import numpy as np
import cv2
from matplotlib import pyplot as plt


def remove_background(file):

    # reads and loads the image
    image = cv2.imread(file)
    height = image.shape[0]
    width = image.shape[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # create a simple mask image similar
    # to the loaded image, with the
    # shape and return type
    mask = np.zeros(image.shape[:2], np.uint8)

    # specify the background and foreground model
    # using numpy the array is constructed of 1 row
    # and 65 columns, and all array elements are 0
    # Data type for the array is np.float64 (default)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    # USER MUST MANUALLY SPECIFY REGION OF INTEREST
    # where the values are entered as
    # (startingPoint_x, startingPoint_y, width, height)
    margin = 0.08
    rectangle = (int(margin * width), int(margin * height),
                 int((1-2*margin)*width), int((1-2*margin)*height))

    # apply the grabcut algorithm with appropriate
    # values as parameters, number of iterations
    # cv2.GC_INIT_WITH_RECT is used because
    # of the rectangle mode is used
    cv2.grabCut(image, mask, rectangle,
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_RECT)

    # In the new mask image, pixels will
    # be marked with four flags
    # four flags denote the background / foreground
    # mask is changed, all the 0 and 2 pixels
    # are converted to the background
    # mask is changed, all the 1 and 3 pixels
    # are now the part of the foreground
    # the return type is also mentioned,
    # this gives us the final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # The final mask is multiplied with
    # the input image to give the segmented image.
    newimage = image * mask2[:, :, np.newaxis]
    # plt.imshow(image)
    # plt.colorbar()
    # plt.show()
    return newimage, image


# remove_background("shirt.jpg")
