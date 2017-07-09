import cv2
import numpy as np


def gray(image, conv=cv2.COLOR_BGR2GRAY):
    """
    Converts a given image to gray scale
    :param image: Color image
    :param conv: Conversion parameter. Default is BGR2GRAY.
    :return: gray scale image
    """
    return cv2.cvtColor(image, conv)


def get_s(image):
    """
    Given an image, returns 'S' image from HLS color space
    :param image: Image in RGB format
    :return: 'S' image from HLS color space
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return hls[:, :, 2]


def absolute(image):
    """
    :param image: Any image
    :return: Absolute of that image
    """
    return np.absolute(image)


def sobel(gray_image, orient= 'x', kern_size=5):
    """
    :param gray_image: Image in gray scale format
    :param orient: The orientation in which sobel derivative neeeds to be taken. Accepted values are 'x' and 'y'
    :param kern_size: The kernel size that needs to be used while taking Sobel derivative
    :return: Sobel derivative image
    """
    if orient is 'x':
        sobelo = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kern_size)
    else:
        sobelo = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kern_size)

    return sobelo


def scale(image):
    """
    :param image: Any image
    :return: Scales the given image to be between 0-255 range for each pixel
    """
    return np.uint8(255 * image / np.max(image))


def binarize(image, thresh_min, thresh_max, hls=False):
    """
    Given an image, coverts the image to binary format. All pixel values in between the thresholds retain 1 and others 0
    :param image: Any image
    :param thresh_min: Minimum threshold to be applied on image
    :param thresh_max: Maximum threshold to be applied on image
    :param hls: if True, 'S' image is taken from HLS color space and a binary output is generated
    :return: Binary form of the image
    """
    if hls:
        image = get_s(image)
    binary_output = np.zeros_like(image)
    binary_output[(image >= thresh_min) & (image <= thresh_max)] = 1
    return binary_output


def direction_thresh(image, thresh_min, thresh_max):
    """
    Does a direction threshold of a given image. Does Sobel operation in both x and y directions and finds out the direction of edges
    :param image: Any image
    :param thresh_min: Minimum threshold to be applied on image
    :param thresh_max: Maximum threshold to be applied on image
    :return: Threshold applied image
    """
    sobelx = absolute(sobel(gray(image, cv2.COLOR_RGB2GRAY), 'x'))
    sobely = absolute(sobel(gray(image, cv2.COLOR_RGB2GRAY), 'y'))

    direc = np.arctan2(sobely, sobelx)
    return binarize(direc, thresh_min, thresh_max)


def mag_thresh(image, thresh_min, thresh_max):
    """
    Does a magnitude threshold of a given image. Does Sobel operation in both x and y directions and finds out the intensity of edge
    :param image: Any image
    :param thresh_min: Minimum threshold to be applied on image
    :param thresh_max: Maximum threshold to be applied on image
    :return: Threshold applied image
    """
    sobelx = sobel(gray(image, cv2.COLOR_RGB2GRAY), 'x')
    sobely = sobel(gray(image, cv2.COLOR_RGB2GRAY), 'y')
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = scale(abs_sobelxy)

    return binarize(scaled_sobel, thresh_min, thresh_max)


def get_binary(image, orient='x', thresh_min=0, thresh_max=255):
    """
    Helper function to return the scaled binary image
    :param image: Any image
    :param orient: Direction in which Sobel derivative needs to be taken
    :param thresh_min: Minimum threshold to be applied on image
    :param thresh_max: Maximum threshold to be applied on image
    :return: Scaled, Binary image with sobel derivative done in given orientation
    """
    return binarize(scale(absolute(sobel(gray(image, cv2.COLOR_RGB2GRAY), orient))), thresh_min, thresh_max)


def get_combined_binary(udi):
    """
    Does Sobel derivative in x, y directions along with magnitude and direction. Also does a 'S' image Sobel derivative.
    Joins all the derivatives to return a combined derivative within given thresholds
    :param udi: Undistorted image
    :return: Binary image with information from all Sobel derivatives
    """
    gbx = get_binary(udi, 'x', 10, 170)
    gby = get_binary(udi, 'y', 10, 170)
    s_binary = binarize(udi, 170, 255, hls=True)
    magt = mag_thresh(udi, 30, 170)
    dirt = direction_thresh(udi, 0.7, 1.3)

    combined_binary = np.zeros_like(gbx)
    combined_binary[(s_binary == 1) | ((gbx == 1) & (gby == 1)) | ((magt == 1) & (dirt == 1))] = 1

    return combined_binary


def undistort_and_warp(image, mtx, dist):
    udi = cv2.undistort(image, mtx, dist)
    return warp(get_combined_binary(udi))


def warp(image):
    """
    Deos a perspective transform on a given image
    :param image: Any image
    :return: Warped image along with transformation matrix and its inverse.
    """
    row, col = image.shape[:2]
    x1 = 220
    x2 = 590
    y1 = 450

    src = np.float32(
        [[x1, row],
         [col - x1, row],
         [x2, y1],
         [col - x2, y1]])

    offset = 200

    dst = np.float32([
        [offset, row],
        [col - offset, row],
        [offset, 0],
        [col - offset, 0]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return warped, M, M_inv
