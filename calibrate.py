import cv2
import pickle
import os.path
import numpy as np
import matplotlib.image as mpimg
import glob

from binarize import gray

nx = 9
ny = 5
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


def calibrate_camera():
    """
    Does camera calibration using OpenCV functions to find ChessboardCorners. Since it is computationally intensive
    and does not change for a given camera, it is only done once and the values are saved as camera.p file. If present,
    the values from it are returned, if not, are computed.
    :return: Camera Matrix, Distortion co-efficients, Rotation and Translation Vectors along with return value indicating
    if the operation was successful or not
    """
    fileName = './camera.p'
    if os.path.exists(fileName):
        fileObject = open(fileName, 'rb')
        pf = pickle.load(fileObject)
        return pf['ret'], pf['mtx'], pf['dist'], pf['rvecs'], pf['tvecs']
    images = glob.glob('camera_cal/calibration*.jpg')
    imgpoints = []
    objpoints = []

    for fname in images:
        img = mpimg.imread(fname)
        gray_image = gray(img)
        ret, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

    pf = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
    fileName = 'camera.p'
    fileObject = open(fileName, 'wb')
    pickle.dump(pf, fileObject)
    fileObject.close()

    return ret, mtx, dist, rvecs, tvecs
