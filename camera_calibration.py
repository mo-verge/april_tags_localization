# import os

# import numpy as np
# import cv2
# import glob
# import copy
# import argparse


# def load_images(filenames):
#     filelist = []
#     for file in filenames:
#         filelist.append(cv2.imread(file))
#     return filelist, filenames


# def get_chessboard_points(chessboard_shape, dx, dy):
#     N = chessboard_shape[0] * chessboard_shape[1]
#     matrix = np.zeros((N, 3))
#     x_value = 0
#     y_value = 0
#     for c in range(chessboard_shape[1]):
#         for r in range(chessboard_shape[0]):
#             matrix[c * chessboard_shape[0] + r] = (x_value, y_value, 0)
#             y_value += dy
#         x_value += dx
#         y_value = 0

#     return matrix


# ap = argparse.ArgumentParser()
# ap.add_argument("--folder", required=True, help="path to where the calibration images are located")
# ap.add_argument("--psize", required=True, type=float, help="size of the chessboard squares")
# args = vars(ap.parse_args())

# folder = args['folder']
# path = os.path.join(folder, '*')
# row = 19
# col = 12
# psize = args['psize']

# imgs, names = load_images(sorted(glob.glob(path)))

# corners = [cv2.findChessboardCorners(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY), (row, col)) for image in imgs]
# # cornerSubPix is destructive. so we copy standard corners and use the new list to refine
# corners2 = copy.deepcopy(corners)

# # Refine corner estimation (images mus be in b&w, use cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) to convert from rgb)
# # termination criteria (see, e.g https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# cornersRefined = [(corners2[i][0], cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), corners2[i][1], (11, 11),
#                                                     (-1, -1), criteria)) for i, img in enumerate(imgs)]
# # for img, c in zip (imgs,cornersRefined):
# #     cv2.drawChessboardCorners(img, (19,12), c[1], True)
# #     cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
# #     cv2.resizeWindow("Resized_Window", 1920, 1080) 
# #     cv2.imshow('Resized_Window', img)
# #     cv2.waitKey(0)
# cb_points = get_chessboard_points((row, col), psize, psize)
# h, w = imgs[0].shape[0:2]
# # Extract the list of valid images with all corners
# valid_corners = [(imgs[i], cornersRefined[i][1]) for i in range(len(cornersRefined)) if cornersRefined[i][0]]
# num_valid_images = len(valid_corners)
# print (num_valid_images, len(imgs))
# # Prepare input data
# # object_points: numpy array with dimensions (number_of_images, number_of_points, 3)
# object_points = np.zeros((num_valid_images, row*col, 3), dtype=np.float32)
# for i in range(num_valid_images):
#     object_points[i] = cb_points

# # image_points: numpy array with dimensions (number_of_images, number_of_points, 2)
# image_points = np.zeros((num_valid_images, row*col, 2), dtype=np.float32)
# for i in range(num_valid_images):
#     image_points[i] = np.reshape(valid_corners[i][1], (row*col, 2))
# # Calibrate for square pixels corners standard
# _, intrinsics, dist_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, (h, w), None, None)

# print (intrinsics)
# print (dist_coeffs)
# np.savez('calibration', intrinsics=intrinsics, dist_coeffs=dist_coeffs)


import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((12*19,3), np.float32)
objp[:,:2] = np.mgrid[0:240:20,0:380:20].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibs/*.bmp')

for fname in images:
    print (fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (12,19), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (19,12), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print (mtx, dist)
np.savez('calibration', intrinsics=mtx, dist_coeffs=dist)


cv.destroyAllWindows()
