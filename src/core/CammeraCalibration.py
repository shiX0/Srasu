import numpy as np
import cv2
import os

def calibrate_camera(image_dir, chessboard_size, square_size):
    """
    Calibrates the camera using chessboard images of a known size.
    for eg. calibrate_camera('calibration_images', (9, 6), 0.025)

    arguments:
        image_dir: Directory containing calibration images.
        chessboard_size: Number of internal corners in the chessboard pattern (rows, columns).
        square_size: Size of a square in the chessboard pattern in your desired unit (e.g., meters).
    Returns:
        ret: Reprojection error.
        mtx: Camera matrix.
        dist: Distortion.
        rvecs: Rotation vectors.
        tvecs: Translation vectors.
    """
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Load all images from the directory
    images = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Print calibration results
    print("Camera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    return ret, mtx, dist, rvecs, tvecs