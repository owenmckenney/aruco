import cv2
import numpy as np
import glob

# Define chessboard size (number of inner corners per row and column)
chessboard_size = (9, 6)  # Change to match your printed pattern (e.g., 9x6)
square_size = 0.025  # Size of a square in meters (or any consistent unit)

# Prepare object points (3D points in the world)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in the world
imgpoints = []  # 2D points in the image

# Load all images for calibration
images = glob.glob("/home/jrichard/Desktop/owen_stereo_calibration/cam0/*.jpg")  # Change the path as needed

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners for visual confirmation
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Chessboard", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the results
if ret:
    print("Camera matrix (Intrinsic parameters):")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
else:
    print("Calibration failed. Ensure the chessboard pattern is clear in the images.")

# Save the calibration results
np.savez("calibration_results.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# Optional: Reprojection error to check calibration quality
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print(f"\nReprojection error: {total_error / len(objpoints):.4f}")