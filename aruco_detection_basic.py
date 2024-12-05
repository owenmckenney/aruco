import cv2
from picamera2 import Picamera2, Preview    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# exponential moving average filter
def ema_filter(alpha, p, prev_p):
    return alpha * p + (1 - alpha) * prev_p

# load ArUco library, set dictionary type (using 7x7)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

marker_size = 200 # in mm

# camera constants gathered from single_camera_calibration.py
camera_matrix = np.array([[577.1182, 0, 292.4001], [0, 575.9194, 249.8583], [0, 0, 1]])
distortion_coefficients = np.array([[-0.1938, -0.3035, -0.0015, -0.0003, 0.07342]])

picam0 = Picamera2(0)
config = picam0.create_video_configuration(
    {"size": (640, 480), "format": "RGB888"} # lowers resolution an makes cam config video which allows for much faster capture
)
picam0.configure(config)
picam0.start()

# first images captured by camera take longer. take 5 before timing
for _ in range(5):
    picam0.capture_array("main")

st = time.time()
image = picam0.capture_array("main")
et = time.time()
print(et - st)

cam_x, cam_y, cam_z = 0, 0, 0
prev_cam_x, prev_cam_y, prev_cam_z = 0, 0, 0

# initialize matplotlib figure
plt.ion()
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
ax1.scatter(0, 0, 0, color='black', s=50, label="Target")
ax1.legend()

while True:
    image = picam0.capture_array("main") # captures image directly as an array. doesn't save locally

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect marker
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    # runs only if aruco is detected
    if ids is not None:

        # estimate pose of the marker. rvecs is the rotation vector, tvecs is the translation vector
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion_coefficients)

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            
            cv2.drawFrameAxes(image, camera_matrix, distortion_coefficients, rvec, tvec, marker_size / 2)
            
            rvec = np.array(rvec).reshape((3, 1))
            tvec = np.array(tvec).reshape((3, 1))

            # get camera position from dot of rvec and tvec
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            camera_pos = -np.dot(rotation_matrix.T, tvec)

            # extract x, y, z, calculate distance
            raw_x, raw_y, raw_z = camera_pos.flatten()

            # filter camera coordinates to reduce variation
            cam_x = ema_filter(0.3, raw_x, prev_cam_x)
            cam_y = ema_filter(0.3, raw_y, prev_cam_y)
            cam_z = ema_filter(0.3, raw_z, prev_cam_z)
            prev_cam_x = cam_x
            prev_cam_y = cam_y
            prev_cam_z = cam_z

            distance = np.sqrt(cam_x**2 + cam_y**2 + cam_z**2)

            print(f"Camera Position: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) mm, Distance: {distance:.2f} mm")


    # update matplotlib
    ax1.cla()
    ax1.scatter(0, 0, 0, color='black', s=50, label="Target")
    ax1.scatter(cam_x, cam_y, cam_z, color='red', s=50, label="Camera")
    ax1.text(cam_x - 50, cam_y, cam_z + 20, f"{cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}")

    ax1.set_xlim(-2000, 2000)
    ax1.set_ylim(-2000, 2000)
    ax1.set_zlim(0, 1000)
    ax1.set_box_aspect([1, 1, 0.5])

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax2.cla()            
    ax2.imshow(np.fliplr(np.flipud(image)))
    ax2.axis('off')

    plt.pause(0.01)

    time.sleep(0.05)
