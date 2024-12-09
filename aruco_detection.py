import cv2
from picamera2 import Picamera2, Preview    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from mantabot_control import Path    

# class for handling arUco target detection & pose/coordinate extraction
class TargetDetection:
    def __init__(self, marker_size, camera_matrix, distortion_coefficients):
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.marker_detected = False

        self.cam_x, self.cam_y, self.cam_z = 0, 0, 0
        self.prev_cam_x, self.prev_cam_y, self.prev_cam_z = 0, 0, 0

        # define aruco dictionary type, create detector
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
        parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    def ema_filter(alpha, p, prev_p):
        return alpha * p + (1 - alpha) * prev_p # exponential moving average 
    
    def detect_aruco(self, image):
        image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect marker
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        # runs only if aruco is detected
        if ids is not None:

            # estimate pose of the marker. rvecs is the rotation vector, tvecs is the translation vector (x, y, z) coordinates
            self.rvecs, self.tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, camera_matrix, distortion_coefficients)

            # code set up to support multiple aruco targets. currently only using with one target
            for i, (rvec, tvec) in enumerate(zip(self.rvecs, self.tvecs)):
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                cv2.drawFrameAxes(image, camera_matrix, distortion_coefficients, rvec, tvec, self.marker_size / 2)
                
            return self.rvecs, self.tvecs

        return None
    
    def extract_coordinates(self, filtered=False):
        rvec = np.array(self.rvecs).reshape((3, 1))
        tvec = np.array(self.tvecs).reshape((3, 1))

        # get camera position from dot of rvec and tvec
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_pos = -np.dot(rotation_matrix.T, tvec)

        # extract x, y, z, calculate distance
        raw_x, raw_y, raw_z = camera_pos.flatten()

        if filtered:
            # filter camera coordinates to reduce variation, alpha = 0.3
            self.cam_x = self.ema_filter(0.3, raw_x, self.prev_cam_x)
            self.cam_y = self.ema_filter(0.3, raw_y, self.prev_cam_y)
            self.cam_z = self.ema_filter(0.3, raw_z, self.prev_cam_z)
            self.prev_cam_x = self.cam_x                                          
            self.prev_cam_y = self.cam_y
            self.prev_cam_z = self.cam_z

            return self.cam_x, self.cam_y, self.cam_z
    
        else:  
            self.prev_cam_x = raw_x                                        
            self.prev_cam_y = raw_y
            self.prev_cam_z = raw_z

            return raw_x, raw_y, raw_z

    def distance_to_target(self):
        return np.sqrt(self.cam_x**2 + self.cam_y**2 + self.cam_z**2)


if __name__ == "__main__":
    # camera matrices
    camera_matrix = np.array([[577.1182, 0, 292.4001], [0, 575.9194, 249.8583], [0, 0, 1]])
    distortion_coefficients = np.array([[-0.1938, -0.3035, -0.0015, -0.0003, 0.07342]])

    picam0 = Picamera2(0)
    config = picam0.create_video_configuration(
        {"size": (640, 480), "format": "RGB888"} # lowers resolution an makes cam config video which allows for much faster capture
    )
    picam0.configure(config)
    picam0.start()

    # path and target object creation
    path = Path()
    target_detector = TargetDetection(200, camera_matrix, distortion_coefficients)

    marker_detected = False
    path_generated = False
    start_detection = False

    # matplotlib visualization
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    ax1.scatter(0, 0, 0, color='black', s=50, label="Target")
    ax1.legend()
    ax1.cla()

    while True:
        image = picam0.capture_array("main")
        aruco = target_detector.detect_aruco(image)

        if aruco is not None and not start_detection and not path_generated:
            sdt = time.time()
            start_detection = True
        else:
            start_detection = False

        vehicle_x, vehicle_y, vehicle_z = target_detector.extract_coordinates(filtered=True)

        if time.time() - sdt > 1.0 and not path_generated:
            path_x, path_y, path_z = path.generate_path(vehicle_x, vehicle_y, vehicle_z)
            path_generated = True

        # if not path_generated :
        #     vehicle_x, vehicle_y, vehicle_z = target_detector.extract_coordinates(filtered=False)
        #     path_x, path_y, path_z = path.generate_path(vehicle_x, vehicle_y, vehicle_z)
        #     path_generated = True
        # else:
            # vehicle_x, vehicle_y, vehicle_z = target_detector.extract_coordinates(filtered=True)

        if path_generated:
            closest_x, closest_y, closest_z = path.closest_point(vehicle_x, vehicle_y, vehicle_z)

            ax1.plot(path_x, path_y, path_z, color='blue')
            ax1.scatter(closest_x, closest_y, closest_z, color='green', s=50, label="Closest Path Point")

        # matplotlib visualization
        ax1.scatter(0, 0, 0, color='black', s=50, label="Target")
        ax1.scatter(vehicle_x, vehicle_y, vehicle_z, color='red', s=50, label="Camera")
        ax1.text(vehicle_x - 50, vehicle_y, vehicle_z + 20, f"{vehicle_x:.2f}, {vehicle_y:.2f}, {vehicle_z:.2f}")
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



# load ArUco library, set dictionary type (using 7x7)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
# parameters = cv2.aruco.DetectorParameters()
# aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# marker_size = 200 # in mm

# camera constants gathered from single_camera_calibration.py
# camera_matrix = np.array([[577.1182, 0, 292.4001], [0, 575.9194, 249.8583], [0, 0, 1]])
# distortion_coefficients = np.array([[-0.1938, -0.3035, -0.0015, -0.0003, 0.07342]])

# picam0 = Picamera2(0)
# config = picam0.create_video_configuration(
#     {"size": (640, 480), "format": "RGB888"} # lowers resolution an makes cam config video which allows for much faster capture
# )
# picam0.configure(config)
# picam0.start()

# for _ in range(5):
#     picam0.capture_array("main")

# st = time.time()
# image = picam0.capture_array("main")
# et = time.time()
# print(et - st)

# cam_x, cam_y, cam_z = 0, 0, 0
# prev_cam_x, prev_cam_y, prev_cam_z = 0, 0, 0
# marker_detected = False

# set up path generation
# path = Path()
# closest_point = [0, 0, 0]

# # initialize matplotlib figure
# plt.ion()
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122)
# ax1.scatter(0, 0, 0, color='black', s=50, label="Target")
# ax1.legend()

# while True:
#     image = picam0.capture_array("main") # captures image directly as an array. doesn't save locally

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # detect marker
#     corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
#     # runs only if aruco is detected
#     if ids is not None:

#         # estimate pose of the marker. rvecs is the rotation vector, tvecs is the translation vector (x, y, z) coordinates
#         rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion_coefficients)

#         for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
#             cv2.aruco.drawDetectedMarkers(image, corners, ids)
            
#             cv2.drawFrameAxes(image, camera_matrix, distortion_coefficients, rvec, tvec, marker_size / 2)
            
#             rvec = np.array(rvec).reshape((3, 1))
#             tvec = np.array(tvec).reshape((3, 1))

#             # get camera position from dot of rvec and tvec
#             rotation_matrix, _ = cv2.Rodrigues(rvec)
#             camera_pos = -np.dot(rotation_matrix.T, tvec)

#             # extract x, y, z, calculate distance
#             raw_x, raw_y, raw_z = camera_pos.flatten()

#             if not marker_detected:
#                 # path = generate_curve(raw_x, raw_y, raw_z)
#                 path_func = path.generate_path(raw_x, raw_y, raw_z)
#                 # print(path_func)
#                 marker_detected = True
#             else:
#                 # filter camera coordinates to reduce variation
#                 cam_x = ema_filter(0.3, raw_x, prev_cam_x)
#                 cam_y = ema_filter(0.3, raw_y, prev_cam_y)
#                 cam_z = ema_filter(0.3, raw_z, prev_cam_z)
#                 prev_cam_x = cam_x                                          
#                 prev_cam_y = cam_y
#                 prev_cam_z = cam_z

#                 distance = np.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
#                 closest_point = path.closest_point(cam_x, cam_y, cam_z)
#                 print(f"Camera Position: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) mm, Distance: {distance:.2f} mm, Closest point on Path: {closest_point} mm")


#     # update matplotlib
#     ax1.cla()
#     ax1.scatter(0, 0, 0, color='black', s=50, label="Target")
#     ax1.scatter(cam_x, cam_y, cam_z, color='red', s=50, label="Camera")
#     ax1.text(cam_x - 50, cam_y, cam_z + 20, f"{cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}")

#     if marker_detected:
#         ax1.plot(path_func[0], path_func[1], path_func[2], color='blue')
#         ax1.scatter(closest_point[0], closest_point[1], closest_point[2], color='green', s=50, label="Closest")

#     ax1.set_xlim(-2000, 2000)
#     ax1.set_ylim(-2000, 2000)
#     ax1.set_zlim(0, 1000)
#     ax1.set_box_aspect([1, 1, 0.5])

#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_zlabel('z')

#     ax2.cla()            
#     ax2.imshow(np.fliplr(np.flipud(image)))
#     ax2.axis('off')

#     plt.pause(0.01)

    # time.sleep(0.05)
