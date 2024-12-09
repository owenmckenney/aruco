import math
from picamera2 import Picamera2, Preview    
import numpy as np
import matplotlib.pyplot as plt
from aruco_detection import TargetDetection
import time

class MantaBot:
    def __init__(self, x, y, z, yaw, vel=0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.vel = vel

    def update_position(self, accel, delta):
        pass


class Path:
    def __init__(self, lookahead):
        self.lookahead = lookahead
        self.xt = 0
        self.yt = 0
        self.zt = 0
        self.last_idx = 0

    def generate_path(self, xi, yi, zi):
        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.t = np.linspace(0, 1, 100) # 100 points t: 0 -> 1

        # quadratic from camera -> target
        self.m = 5
        self.func_x = self.xt + self.t * (self.xi - self.xt)
        self.func_y = self.yt + self.t * (self.yi - self.yt)
        self.func_z = (self.zi - self.zt - self.m) * self.t ** 2 + self.m * self.t + self.zt
 
        return self.func_x, self.func_y ,self.func_z
    
    def get_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    
    def get_point(self, idx):
        return [self.func_x[idx], self.func_y[idx], self.func_z[idx]]
        
    # returns the closest point on the path in relation to the input coordinates and the scalar distance to that point
    def closest_point(self, vx, vy, vz):
        distances = np.sqrt((self.func_x - vx) ** 2 + (self.func_y - vy) ** 2 + (self.func_z - vz) ** 2)
        min_dist_idx = np.argmin(distances)
        closest_point = self.get_point(min_dist_idx)

        return closest_point

    def get_target_point(self, vx, vy, vz):
        target_idx = self.last_idx
        target_point = self.get_point(target_idx)

        current_distance = self.get_distance([vx, vy, vz], target_point)

        while current_distance < self.lookahead and target_idx < len(self.func_x) - 1:
            target_idx += 1
            target_point = self.get_point(target_idx)
            current_distance = self.get_distance([vx, vy, vz], target_point)
        
        self.last_idx = target_idx
        return self.get_point(target_idx)


class PID:
    def __init__(self, dt=0.1, kp=1.0, ki=0.1, kd=0.0):
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.proportional = 0
        self.integral = 0
        self.derivative = 0

    def control(self, error):
        self.proportional = self.kp * error
        self.integral += error * self.dt
        self.derivative = error / self.dt

        output = self.proportional + self.ki * self.integral + self.kd * self.derivative
        return output
    

if __name__ == "__main__":
    frequency = 20 # Hz

    # camera setup
    camera_matrix = np.array([[577.1182, 0, 292.4001], [0, 575.9194, 249.8583], [0, 0, 1]])
    distortion_coefficients = np.array([[-0.1938, -0.3035, -0.0015, -0.0003, 0.07342]])
    picam0 = Picamera2(0)
    config = picam0.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"} # lowers resolution an makes cam config video which allows for much faster capture
    )
    picam0.configure(config)
    picam0.start()

    # path and target object creation & control variables
    path = Path()
    target_detector = TargetDetection(200, camera_matrix, distortion_coefficients)
    marker_detected = False
    path_generated = False
    start_detection = False

    # pure pursuit & pid setup
    lookahead = 5
    dt = 0.1
    accel_pid = PID(dt)
    yaw_pid = PID(dt)

    # main control loop
    while True:
        image = picam0.capture_array("main")
        aruco = target_detector.detect_aruco(image)



        time.sleep(frequency)
        