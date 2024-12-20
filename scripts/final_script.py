import os
import signal
import numpy as np
from threading import Thread
import time
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.products.qcar import QCarEKF
from hal.products.mats import SDCSRoadMap
from pal.products.traffic_light import TrafficLight

# ================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 200
startDelay = 1
controllerUpdateRate = 100

# ===== Speed Controller Parameters
# - v_cruise: desired cruise velocity in m/s
# - K_p: proportional gain for speed controller
# - K_i: integral gain for speed controller
v_cruise = 0.5
K_p = 0.1
K_i = 1

# ===== Steering Controller Parameters
# - enableSteeringControl: whether or not to enable steering control
# - K_stanley: K gain for stanley controller
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableSteeringControl = True
K_stanley = 1
nodeSequence = [11, 2, 4, 20, 10, 2, 4, 20, 10]

# Define the calibration pose
# Calibration pose is either [0,0,-pi/2] or [-2,0,-pi/2]

# Comment out the one that is not used:

# Calibration Pose 1 (node 0) center of the small SDCS map
# calibrationPose = [0,0,-np.pi/2]
# Calibration Pose 2 (node 11) center of the larger SDCS map
calibrationPose = [0, 2, -np.pi / 2]

# Stop Configuration
deceleration_rate = 0.1  # Rate of deceleration (m/s^2)
acceleration_rate = 0.1  # Rate of acceleration (m/s^2)
stop_tolerance = 0.2  # Stop tolerance (meters)
stop_distance_offset = 0.5  # Distance to stop before the target
# List of Coordinates to stop
target_coordinates_list = [[2.38, 0.97], [-2.11, 1.13]]

# Traffic light IP address
TRAFFIC_LIGHT_IP = "192.168.2.12"
traffic_light = TrafficLight(TRAFFIC_LIGHT_IP)

# Setting the Traffic light sequence
traffic_light.timed(red=15, yellow=1, green=4)

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False


def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)


class SpeedController:

    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki

        self.ei = 0

    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e

        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )


class SteeringController:

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

    # ==============  SECTION B -  Steering Control  ====================
    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p - wp_1, v_uv)

        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)

        self.p_ref = ep
        self.th_ref = tangent

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


def calculate_adjusted_target(
    current_position, target_coordinates, stop_distance_offset
):
    """
    Calculate adjusted target coordinates to stop stop_distance_offset meters before the target.

    Parameters:
    - current_position: tuple (x, y) representing the current vehicle position
    - target_coordinates: list [x, y] representing target coordinates
    - stop_distance_offset: float indicating the distance to stop before the target

    Returns:
    - adjusted_target: list [x, y] representing the adjusted target coordinates
    """
    direction_vector = np.array(target_coordinates) - np.array(
        current_position
    )  # calculates the vector point
    distance_to_target = np.linalg.norm(direction_vector)  # Converting into magnitude

    if (
        distance_to_target > stop_distance_offset
    ):  # Only adjust if we're far enough from the target
        adjusted_target = np.array(target_coordinates) - stop_distance_offset * (
            direction_vector / distance_to_target
        )
    else:
        adjusted_target = np.array(
            target_coordinates
        )  # No adjustment needed if too close

    return adjusted_target


def get_traffic_light_status():
    """
    Fetch the real-time traffic light status using the updated library.
    """
    try:
        status_code = traffic_light.status()  # Directly call library function
        if status_code == "1":
            print("Traffic Light Status: RED")
            return "red"
        elif status_code == "2":
            print("Traffic Light Status: YELLOW")
            return "yellow"
        elif status_code == "3":
            print("Traffic Light Status: GREEN")
            return "green"
        else:
            print(f"Unknown status code received: {status_code}")
            return "unknown"
    except Exception as e:
        print(f"Error fetching traffic light status: {e}")
        return "unknown"

        # # Check and print the stdout and stderr
        # stdout_output = result.stdout.decode("utf-8").strip()
        # stderr_output = result.stderr.decode("utf-8").strip()
        # if stderr_output:
        #     print(f"Standard Error: {stderr_output}")

        # # Print and decode the output to get the status code
        # print(f"Standard Output: {stdout_output}")
        # status_code = stdout_output

    #     # Map the status code to color
    #     if status_code == "1":
    #         return "red"
    #     elif status_code == "2":
    #         return "yellow"
    #     elif status_code == "3":
    #         return "green"
    #     else:
    #         print("Unknown status code received:", status_code)
    #         return "unknown"
    # except subprocess.TimeoutExpired:
    #     print("Traffic light status check timed out.")
    #     return "unknown"
    # except FileNotFoundError:
    #     print(
    #         "command_light.py not found. Please ensure the script is in the correct location."
    #     )
    #     return "unknown"
    # except Exception as e:
    #     print(f"Error fetching traffic light status: {e}")
    #     return "unknown"


def traffic_light_status_thread():
    global traffic_light_status
    while not KILL_THREAD:
        traffic_light_status = get_traffic_light_status()
        time.sleep(1)  # Check the traffic light status every 2 seconds


def controlLoop():
    # region controlLoop setup
    global KILL_THREAD
    global traffic_light_status
    u = 0
    delta = 0
    countMax = controllerUpdateRate / 10
    count = 0
    has_stopped = False
    current_target_index = 0
    v_ref = v_cruise
    traffic_light_status = "unknown"
    # endregion

    # region Controller initialization
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)

    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)
    # endregion

    # region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=calibrationPose)
        gps = QCarGPS(initialPose=calibrationPose)
    else:
        gps = memoryview(b"")
    # endregion

    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            # region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t - tp
            # endregion

            # region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach
            # endregion

            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                # Calculate adjusted target for current target index
                if current_target_index < len(target_coordinates_list):
                    target_coordinates = target_coordinates_list[current_target_index]
                    adjusted_target = calculate_adjusted_target(
                        [x, y], target_coordinates, stop_distance_offset
                    )
                    distance_to_adjusted_target = np.linalg.norm(
                        np.array([x, y]) - adjusted_target
                    )

                    # Stop only if the traffic light is red
                    if (
                        not has_stopped
                        and distance_to_adjusted_target <= stop_tolerance
                        and traffic_light_status == "red"
                    ):
                        # Begin stopping process
                        v_ref = 0  # Set speed reference to zero to stop the car
                        has_stopped = True
                        print(
                            f"Stopping at adjusted coordinates: ({adjusted_target[0]:.2f}, {adjusted_target[1]:.2f})"
                        )  # Print adjusted coordinates when stopping
                        print(
                            f"Current QCar coordinates: ({x:.2f}, {y:.2f})"
                        )  # Print current coordinates when stopping
                    elif has_stopped and traffic_light_status == "green":
                        # Resume movement when the light turns green
                        print("Traffic light turned GREEN. Resuming movement...")
                        has_stopped = False  # Reset for next stop
                        current_target_index += 1  # Move to the next target
                        v_ref = v_cruise

                # region : Speed controller update
                u = speedController.update(v, v_ref, dt)
                # endregion

                # region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0
                # endregion

            qcar.write(u, delta)
            # endregion

            count += 1
            if count >= countMax and t > startDelay:
                count = 0
                continue


# endregion

# region : Setup and run experiment
if __name__ == "__main__":
    # Start the traffic light status thread
    trafficLightThread = Thread(target=traffic_light_status_thread)
    trafficLightThread.start()

    # Start the control loop thread
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        trafficLightThread.join()
    input("Experiment complete. Press any key to exit...")
# endregion
