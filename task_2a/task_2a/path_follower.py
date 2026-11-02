#!/usr/bin/env python3
# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

'''
*****************************************************************************************
*
*        =============================================
*           Hola The Explorer (HE) Theme (eYRC 2026-27)
*        =============================================
*
*  This script is to implement Task 2A of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:      [ Names of team members who worked on this file, separated by comma ]
# Filename:         path_follower.py
# Functions:        body_velocity_to_wheel_speeds, body_to_wheels, yaw_from_quat,
#                   wrap, to_body, frame_check, GoToPoint.__init__, GoToPoint.step,
#                   RobotNode.__init__, RobotNode.odom_cb, RobotNode.send,
#                   RobotNode.tick, main
# Global variables: ARENA_SIZE, ROBOT_RADIUS, START, GOALS, ORDER, CONTROL_HZ,
#                   GOAL_TOLERANCE, SETTLE_TICKS, WHEEL_RADIUS_M, CHASSIS_RADIUS_M,
#                   WHEEL_ANGLES_RAD, IK_MATRIX, _CTRL_LIMIT, V_MAX, W_MAX,
#                   KP, KI, KD, KYAW, KIYAW, KDYAW, INTEGRAL_ZONE, MAX_INTEGRAL


############################ WHAT YOU HAVE TO DO ##############################
#
#  Drive r1, r2 and r3 AT THE SAME TIME, each in a straight line to its goal
#  in GOALS, holding its starting heading, and park it there.
#
#   1. Paste in your Task 1B / 1C code   (body_velocity_to_wheel_speeds, ...)
#   2. Write wrap(), to_body() and the GoToPoint PID controller
#   3. Complete RobotNode                (main() runs one per robot)
#
#  Frame (/rN/odom): origin at the arena's TOP-LEFT corner, x right, y DOWN,
#  metres. Yaw 0 along +x, growing clockwise; the robots start at -pi/2.
#
#  Run:  ros2 launch hb_description task2a.launch.py
#        ros2 run task_2a path_follower
#
###############################################################################

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

################# ADD EXTRA IMPORTS / GLOBALS HERE ############

# Control tuning (mirrors the Task 1C PI(D) controller).
V_MAX = 0.50                    # m/s, cap on the approach speed
W_MAX = 1.0                     # rad/s, cap on the yaw rate
KP, KI, KD = 0.9, 0.20, 0.07    # position PID gains
KYAW, KIYAW, KDYAW = 0.75, 0.0, 0.05   # heading-hold PID gains
INTEGRAL_ZONE = 0.05            # m, integrate position error only inside this
MAX_INTEGRAL = 0.10             # anti-windup clamp on the position integral

###############################################################


# ----------------------------------------------------- arena (given, do not edit)
ARENA_SIZE = 2.4384           # m, the floor is square
ROBOT_RADIUS = 0.084          # m

START = {"r1": (0.8969, 2.2616),
         "r2": (1.2192, 2.2616),
         "r3": (1.5415, 2.2616)}
GOALS = {"r1": (0.6692, 0.8692),
         "r2": (1.2192, 0.6692),
         "r3": (1.7692, 0.8692)}
ORDER = ("r1", "r2", "r3")    # one node per robot

# ----------------------------------------------------- yours to set
CONTROL_HZ = 25.0             # how often tick() runs
GOAL_TOLERANCE = 0.010        # m, how close counts as "at the goal"
SETTLE_TICKS = 5              # ticks in a row inside GOAL_TOLERANCE = arrived

# ----------------------------------------------------- robot (same as Task 1B)
WHEEL_RADIUS_M = 0.0255       # m
CHASSIS_RADIUS_M = 0.06412    # m, chassis centre to each wheel's axle
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0])   # from Task 1B, [left, right, back]
# Inverse kinematics: wheel_speeds = IK_MATRIX @ [vx, vy, w] (Task 1B).
IK_MATRIX = np.array([
    [np.cos(t) / WHEEL_RADIUS_M,
     np.sin(t) / WHEEL_RADIUS_M,
     CHASSIS_RADIUS_M / WHEEL_RADIUS_M]
    for t in WHEEL_ANGLES_RAD
])
_CTRL_LIMIT = 30.0            # rad/s, the wheels' ctrlrange in the robot's MJCF:
                              # faster commands are clamped by the simulation


##################### FROM TASK 1B / 1C #######################

def body_velocity_to_wheel_speeds(vx, vy, w):
    """Body twist (vx, vy, w) -> wheel speeds [left, right, back], rad/s."""
    ##############  ADD YOUR CODE HERE  ##############
    speeds = IK_MATRIX @ np.array([vx, vy, w], dtype=float)
    return [float(speed) for speed in speeds]
    ##################################################


def body_to_wheels(vx, vy, wz):
    """Like body_velocity_to_wheel_speeds(), but kept inside +/-_CTRL_LIMIT.
    Returns a list."""
    ##############  ADD YOUR CODE HERE  ##############
    # Task 1C. Scale all three wheels down together, do not clip each one:
    # clipping changes the direction the robot drives in.
    speeds = body_velocity_to_wheel_speeds(vx, vy, wz)

    peak = max(abs(speed) for speed in speeds)
    if peak > _CTRL_LIMIT:
        scale = _CTRL_LIMIT / peak
        speeds = [speed * scale for speed in speeds]

    return [float(speed) for speed in speeds]
    ##################################################


def yaw_from_quat(w, x, y, z):
    """Quaternion -> yaw, radians."""
    ##############  ADD YOUR CODE HERE  ##############
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
    ##################################################


######################### NEW IN 2A ###########################

def wrap(a):
    """Angle -> the same angle in (-pi, pi]. 350 degrees becomes -10."""
    ##############  ADD YOUR CODE HERE  ##############
    return math.pi - (math.pi - a) % (2.0 * math.pi)
    ##################################################


def to_body(vx_a, vy_a, yaw):
    """Arena-frame velocity -> body-frame velocity (vx, vy).
    Check: at yaw = -pi/2 (facing up), arena (0, -v) must give body (v, 0)."""
    ##############  ADD YOUR CODE HERE  ##############
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * vx_a + s * vy_a, -s * vx_a + c * vy_a)
    ##################################################


def frame_check(x, y, size=ARENA_SIZE):
    """Given. A complaint if (x, y) is not in the /odom frame, else None."""
    if not (0.0 <= x <= size and 0.0 <= y <= size):
        return ("odom pose (%.3f, %.3f) is outside 0..%.4f: this code works "
                "in the top-left-origin arena frame, not MuJoCo's "
                "centre-origin one" % (x, y, size))
    return None


class GoToPoint:
    """PID controller: drive one robot straight at a point, holding a fixed yaw.
    One instance per robot -- the PID terms keep state between steps."""

    def __init__(self, v_max=0.0, w_max=0.0, kp=0.0, ki=0.0, kd=0.0,
                 kyaw=0.0, kiyaw=0.0, kdyaw=0.0, tol=0.0, dt=0.0):
        self.v_max, self.w_max = v_max, w_max            # m/s, rad/s caps
        self.kp, self.ki, self.kd = kp, ki, kd           # position PID
        self.kyaw, self.kiyaw, self.kdyaw = kyaw, kiyaw, kdyaw   # yaw PID
        self.tol = tol                                   # m, stop inside this
        self.dt = dt                                     # s, 1 / CONTROL_HZ
        ##############  ADD YOUR CODE HERE  ##############
        # PID state -- integrals and previous errors for (x, y) and yaw.
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_yaw = 0.0
        self.prev_x_err = 0.0
        self.prev_y_err = 0.0
        self.prev_yaw_err = 0.0
        self._first = True                # suppress the derivative kick on tick 1
        ##################################################

    def step(self, pose, target, hold_yaw):
        """pose (x, y, yaw), target (x, y), hold_yaw -> (wheels, distance)."""
        ##############  ADD YOUR CODE HERE  ##############
        x, y, yaw = pose
        ex = target[0] - x
        ey = target[1] - y
        dist = math.hypot(ex, ey)

        # 1./2. position PID -> arena-frame velocity, zeroed inside tol and
        #       capped at v_max, with the integral kept from winding up.
        if dist <= self.tol:
            self.int_x = 0.0
            self.int_y = 0.0
            vx_a = 0.0
            vy_a = 0.0
        else:
            if dist <= INTEGRAL_ZONE:
                self.int_x += ex * self.dt
                self.int_y += ey * self.dt
            else:
                self.int_x = 0.0
                self.int_y = 0.0
            self.int_x = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.int_x))
            self.int_y = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.int_y))

            if self._first:
                d_x = 0.0
                d_y = 0.0
            else:
                d_x = (ex - self.prev_x_err) / self.dt
                d_y = (ey - self.prev_y_err) / self.dt

            vx_a = self.kp * ex + self.ki * self.int_x + self.kd * d_x
            vy_a = self.kp * ey + self.ki * self.int_y + self.kd * d_y

            speed = math.hypot(vx_a, vy_a)
            if speed > self.v_max:
                vx_a *= self.v_max / speed
                vy_a *= self.v_max / speed

        self.prev_x_err = ex
        self.prev_y_err = ey

        # 3. yaw PID: hold the starting heading, error wrapped to (-pi, pi].
        yaw_err = wrap(hold_yaw - yaw)
        if self._first:
            d_yaw = 0.0
        else:
            d_yaw = (yaw_err - self.prev_yaw_err) / self.dt
        self.prev_yaw_err = yaw_err

        wz = self.kyaw * yaw_err + self.kiyaw * self.int_yaw + self.kdyaw * d_yaw
        wz = max(-self.w_max, min(self.w_max, wz))

        self._first = False

        # 4. arena velocity -> body velocity -> wheel speeds.
        vx_b, vy_b = to_body(vx_a, vy_a, yaw)
        return body_to_wheels(vx_b, vy_b, wz), dist
        ##################################################


######################### ONE NODE PER ROBOT ##################

class RobotNode(Node):
    """Drives one robot to GOALS[robot]. Nodes: task_2a_r1, _r2, _r3.
    No robot moves until all three are in `ready`, so they start together."""

    def __init__(self, robot, ready, on_arrival):
        super().__init__(f"task_2a_{robot}")
        self.robot = robot
        self.ready = ready            # shared by the three nodes
        self.on_arrival = on_arrival  # call once: on_arrival(robot, seconds)

        self.pose = None              # (x, y, yaw) from /<robot>/odom
        self.hold_yaw = None          # yaw on the first odom message
        self.settled = 0
        self.arrived = False
        self.t_start = None
        ##############  ADD YOUR CODE HERE  ##############
        self.sub = self.create_subscription(
            Odometry, f"/{robot}/odom", self.odom_cb, 10)
        self.pub = self.create_publisher(
            Float64MultiArray, f"/{robot}/wheel_commands", 10)

        self.ctl = GoToPoint(
            v_max=V_MAX, w_max=W_MAX,
            kp=KP, ki=KI, kd=KD,
            kyaw=KYAW, kiyaw=KIYAW, kdyaw=KDYAW,
            tol=GOAL_TOLERANCE, dt=1.0 / CONTROL_HZ)

        self.timer = self.create_timer(1.0 / CONTROL_HZ, self.tick)
        ##################################################

    def odom_cb(self, msg):
        ##############  ADD YOUR CODE HERE  ##############
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.w, q.x, q.y, q.z)
        self.pose = (p.x, p.y, yaw)

        if self.hold_yaw is None:
            self.hold_yaw = yaw
            complaint = frame_check(p.x, p.y)
            if complaint:
                self.get_logger().warn(complaint)
            self.ready.add(self.robot)
        ##################################################

    def send(self, wheels):
        self.pub.publish(Float64MultiArray(data=[float(w) for w in wheels]))

    def tick(self):
        if len(self.ready) < len(ORDER):
            return                                   # wait for every robot's odom
        if self.t_start is None:
            self.t_start = self.get_clock().now()
        ##############  ADD YOUR CODE HERE  ##############
        if self.arrived:
            self.send((0.0, 0.0, 0.0))
            return

        if self.pose is None:
            return

        wheels, dist = self.ctl.step(
            self.pose, GOALS[self.robot], self.hold_yaw)
        self.send(wheels)

        if dist <= GOAL_TOLERANCE:
            self.settled += 1
        else:
            self.settled = 0

        if self.settled >= SETTLE_TICKS:
            self.arrived = True
            self.send((0.0, 0.0, 0.0))
            elapsed = (self.get_clock().now() - self.t_start).nanoseconds * 1e-9
            self.get_logger().info(
                "%s reached its goal (%.1f mm out)" % (self.robot, dist * 1000.0))
            self.on_arrival(self.robot, elapsed)
        ##################################################


def main():
    """Given. One RobotNode per robot; logs "task 2A complete" at the end."""
    rclpy.init()
    ready = set()
    arrived = {}

    def on_arrival(robot, t):
        arrived[robot] = t
        if len(arrived) == len(ORDER):
            nodes[0].get_logger().info("task 2A complete in %.1fs" % max(arrived.values()))

    nodes = [RobotNode(r, ready, on_arrival) for r in ORDER]
    executor = MultiThreadedExecutor(num_threads=len(ORDER))
    for n in nodes:
        executor.add_node(n)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for n in nodes:
            try:
                n.send((0.0, 0.0, 0.0))          # leave nothing driving
            except Exception:                    # no publisher yet, or Ctrl+C took the context down
                pass
            n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

