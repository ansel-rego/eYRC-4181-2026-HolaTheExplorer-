#!/usr/bin/env python3
"""
Boilerplate controller for the HE bot.

Fetches a shape from the get_shape service and builds a list of waypoints
to trace it. Fill in the control loop to drive the robot through them.


"""

import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from shape_interface.srv import GetShape

# Wheel <-> body-velocity mapping, columns are [left, right, back] wheel
# speed (rad/s); rows are body frame [vx, vy, wz] per unit wheel speed.
_WHEEL_TO_BODY = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

_CTRL_LIMIT = 10.0

WAYPOINT_TOLERANCE  = 0.004
CIRCLE_SEGMENTS     = 72
POSITION_KP         = 0.6
POSITION_KI         = 0.30
POSITION_KD         = 0.07
INTEGRAL_ZONE       = 0.05
MAX_INTEGRAL        = 0.10
YAW_HOLD_KP         = 0.75
YAW_HOLD_KD         = 0.05
CONTROL_PERIOD      = 0.05


def body_to_wheels(vx, vy, wz):
    """Body-frame (vx, vy, wz) -> wheel angular velocities [left, right, back]."""
    wheel_radius = 0.0255
    chassis_radius = 0.06412
    wheel_angles = np.radians([30.0, 150.0, 270.0])

    speeds = [
        (vx * np.cos(theta) + vy * np.sin(theta)
         + wz * chassis_radius) / wheel_radius
        for theta in wheel_angles
    ]

    return [
        float(np.clip(speed, -_CTRL_LIMIT, _CTRL_LIMIT))
        for speed in speeds
    ]


def yaw_from_quat(w, x, y, z):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _regular_polygon(cx, cy, n_sides, side_length, start_angle=math.pi / 2):
    """Vertices of a regular polygon centred on (cx, cy), closed back to the
    first vertex so the last waypoint returns the robot to where it started
    drawing."""
    r = side_length / (2 * math.sin(math.pi / n_sides))
    pts = [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]
    return pts + [pts[0]]


def build_waypoints(shape_name, data):
    """World-frame waypoints for `shape_name`, as returned by the get_shape
    service: data[0:2] is the shape's centre (x, y); the remaining entries
    are its size parameters (see shape_service.cpp's shape_map)."""
    cx, cy = data[0], data[1]

    if shape_name == "Circle":
        radius = data[2]
        return [
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS))
            for i in range(1, CIRCLE_SEGMENTS + 1)
        ]

    if shape_name == "Square":
        return _regular_polygon(cx, cy, 4, data[2], start_angle=math.pi / 4)

    if shape_name == "Triangle":
        return _regular_polygon(cx, cy, 3, data[2])

    if shape_name == "Pentagon":
        return _regular_polygon(cx, cy, 5, data[2])

    if shape_name == "Rectangle":
        w, h = data[2], data[3]
        corners = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        return corners + [corners[0]]

    raise ValueError(f"unknown shape '{shape_name}'")


class ShapeController(Node):
    def __init__(self, speed):
        super().__init__("shape_controller")
        self.speed = speed

        self.pose = None
        self.start_pose = None
        self.wp_index = 0
        self.done = False

        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_e_theta = 0.0

        self.int_ex = 0.0
        self.int_ey = 0.0

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/wheel_commands",
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_cb,
            10
        )

        #Add the publsiher and subscriber scripts

        self.shape_name, self.waypoints = self._request_shape()

        self.timer = self.create_timer(
            CONTROL_PERIOD,
            self._control_step
        )

    def _request_shape(self):
        client = self.create_client(GetShape, "get_shape")

        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for get_shape service...")

        future = client.call_async(GetShape.Request())
        rclpy.spin_until_future_complete(self, future)

        response = future.result()

        if response is None or not response.success:
            raise RuntimeError(
                f"get_shape service call failed: "
                f"{response and response.message}"
            )

        waypoints = build_waypoints(
            response.shape_name,
            list(response.data)
        )

        self.get_logger().info(
            f"Shape: {response.shape_name}, "
            f"Waypoints: {len(waypoints)}"
        )

        return response.shape_name, waypoints

    def _odom_cb(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        yaw = yaw_from_quat(
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z
        )

        self.pose = (
            position.x,
            position.y,
            yaw
        )

        if self.start_pose is None:
            self.start_pose = self.pose

    def _publish(self, wheels):
        self.cmd_pub.publish(
            Float64MultiArray(data=wheels)
        )

    def _control_step(self):
        if self.done or self.pose is None:
            return

        if self.wp_index >= len(self.waypoints):
            self.done = True
            self._publish([0.0, 0.0, 0.0])
            return

        x, y, yaw = self.pose

        target_x, target_y = self.waypoints[self.wp_index]

        ex = target_x - x
        ey = target_y - y

        distance = math.sqrt(ex * ex + ey * ey)

        if distance <= WAYPOINT_TOLERANCE:
            self.wp_index += 1
            self.int_ex = 0.0
            self.int_ey = 0.0

            if self.wp_index >= len(self.waypoints):
                self.done = True
                self._publish([0.0, 0.0, 0.0])
                return

            target_x, target_y = self.waypoints[self.wp_index]

            ex = target_x - x
            ey = target_y - y

            # Re-seed the derivative history on the new target instead of
            # zeroing it: zeroing makes the first derivative sample a large
            # spurious velocity kick that rounds the corner off.
            self.prev_ex = ex
            self.prev_ey = ey

        distance = math.sqrt(ex * ex + ey * ey)

        dex = (ex - self.prev_ex) / CONTROL_PERIOD
        dey = (ey - self.prev_ey) / CONTROL_PERIOD

        self.prev_ex = ex
        self.prev_ey = ey

        # Integrate only inside the terminal zone: this closes the last few
        # millimetres onto each vertex (the residual standoff that shifts the
        # whole edge sideways) without winding up during the long traverse.
        if distance <= INTEGRAL_ZONE:
            self.int_ex += ex * CONTROL_PERIOD
            self.int_ey += ey * CONTROL_PERIOD
        else:
            self.int_ex = 0.0
            self.int_ey = 0.0

        self.int_ex = float(np.clip(self.int_ex, -MAX_INTEGRAL, MAX_INTEGRAL))
        self.int_ey = float(np.clip(self.int_ey, -MAX_INTEGRAL, MAX_INTEGRAL))

        vx_world = (
            POSITION_KP * ex +
            POSITION_KI * self.int_ex +
            POSITION_KD * dex
        )
        vy_world = (
            POSITION_KP * ey +
            POSITION_KI * self.int_ey +
            POSITION_KD * dey
        )

        velocity = math.sqrt(
            vx_world * vx_world +
            vy_world * vy_world
        )

        if velocity > self.speed:
            scale = self.speed / velocity
            vx_world *= scale
            vy_world *= scale

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        vx_body = (
            cos_yaw * vx_world +
            sin_yaw * vy_world
        )

        vy_body = (
            -sin_yaw * vx_world +
            cos_yaw * vy_world
        )

        desired_yaw = self.start_pose[2]

        e_theta = desired_yaw - yaw

        while e_theta > math.pi:
            e_theta -= 2.0 * math.pi

        while e_theta < -math.pi:
            e_theta += 2.0 * math.pi

        de_theta = (
            e_theta - self.prev_e_theta
        ) / CONTROL_PERIOD

        self.prev_e_theta = e_theta

        wz = (
            YAW_HOLD_KP * e_theta +
            YAW_HOLD_KD * de_theta
        )

        wz = float(np.clip(wz, -1.0, 1.0))

        wheels = body_to_wheels(
            vx_body,
            vy_body,
            wz
        )

        self._publish(wheels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speed",
        type=float,
        default=0.10,
        help="max approach speed, m/s"
    )

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = ShapeController(args.speed)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

