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
*  This script is to implement Task 3A of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ 4181 ]
# Author List:      [ Ankita Sivaramakrishnan, Ansel Vivian Rego, Daniya Mary John, Deeksha NG ]
# Filename:         path_follower.py
# Functions:        body_velocity_to_wheel_speeds, body_to_wheels, yaw_from_quat, wrap,
#                   to_body, GoToPoint.__init__, GoToPoint.step, HeadingHold.__init__,
#                   HeadingHold.step, path_to_traj, path_yaw,
#                   stops_in, position_at, Leg.__init__, Leg.target, RobotNode.__init__,
#                   RobotNode.on_odom, RobotNode.on_plan, RobotNode.on_carry,
#                   RobotNode.on_return, RobotNode.send, RobotNode.now_s,
#                   RobotNode.request_attach, RobotNode.on_attach_reply,
#                   RobotNode.request_detach, RobotNode.on_detach_reply, RobotNode.start_leg,
#                   RobotNode.settled, RobotNode._offset_now, RobotNode.carried_pose,
#                   RobotNode.robot_target, RobotNode.drive, RobotNode.tick, main
# Global variables: V_MAX, W_MAX, KP, KI, KD, KYAW, KIYAW, KDYAW, INTEGRAL_ZONE,
#                   MAX_INTEGRAL, PAYLOAD_NAME, CARRY_V_MAX, SETTLE_TOL, SETTLE_YAW,
#                   DETACH_TOL, SETTLE_TICKS, SETTLE_TIMEOUT, RETRY_AFTER, MAX_TRIES, HEADING_KI,
#                   HEADING_MAX_BIAS, HEADING_ZONE, CARRY_KYAW, NUDGE, MAX_NUDGE


############################ WHAT THIS NODE DOES ##############################
#
#   APPROACH  follow /r2/plan to the docking point, turning to the docking
#             heading (carried in the plan's orientation) on the way
#   ATTACH    settle on the docking point, call /r2/attach_payload
#   CARRY     follow /carry_plan (the CRATE's centre) with the robot at the
#             welded offset from it, holding the docking heading (any heading
#             error is allowed for, so the crate itself stays on the plan)
#   DETACH    settle with the crate on the hexagon, call /r2/detach_payload
#   RETURN    follow /r2/return_plan back to the start circle, and park on it
#
#  Service calls never block tick(): call_async() + a done-callback.
#
#  Run:  ros2 launch hb_description task3a.launch.py
#        ros2 run task_3a path_planner
#        ros2 run task_3a path_follower
#
###############################################################################

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from hb_description.srv import AttachPayload, DetachPayload

################# ADD EXTRA IMPORTS / GLOBALS HERE ############

# Control tuning (Task 2A controller, retuned in 2B for a moving target)
V_MAX = 0.7                    # m/s, cap on the speed
W_MAX = 1.3                     # rad/s, cap on the yaw rate
KP, KI, KD = 4.0, 0.30, 0.05    # position PID gains
KYAW, KIYAW, KDYAW = 0.75, 0.0, 0.05   # heading-hold PID gains
INTEGRAL_ZONE = 0.05            # m, integrate position error only inside this
MAX_INTEGRAL = 0.10             # anti-windup clamp on the position integral

# Task 3A
PAYLOAD_NAME = "payload_beta"
CARRY_V_MAX = 0.7              # m/s, cap while the crate is welded on
SETTLE_TOL = 0.015              # m, "on the point" before a service call
SETTLE_YAW = math.radians(3.0)  # rad, heading error allowed then
DETACH_TOL = 0.008              # m, crate this close to the hexagon's centre: let go
SETTLE_TICKS = 3               # ticks in a row inside both
SETTLE_TIMEOUT = 8.0            # s, call anyway after this long (the gap rule allows 5 cm)
RETRY_AFTER = 1.0               # s, wait between refused service calls
MAX_TRIES = 8                   # service calls before giving up
HEADING_KI = 1.5               # 1/s, integral on the heading error (see HeadingHold)
HEADING_MAX_BIAS = 0.35         # rad, cap on that integral
HEADING_ZONE = 0.15             # rad, integrate only inside this heading error
CARRY_KYAW = 1.5                # heading P gain while carrying
NUDGE = 0.006                   # m, a refused attach moves the target this much
                                # closer (or, if "too close", further away)
MAX_NUDGE = 0.03                # m, ... up to this much

###############################################################


# ----------------------------------------------------- follower (as in Task 2B)
CONTROL_HZ = 75.0
GOAL_TOLERANCE = 0.020        # m, "on the goal"
LEAD = 0.10                   # s, aim this far ahead on the path
MIN_STOP = 0.4                # s, a pause this long in a path is a goal
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

# ----------------------------------------------------- robot (same as Task 1B)
WHEEL_RADIUS_M = 0.0255       # m
CHASSIS_RADIUS_M = 0.06412    # m, chassis centre to each wheel's axle
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0])   # [left, right, back]
IK_MATRIX = np.array([
    [np.cos(t) / WHEEL_RADIUS_M,
     np.sin(t) / WHEEL_RADIUS_M,
     CHASSIS_RADIUS_M / WHEEL_RADIUS_M]
    for t in WHEEL_ANGLES_RAD
])
_CTRL_LIMIT = 30.0            # rad/s, the wheels' ctrlrange in the robot's MJCF


######################### FROM TASK 2A ########################

def body_velocity_to_wheel_speeds(vx, vy, w):
    """Body twist (vx, vy, w) -> wheel speeds [left, right, back], rad/s."""
    speeds = IK_MATRIX @ np.array([vx, vy, w], dtype=float)
    return [float(speed) for speed in speeds]


def body_to_wheels(vx, vy, wz):
    """Like body_velocity_to_wheel_speeds(), but kept inside +/-_CTRL_LIMIT.
    Scales all three wheels together so the driving direction is unchanged."""
    speeds = body_velocity_to_wheel_speeds(vx, vy, wz)
    peak = max(abs(speed) for speed in speeds)
    if peak > _CTRL_LIMIT:
        scale = _CTRL_LIMIT / peak
        speeds = [speed * scale for speed in speeds]
    return [float(speed) for speed in speeds]


def yaw_from_quat(w, x, y, z):
    """Quaternion -> yaw, radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(a):
    """Angle -> the same angle in (-pi, pi]."""
    return math.pi - (math.pi - a) % (2.0 * math.pi)


def to_body(vx_a, vy_a, yaw):
    """Arena-frame velocity -> body-frame velocity (vx, vy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * vx_a + s * vy_a, -s * vx_a + c * vy_a)


class GoToPoint:
    """PID go-to-point holding a fixed yaw (Task 2A, gains retuned in 2B)."""

    def __init__(self, v_max=V_MAX, w_max=W_MAX, kp=KP, ki=KI, kd=KD,
                 kyaw=KYAW, kiyaw=KIYAW, kdyaw=KDYAW, tol=0.012, dt=1.0 / CONTROL_HZ):
        self.v_max, self.w_max = v_max, w_max
        self.kp, self.ki, self.kd = kp, ki, kd
        self.kyaw, self.kiyaw, self.kdyaw = kyaw, kiyaw, kdyaw
        self.tol = tol
        self.dt = dt
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_yaw = 0.0
        self.prev_x_err = 0.0
        self.prev_y_err = 0.0
        self.prev_yaw_err = 0.0
        self._first = True

    def step(self, pose, target, hold_yaw):
        """pose (x, y, yaw), target (x, y), hold_yaw -> (wheels, distance)."""
        x, y, yaw = pose
        ex = target[0] - x
        ey = target[1] - y
        dist = math.hypot(ex, ey)

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

        yaw_err = wrap(hold_yaw - yaw)
        if self._first:
            d_yaw = 0.0
        else:
            d_yaw = (yaw_err - self.prev_yaw_err) / self.dt
        self.prev_yaw_err = yaw_err

        wz = self.kyaw * yaw_err + self.kiyaw * self.int_yaw + self.kdyaw * d_yaw
        wz = max(-self.w_max, min(self.w_max, wz))

        self._first = False

        vx_b, vy_b = to_body(vx_a, vy_a, yaw)
        return body_to_wheels(vx_b, vy_b, wz), dist


class HeadingHold(GoToPoint):
    """GoToPoint with an integral on the heading. The 2A controller's yaw loop
    is P only, and a welded crate drags on the robot, so the heading sits a few
    degrees off; 0.18 m out in front, that moves the crate by centimetres.
    The integral is fed in as a bias on hold_yaw, leaving GoToPoint as it was."""

    def __init__(self, ki_heading=HEADING_KI, max_bias=HEADING_MAX_BIAS, **kw):
        super().__init__(**kw)
        self.ki_heading, self.max_bias, self.bias = ki_heading, max_bias, 0.0

    def step(self, pose, target, hold_yaw):
        err = wrap(hold_yaw - pose[2])
        if abs(err) < HEADING_ZONE:                     # not while still turning: no windup
            self.bias += self.ki_heading * err * self.dt
        self.bias = max(-self.max_bias, min(self.max_bias, self.bias))
        return super().step(pose, target, hold_yaw + self.bias)


######################### PATH (Task 2B) ######################

def path_to_traj(msg):
    """nav_msgs/Path -> [(t, x, y), ...], t = pose stamp - path stamp, in s."""
    t0 = Time.from_msg(msg.header.stamp)
    return [((Time.from_msg(p.header.stamp) - t0).nanoseconds * 1e-9,
             p.pose.position.x, p.pose.position.y)
            for p in msg.poses]


def path_yaw(msg):
    """The heading the planner stored in the path's (last) pose."""
    q = msg.poses[-1].pose.orientation
    return yaw_from_quat(q.w, q.x, q.y, q.z)


def stops_in(traj):
    """Goals on a timed path: [(arrive, leave, x, y), ...], in s.
    A goal is a pause of at least MIN_STOP, or the end; the last is never left."""
    stops = [(t0, t1, x1, y1) for (t0, x0, y0), (t1, x1, y1) in zip(traj, traj[1:])
             if (x0, y0) == (x1, y1) and t1 - t0 >= MIN_STOP]
    if not stops or stops[-1][2:] != traj[-1][1:]:
        stops.append((traj[-1][0], traj[-1][0], *traj[-1][1:]))
    arrive, _, x, y = stops[-1]
    stops[-1] = (arrive, math.inf, x, y)
    return stops


def position_at(traj, t):
    """Where the path says the robot should be at time t."""
    if t <= traj[0][0]:
        return traj[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(traj, traj[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return traj[-1][1:]


class Leg:
    """One timed path for the robot's centre, followed as in Task 2B: aim
    LEAD s ahead, stop on each goal, hold the path clock while late."""

    def __init__(self, traj, offset=(0.0, 0.0)):
        self.traj = [(t, x + offset[0], y + offset[1]) for t, x, y in traj]
        self.stops = stops_in(self.traj)
        self.reached, self.held = 0, 0.0

    def target(self, now, pose):
        """now = s since this leg's start -> (target, finished)."""
        t = now - self.held
        arrive, leave, *goal = self.stops[self.reached]
        if t >= arrive:
            if math.dist(pose[:2], goal) < GOAL_TOLERANCE:
                self.reached += 1
                if self.reached == len(self.stops):
                    return tuple(goal), True
                arrive, leave, *goal = self.stops[self.reached]
            elif t > leave:
                self.held += t - leave
                t = leave
        target = goal if t >= arrive else position_at(self.traj, t + LEAD)
        return tuple(target), False


######################### NODE ################################

class RobotNode(Node):
    """Approach, dock, attach, carry, let go, drive back. Node: task_3a_r2."""

    def __init__(self, robot="r2"):
        super().__init__(f"task_3a_{robot}")
        self.robot = robot
        self.pose = None
        self.plan = self.carry = self.ret = None       # Path messages
        self.phase = "WAIT"
        self.leg = None
        self.leg_start = 0.0                            # s (node clock) the leg started
        self.t_start = None                             # s, when r2 set off
        self.hold_yaw = None
        self.slot = None                                # planned robot centre - crate centre
        self.offset = None                              # the same, as welded
        self.dock = None
        self.settle, self.settle_since = 0, None
        self.pending, self.tries, self.next_try = False, 0, 0.0
        self.nudge = 0.0

        self.create_subscription(Odometry, f"/{robot}/odom", self.on_odom, 10)
        self.create_subscription(Path, f"/{robot}/plan", self.on_plan, LATCHED)
        self.create_subscription(Path, "/carry_plan", self.on_carry, LATCHED)
        self.create_subscription(Path, f"/{robot}/return_plan", self.on_return, LATCHED)
        self.pub = self.create_publisher(Float64MultiArray, f"/{robot}/wheel_commands", 10)
        self.attach_cli = self.create_client(AttachPayload, f"/{robot}/attach_payload")
        self.detach_cli = self.create_client(DetachPayload, f"/{robot}/detach_payload")

        self.ctl = HeadingHold(v_max=V_MAX, tol=0.012, dt=1.0 / CONTROL_HZ)
        self.ctl_carry = HeadingHold(v_max=CARRY_V_MAX, kyaw=CARRY_KYAW, tol=0.004,
                                     dt=1.0 / CONTROL_HZ)
        self.ctl_fine = HeadingHold(v_max=0.08, kyaw=CARRY_KYAW, tol=0.003, dt=1.0 / CONTROL_HZ)
        self.create_timer(1.0 / CONTROL_HZ, self.tick)
        self.get_logger().info("task 3A follower: waiting for odom and the three plans")

    # ------------------------------------------------ inputs
    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(q.w, q.x, q.y, q.z))

    def on_plan(self, msg):
        if self.plan is None and msg.poses:
            self.plan = msg
            self.get_logger().info("got /%s/plan: %d poses" % (self.robot, len(msg.poses)))

    def on_carry(self, msg):
        if self.carry is None and msg.poses:
            self.carry = msg
            self.get_logger().info("got /carry_plan: %d poses" % len(msg.poses))

    def on_return(self, msg):
        if self.ret is None and msg.poses:
            self.ret = msg
            self.get_logger().info("got /%s/return_plan: %d poses" % (self.robot, len(msg.poses)))

    def send(self, wheels):
        self.pub.publish(Float64MultiArray(data=[float(w) for w in wheels]))

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------ services (never block)
    def request_attach(self):
        """Send /r2/attach_payload and return at once; on_attach_reply acts."""
        if self.pending:
            return
        if not self.attach_cli.service_is_ready():
            self.get_logger().warn("/%s/attach_payload not available yet" % self.robot,
                                   throttle_duration_sec=2.0)
            return
        req = AttachPayload.Request()
        req.payload_name = PAYLOAD_NAME
        self.pending = True
        self.tries += 1
        future = self.attach_cli.call_async(req)
        future.add_done_callback(self.on_attach_reply)

    def on_attach_reply(self, future):
        self.pending = False
        try:
            reply = future.result()
        except Exception as e:                          # the call itself failed
            reply = None
            self.get_logger().warn("attach call failed: %s" % e)
        if reply is not None and (reply.success or "already attached to '%s'"
                                  % PAYLOAD_NAME in reply.message.lower()):
            self.get_logger().info("attached %s: carrying" % PAYLOAD_NAME)
            # the crate is welded where it is: along the arrow it touches the
            # solenoid (so it sits |slot| ahead), but sideways it is wherever the
            # robot stopped relative to the crate the camera found
            # welded where it is: robot centre - crate centre, the crate
            # being where the planner found it
            carry = path_to_traj(self.carry)
            self.offset = (self.pose[0] - carry[0][1], self.pose[1] - carry[0][2])
            self.start_leg("CARRY", carry)              # followed in crate coordinates
            return
        self.get_logger().warn("attach refused (try %d): %s"
                               % (self.tries, reply.message if reply else "no reply"))
        # "too close": back off a little; anything else (not touching yet):
        # creep a little closer. Then try again
        if reply is not None and "too close" in reply.message.lower():
            self.nudge = max(-MAX_NUDGE, self.nudge - NUDGE)
        else:
            self.nudge = min(MAX_NUDGE, self.nudge + NUDGE)
        self.next_try = self.now_s() + RETRY_AFTER
        self.settle, self.settle_since = 0, None
        if self.tries >= MAX_TRIES:
            self.get_logger().error("giving up on attaching %s" % PAYLOAD_NAME)
            self.phase = "FAILED"

    def request_detach(self):
        """Send /r2/detach_payload and return at once; on_detach_reply acts."""
        if self.pending:
            return
        if not self.detach_cli.service_is_ready():
            self.get_logger().warn("/%s/detach_payload not available yet" % self.robot,
                                   throttle_duration_sec=2.0)
            return
        req = DetachPayload.Request()
        req.payload_name = PAYLOAD_NAME
        self.pending = True
        self.tries += 1
        future = self.detach_cli.call_async(req)
        future.add_done_callback(self.on_detach_reply)

    def on_detach_reply(self, future):
        self.pending = False
        try:
            reply = future.result()
        except Exception as e:
            reply = None
            self.get_logger().warn("detach call failed: %s" % e)
        if reply is not None and reply.success:
            self.get_logger().info("let go of %s" % (reply.detached_name or PAYLOAD_NAME))
            self.start_leg("RETURN", path_to_traj(self.ret))
            return
        self.get_logger().warn("detach refused (try %d): %s"
                               % (self.tries, reply.message if reply else "no reply"))
        self.next_try = self.now_s() + RETRY_AFTER
        self.settle, self.settle_since = 0, None
        if self.tries >= MAX_TRIES:
            self.get_logger().error("giving up on letting go of %s" % PAYLOAD_NAME)
            self.phase = "FAILED"

    # ------------------------------------------------ helpers
    def start_leg(self, phase, traj, offset=(0.0, 0.0)):
        self.phase = phase
        self.leg = Leg(traj, offset)
        self.leg_start = self.now_s()
        self.settle, self.settle_since = 0, None
        self.tries, self.next_try = 0, 0.0

    def settled(self, target, pose=None, check_yaw=True, tol=SETTLE_TOL):
        """True once `pose` (default: the robot's) has sat on `target` with the
        right heading for SETTLE_TICKS ticks (or SETTLE_TIMEOUT s have passed)."""
        pose = self.pose if pose is None else pose
        if self.settle_since is None:
            self.settle_since = self.now_s()
        ok = (math.dist(pose[:2], target) < tol
              and (not check_yaw or abs(wrap(self.hold_yaw - self.pose[2])) < SETTLE_YAW))
        self.settle = self.settle + 1 if ok else 0
        return self.settle >= SETTLE_TICKS or self.now_s() - self.settle_since > SETTLE_TIMEOUT

    def _offset_now(self):
        """Robot centre - crate centre right now: the welded offset, turned by
        however far the heading is off the one it was welded at."""
        a = wrap(self.pose[2] - self.hold_yaw)
        c, s = math.cos(a), math.sin(a)
        ox, oy = self.offset
        return (c * ox - s * oy, s * ox + c * oy)

    def carried_pose(self):
        """Where the welded crate's centre is, from the robot's pose."""
        ox, oy = self._offset_now()
        return (self.pose[0] - ox, self.pose[1] - oy, self.pose[2])

    def robot_target(self, crate_target):
        """Robot position that puts the crate's centre on `crate_target`."""
        ox, oy = self._offset_now()
        return (crate_target[0] + ox, crate_target[1] + oy)

    def drive(self, ctl, target):
        wheels, _ = ctl.step(self.pose, target, self.hold_yaw)
        self.send(wheels)

    # ------------------------------------------------ control loop
    def tick(self):
        if self.pose is None or self.phase in ("DONE", "FAILED"):
            self.send((0.0, 0.0, 0.0))
            return

        if self.phase == "WAIT":
            if self.plan is None or self.carry is None or self.ret is None:
                self.send((0.0, 0.0, 0.0))
                return
            start = Time.from_msg(self.plan.header.stamp).nanoseconds * 1e-9
            if self.now_s() < start:                    # not the start time yet
                self.send((0.0, 0.0, 0.0))
                return
            self.hold_yaw = path_yaw(self.plan)         # turn to the docking heading
            approach = path_to_traj(self.plan)
            self.dock = approach[-1][1:]
            crate0 = path_to_traj(self.carry)[0][1:]
            self.slot = (self.dock[0] - crate0[0], self.dock[1] - crate0[1])
            self.t_start = start
            self.start_leg("APPROACH", approach)
            # started late (follower launched after the set-off time): start the
            # path clock now, or the first target would be far down the path
            self.leg_start = max(start, self.now_s() - 0.1)

        now = self.now_s() - self.leg_start

        if self.phase == "APPROACH":
            target, done = self.leg.target(now, self.pose)
            self.drive(self.ctl, target)
            if done:
                self.get_logger().info("on the docking point (%.0f mm out): attaching"
                                       % (1000 * math.dist(self.pose[:2], self.dock)))
                self.phase = "ATTACH"

        elif self.phase == "ATTACH":
            # sit on the docking point (crept closer after each refusal)
            n = math.hypot(*self.slot)
            target = (self.dock[0] - self.slot[0] / n * self.nudge,
                      self.dock[1] - self.slot[1] / n * self.nudge)
            self.drive(self.ctl_fine, target)
            if not self.pending and self.now_s() >= self.next_try and self.settled(target):
                self.request_attach()

        elif self.phase == "CARRY":
            target, done = self.leg.target(now, self.carried_pose())
            self.drive(self.ctl_carry, self.robot_target(target))
            if done:
                self.get_logger().info("crate on the hexagon's centre: letting go")
                self.phase = "DETACH"
                self.settle, self.settle_since = 0, None

        elif self.phase == "DETACH":
            target = self.leg.traj[-1][1:]
            self.drive(self.ctl_fine, self.robot_target(target))
            if (not self.pending and self.now_s() >= self.next_try
                    and self.settled(target, self.carried_pose(), check_yaw=False,
                                     tol=DETACH_TOL)):
                # the crate's position already allows for the heading being off:
                # no need to wait for the robot to turn back (slow with the crate on)
                self.request_detach()

        elif self.phase == "RETURN":
            target, done = self.leg.target(now, self.pose)
            self.drive(self.ctl, target)
            if done:                                    # within GOAL_TOLERANCE of the start
                self.phase = "DONE"
                self.send((0.0, 0.0, 0.0))
                self.get_logger().info("%s back on its start circle (%.0f mm out)"
                                       % (self.robot, 1000 * math.dist(self.pose[:2], target)))
                self.get_logger().info("task 3A complete in %.1fs" % (self.now_s() - self.t_start))

        elif self.phase == "PARK":                      # settle exactly on the start circle
            target = self.leg.traj[-1][1:]
            self.drive(self.ctl_fine, target)
            if self.settled(target):
                self.phase = "DONE"
                self.send((0.0, 0.0, 0.0))
                self.get_logger().info("%s back on its start circle (%.0f mm out)"
                                       % (self.robot, 1000 * math.dist(self.pose[:2], target)))
                self.get_logger().info("task 3A complete in %.1fs" % (self.now_s() - self.t_start))


def main():
    rclpy.init()
    node = RobotNode("r2")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.send((0.0, 0.0, 0.0))                  # leave nothing driving
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

