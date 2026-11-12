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
*  This script is to implement Task 3B of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ 4181 ]
# Author List:      [ Names of team members who worked on this file, separated by comma ]
# Filename:         path_follower.py
# Functions:        body_velocity_to_wheel_speeds, body_to_wheels, yaw_from_quat,
#                   wrap, to_body, GoToPoint.__init__, GoToPoint.step, path_to_traj,
#                   request_attach, request_detach
# Global variables: V_MAX, W_MAX, KP, KI, KD, KYAW, KIYAW, KDYAW,
#                   INTEGRAL_ZONE, MAX_INTEGRAL


############################ WHAT YOU HAVE TO DO ##############################
#
#  Each robot approaches along /rN/plan, docks on its side of payload_gamma,
#  and -- once ALL THREE are docked -- welds it with /rN/attach_payload. Once
#  all three hold it they carry it along /carry_plan as one rigid shape, call
#  /rN/detach_payload on the hexagon's centre and back straight away.
#
#   1. Paste in your Task 2A code      (the FROM TASK 2A section)
#   2. Copy in path_to_traj()          (from Task 2B)
#   3. Write request_attach() and request_detach()   (NEVER block on the reply)
#
#  Run:  ros2 launch hb_description task3b.launch.py
#        ros2 run task_3b path_planner
#        ros2 run task_3b path_follower
#
###############################################################################

import math
import threading

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from hb_description.srv import AttachPayload, DetachPayload

################# ADD EXTRA IMPORTS / GLOBALS HERE ############

# Control tuning (Task 2A controller, retuned for following a moving target).
# A pure P term lags a target moving at v by v / KP; LEAD (below) cancels most
# of it. The yaw loop is stiffer than in 2A: r1 has to turn 180 deg and r2
# 90 deg on the way in, and must be square to the payload when it docks.
V_MAX = 0.7                    # m/s, cap on the speed
W_MAX = 3.0                     # rad/s, cap on the yaw rate
KP, KI, KD = 6.5, 2.0, 0.08    # position PID gains
KYAW, KIYAW, KDYAW = 4.0, 0.0, 0.08    # heading PID gains
INTEGRAL_ZONE = 0.03            # m, integrate position error only inside this
MAX_INTEGRAL = 0.05             # anti-windup clamp on the position integral

###############################################################


# ----------------------------------------------------- follower (given)
ROBOTS = ("r1", "r2", "r3")   # one node per robot
PAYLOAD_NAME = "payload_gamma"
DOCK_YAW = {"r1": math.pi / 2, "r2": 0.0, "r3": -math.pi / 2}   # facing the payload
CONTROL_HZ = 50.0
LEAD = 0.18                   # s, aim this far ahead on a path
DOCK_TOL = 0.005              # m, "on the slot"
DOCK_PRELOAD = 0.006          # m, aim this far INTO the slot: the slot leaves a 3 mm
                              # gap and the robot settles ~1 mm short, so the tip lands
                              # on the crate's face
SLOT_DIST = {"r1": 0.1827, "r2": 0.1796, "r3": 0.1827}   # m, robot centre to payload
                              # centre when the solenoid's tip is on the crate's face
DOCK_YAW_TOL = math.radians(5.0)
SETTLE = 0.3                  # s, past the plan's end, accept 2x the tolerance
                              # (a camera error of a few mm can leave the slot
                              # inside the crate: let the attach rule decide)
CREEP_SPEED = 0.05            # m/s, after "not close enough" the slot creeps in
                              # (or out, after "pushed into it") at this speed ...
CREEP_RETRY = 0.05            # s, ... and the attach is retried this often, so the
                              # tip stops within ~3 mm of first touching the face
MAX_CREEP = 0.040             # m, never creep further than this from the planned slot
MAX_STEP = 0.025              # m, largest jump in when the reply's `distance` says how
                              # far short the robot is
RETRY_AFTER = 0.40            # s, retry interval when not creeping (heading refusals)
CARRY_TOL = 0.010             # m, "the payload is on the destination"
CARRY_V_MAX = 0.5            # m/s, speed cap while carrying
BACK_DIST = 0.08              # m, how far each robot backs off the payload
BACK_TOL = 0.015              # m
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
## Paste your Task 2A functions and GoToPoint here, unchanged.##
###############################################################

##############  ADD YOUR CODE HERE  ##############

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
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    """Angle -> the same angle in (-pi, pi]."""
    return math.pi - (math.pi - a) % (2.0 * math.pi)


def to_body(vx_a, vy_a, yaw):
    """Arena-frame velocity -> body-frame velocity (vx, vy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * vx_a + s * vy_a, -s * vx_a + c * vy_a)


class GoToPoint:
    """From Task 2A: position PID in the arena frame + heading PID, converted
    to wheel speeds. The TUNED gains are the defaults."""

    def __init__(self, v_max=V_MAX, w_max=W_MAX, kp=KP, ki=KI, kd=KD,
                 kyaw=KYAW, kiyaw=KIYAW, kdyaw=KDYAW, tol=0.012, dt=1.0 / CONTROL_HZ):
        self.v_max, self.w_max = v_max, w_max
        self.kp, self.ki, self.kd = kp, ki, kd
        self.kyaw, self.kiyaw, self.kdyaw = kyaw, kiyaw, kdyaw
        self.tol, self.dt = tol, dt
        self.int_x = self.int_y = self.int_yaw = 0.0
        self.prev_x_err = self.prev_y_err = self.prev_yaw_err = 0.0
        self._first = True                # suppress the derivative kick on tick 1

    def step(self, pose, target, hold_yaw):
        """pose (x, y, yaw), target (x, y), hold_yaw -> (wheels, distance)."""
        x, y, yaw = pose
        ex, ey = target[0] - x, target[1] - y
        dist = math.hypot(ex, ey)

        if dist <= self.tol:
            self.int_x = self.int_y = 0.0
            vx_a = vy_a = 0.0
        else:
            if dist <= INTEGRAL_ZONE:
                self.int_x = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.int_x + ex * self.dt))
                self.int_y = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.int_y + ey * self.dt))
            else:
                self.int_x = self.int_y = 0.0
            d_x = 0.0 if self._first else (ex - self.prev_x_err) / self.dt
            d_y = 0.0 if self._first else (ey - self.prev_y_err) / self.dt
            vx_a = self.kp * ex + self.ki * self.int_x + self.kd * d_x
            vy_a = self.kp * ey + self.ki * self.int_y + self.kd * d_y
            speed = math.hypot(vx_a, vy_a)
            if speed > self.v_max:
                vx_a *= self.v_max / speed
                vy_a *= self.v_max / speed
        self.prev_x_err, self.prev_y_err = ex, ey

        yaw_err = wrap(hold_yaw - yaw)
        d_yaw = 0.0 if self._first else wrap(yaw_err - self.prev_yaw_err) / self.dt
        self.prev_yaw_err = yaw_err
        wz = self.kyaw * yaw_err + self.kiyaw * self.int_yaw + self.kdyaw * d_yaw
        wz = max(-self.w_max, min(self.w_max, wz))
        self._first = False

        vx_b, vy_b = to_body(vx_a, vy_a, yaw)
        return body_to_wheels(vx_b, vy_b, wz), dist

##################################################


######################### PATH ################################

def path_to_traj(msg):
    """nav_msgs/Path -> [(t, x, y), ...], t = pose stamp - path stamp, in s."""
    ##############  ADD YOUR CODE HERE  ##############
    t0 = Time.from_msg(msg.header.stamp)
    return [((Time.from_msg(p.header.stamp) - t0).nanoseconds * 1e-9,
             p.pose.position.x, p.pose.position.y)
            for p in msg.poses]
    ##################################################


def position_at(traj, t):
    """Given. Where the path says the robot should be at time t."""
    if t <= traj[0][0]:
        return traj[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(traj, traj[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return traj[-1][1:]


######################### SERVICES ############################

def request_attach(client, payload_name, on_reply):
    """Send /rN/attach_payload for `payload_name` WITHOUT waiting; `on_reply`
    gets the reply (or None if the call failed). False if the service is not
    up yet (try again on a later tick)."""
    ##############  ADD YOUR CODE HERE  ##############
    if not client.service_is_ready():
        return False
    req = AttachPayload.Request()
    req.payload_name = payload_name
    future = client.call_async(req)
    future.add_done_callback(lambda f: on_reply(f.result() if f.exception() is None else None))
    return True
    ##################################################


def request_detach(client, payload_name, on_reply):
    """Same as request_attach(), for /rN/detach_payload."""
    ##############  ADD YOUR CODE HERE  ##############
    if not client.service_is_ready():
        return False
    req = DetachPayload.Request()
    req.payload_name = payload_name
    future = client.call_async(req)
    future.add_done_callback(lambda f: on_reply(f.result() if f.exception() is None else None))
    return True
    ##################################################


######################### TEAM (given) ########################

class Team:
    """Given. What the three nodes share in this one process: who has docked,
    attached, arrived, let go and finished, plus the carry's clock and plan."""

    def __init__(self):
        self.lock = threading.Lock()
        self.flags = {k: set() for k in ("docked", "attached", "arrived", "detached", "done")}
        self.start = None                 # rclpy Time, the approach's shared start
        self.carry = None                 # [(t, x, y), ...], the payload's centre
        self.carry_start = None           # rclpy Time, when all three attached
        self.logger = None

    def mark(self, flag, robot):
        """Set `flag` for `robot`; True exactly once, when the third robot sets it."""
        with self.lock:
            before = len(self.flags[flag])
            self.flags[flag].add(robot)
            return before < len(ROBOTS) == len(self.flags[flag])

    def unmark(self, flag, robot):
        with self.lock:
            self.flags[flag].discard(robot)

    def all(self, flag):
        with self.lock:
            return len(self.flags[flag]) == len(ROBOTS)


######################### NODE (given) ########################

APPROACH, DOCKED, ATTACHED, CARRY, BACK, DONE = range(6)


class RobotNode(Node):
    """Drives one robot: approach -> dock -> attach -> carry -> detach -> back off.
    Nodes: task_3b_r1, _r2, _r3."""

    def __init__(self, robot, team):
        super().__init__(f"task_3b_{robot}")
        self.robot, self.team = robot, team
        self.pose = self.traj = None
        self.state, self.pending, self.tries, self.retry_at = APPROACH, False, 0, None
        self.target = self.hold = self.offset = self.slot = None
        self.creep = 0.0                        # +1 creeping in, -1 out, 0 holding
        self.ctl = GoToPoint(tol=0.002)
        self.create_subscription(Odometry, f"/{robot}/odom", self.on_odom, 10)
        self.create_subscription(Path, f"/{robot}/plan", self.on_path, LATCHED)
        self.create_subscription(Path, "/carry_plan", self.on_carry, LATCHED)
        self.pub = self.create_publisher(Float64MultiArray, f"/{robot}/wheel_commands", 10)
        self.attach_cli = self.create_client(AttachPayload, f"/{robot}/attach_payload")
        self.detach_cli = self.create_client(DetachPayload, f"/{robot}/detach_payload")
        self.create_timer(1.0 / CONTROL_HZ, self.tick)

    # ------------------------------------------------ inputs
    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(q.w, q.x, q.y, q.z))

    def on_path(self, msg):
        if self.traj is None and msg.poses:
            self.traj = path_to_traj(msg)
            self.start = Time.from_msg(msg.header.stamp)
            yaw = DOCK_YAW[self.robot]
            self.target = (self.traj[-1][1] + DOCK_PRELOAD * math.cos(yaw),
                           self.traj[-1][2] + DOCK_PRELOAD * math.sin(yaw))
            self.slot = self.target
            with self.team.lock:
                if self.team.start is None:
                    self.team.start = self.start
            self.get_logger().info("got my path: %d poses, dock at t=%.1f s"
                                   % (len(msg.poses), self.traj[-1][0]))

    def on_carry(self, msg):
        with self.team.lock:
            if self.team.carry is None and msg.poses:
                self.team.carry = path_to_traj(msg)

    def send(self, wheels):
        self.pub.publish(Float64MultiArray(data=[float(w) for w in wheels]))

    def since(self, t0):
        return (self.get_clock().now() - t0).nanoseconds * 1e-9

    def drive(self, target, yaw):
        wheels, dist = self.ctl.step(self.pose, target, yaw)
        self.send(wheels)
        return dist

    def phase(self, state, ctl=None):
        self.state, self.pending, self.tries, self.retry_at = state, False, 0, None
        self.creep = 0.0
        if ctl is not None:
            self.ctl = ctl                      # fresh integral / derivative per phase

    # ------------------------------------------------ service replies
    def on_attach(self, reply):
        self.pending = False
        if reply is not None and reply.success:
            self.get_logger().info("%s attached (%.1f mm)"
                                   % (self.robot, 1000.0 * getattr(reply, "distance", 0.0)))
            self.phase(ATTACHED)
            if self.team.mark("attached", self.robot):
                self.team.carry_start = self.get_clock().now()
                self.team.logger.info("all three attached: carrying %s" % PAYLOAD_NAME)
            return
        why = reply.message if reply is not None else "no reply"
        self.tries += 1
        # The camera puts the payload a few mm off, so the slot can be short of
        # (or inside) the crate's face. Close the gap on what the RULE says, not
        # on the camera: creep towards the face, retrying every CREEP_RETRY, and
        # if the reply's `distance` says the robot is well short, jump most of
        # that shortfall first. A facing refusal just retries once settled.
        w = why.lower()
        yaw, jump = DOCK_YAW[self.robot], 0.0
        if any(k in w for k in ("not close", "drive up", "too far", "farther", "further")):
            self.creep = 1.0
            d = getattr(reply, "distance", 0.0) or 0.0
            short = d - SLOT_DIST[self.robot]
            if math.isfinite(short) and 0.006 < short < 0.10:
                jump = min(MAX_STEP, short - 0.003)     # leave the last mm to the creep
        elif any(k in w for k in ("into", "pushed", "too close", "overlap", "back off")):
            self.creep = -1.0
        else:
            self.creep = 0.0
        if jump:
            self.target = (self.target[0] + jump * math.cos(yaw),
                           self.target[1] + jump * math.sin(yaw))
        if self.tries == 1 or self.tries % 10 == 0:
            self.get_logger().warn("%s attach refused (%d): %s -> %s"
                                   % (self.robot, self.tries, why,
                                      {1.0: "creeping in", -1.0: "backing out", 0.0: "retrying"}[self.creep]))
        self.retry_at = self.get_clock().now()

    def on_detach(self, reply):
        self.pending = False
        if reply is not None and reply.success:
            yaw = self.hold
            self.target = (self.pose[0] - BACK_DIST * math.cos(yaw),
                           self.pose[1] - BACK_DIST * math.sin(yaw))
            self.phase(BACK, GoToPoint(tol=0.004))
            if self.team.mark("detached", self.robot):
                self.team.logger.info("all three let go of %s" % PAYLOAD_NAME)
            return
        self.get_logger().warn("%s detach refused: %s"
                               % (self.robot, reply.message if reply is not None else "no reply"))
        self.retry_at = self.get_clock().now()

    def ready_to_retry(self):
        """No call in flight, and the first try or the retry interval is up."""
        if self.pending:
            return False
        if self.retry_at is None:
            return True
        creeping = self.state == DOCKED and self.creep != 0.0
        return self.since(self.retry_at) >= (CREEP_RETRY if creeping else RETRY_AFTER)

    # ------------------------------------------------ control loop
    def tick(self):
        if self.traj is None or self.pose is None or self.state == DONE:
            self.send((0.0, 0.0, 0.0))
            return

        if self.state == APPROACH:
            t = self.since(self.start)
            if t < 0:
                self.send((0.0, 0.0, 0.0))
                return
            yaw = DOCK_YAW[self.robot]
            late = t - self.traj[-1][0]
            aim = self.target if late + LEAD >= 0 else position_at(self.traj, t + LEAD)
            dist = self.drive(aim, yaw)
            tol = DOCK_TOL if late < SETTLE else 2 * DOCK_TOL
            if late >= 0 and dist < tol and abs(wrap(yaw - self.pose[2])) < DOCK_YAW_TOL:
                self.get_logger().info("%s docked (%.1f mm out) at t=%.1f s"
                                       % (self.robot, 1000.0 * dist, t))
                self.phase(DOCKED)
                self.team.mark("docked", self.robot)

        elif self.state == DOCKED:
            # hold on the slot; the attach waits for all three (attaching nudges the payload)
            yaw = DOCK_YAW[self.robot]
            if self.creep and not self.pending:
                step = self.creep * CREEP_SPEED / CONTROL_HZ
                nxt = (self.target[0] + step * math.cos(yaw), self.target[1] + step * math.sin(yaw))
                if math.dist(nxt, self.slot) <= MAX_CREEP:
                    self.target = nxt
                elif self.creep:
                    self.get_logger().error("%s crept %.0f mm and still cannot attach: check "
                                            "find_payload()" % (self.robot, 1000 * MAX_CREEP))
                    self.creep = 0.0
            self.drive(self.target, yaw)
            if self.team.all("docked") and self.ready_to_retry():
                self.pending = request_attach(self.attach_cli, PAYLOAD_NAME, self.on_attach)

        elif self.state == ATTACHED:
            # welded: stop pushing until the other two hold it too
            self.send((0.0, 0.0, 0.0))
            if self.team.carry_start is not None and self.team.carry is not None:
                carry = self.team.carry
                # every robot keeps its offset from the carry's start: the shape holds
                self.offset = (self.pose[0] - carry[0][1], self.pose[1] - carry[0][2])
                self.hold = self.pose[2]
                self.phase(CARRY, GoToPoint(v_max=CARRY_V_MAX, tol=0.002))

        elif self.state == CARRY:
            carry, t = self.team.carry, self.since(self.team.carry_start)
            cx, cy = position_at(carry, t + LEAD)
            self.drive((cx + self.offset[0], cy + self.offset[1]), self.hold)
            end, late = carry[-1], t - carry[-1][0]
            off = math.hypot(self.pose[0] - end[1] - self.offset[0],
                             self.pose[1] - end[2] - self.offset[1])
            if late >= 0 and off < (CARRY_TOL if late < SETTLE else 2 * CARRY_TOL):
                self.team.mark("arrived", self.robot)
            if self.team.all("arrived") and self.ready_to_retry():
                self.pending = request_detach(self.detach_cli, PAYLOAD_NAME, self.on_detach)

        elif self.state == BACK:
            if self.drive(self.target, self.hold) < BACK_TOL:
                self.send((0.0, 0.0, 0.0))
                self.phase(DONE)
                if self.team.mark("done", self.robot):
                    self.team.logger.info("task 3B complete in %.1fs" % self.since(self.team.start))


def main():
    rclpy.init()
    team = Team()
    nodes = [RobotNode(r, team) for r in ROBOTS]
    team.logger = nodes[0].get_logger()
    # service replies arrive on their own callbacks: give the executor threads to spare
    executor = MultiThreadedExecutor(num_threads=2 * len(ROBOTS))
    for n in nodes:
        executor.add_node(n)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for n in nodes:
            try:
                n.send((0.0, 0.0, 0.0))
            except Exception:
                pass
            n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
