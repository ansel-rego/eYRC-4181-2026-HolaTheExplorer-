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
# Filename:         path_planner.py
# Functions:        read_frame, crop_arena, pixel_to_odom, find_payload, find_plates,
#                   wrap_angle, docking_slots, crate_extent, frame_to_occupancy,
#                   clear_circle, clear_circle_value, remove_drivable, planning_grid,
#                   carry_grid, to_cell, to_world, nearest_free, _grow, _astar, _line_clear,
#                   _straighten, _leg, _best_leg, _distance_map, _time_points, _shift,
#                   plan_route, plan_carry, _rotate, _straight, plan_mission, _eta,
#                   _plan_sides, _save_debug, PathPlanner.__init__,
#                   PathPlanner._grab, PathPlanner._odom_cb, PathPlanner.to_path_msg, PathPlanner.tick, main
# Global variables: PAYLOAD_NAME, DESTINATION, SLOT, START, CRATE_HALF, CRATE_RADIUS,
#                   SOLENOID_TIP, TOP_LOW, TOP_HIGH, MIN_TOP_PX, CAMERA_HEIGHT,
#                   PAYLOAD_TOP_HEIGHT, PALE_V_MIN, PALE_S_MAX, DARK_V_MAX, PAINT_S_MIN, PAINT_V_MIN,
#                   PAINT_GROW_PX, PAINT_CELL_FRACTION, THIN_CELLS, MAX_FILL_CELLS, PADS,
#                   PAYLOAD_RADIUS, CRATE_CLEAR, START_CLEAR, ROBOT_CLEAR, GOAL_CLEAR,
#                   DEBUG_IMAGE, EXTRA_CLEAR, MAX_DETOUR, MAX_END_SHIFT, COMFORT,
#                   COMFORT_WEIGHT, STRAIGHT_CLEAR, SLOW_SPEED, SLOW_BELOW, LOOK_AHEAD, STEP,
#                   PREDOCKS, LINE_UP_STOP, DOCK_SPEED, CARRY_SPEED, CARRY_SLOW_SPEED,
#                   CARRY_SHAPES, ROBOT_MARGINS, TRY_OTHER_SIDES, DOCK_GAP, PLATES_ONLY, PLATE_R, PLATE_V_MAX,
#                   PLATE_MIN_STRENGTH, CARRY_SWEEPS, CARRY_SWEEP_STEPS,
#                   ETA_ROBOT, ETA_DRAG


############################ WHAT THIS NODE DOES ##############################
#
#  r2 fetches payload_beta (teal top) and leaves it on the hexagon's centre.
#  This node plans everything ONCE from one camera frame and publishes:
#
#    /r2/plan         robot centre: start circle -> docking point
#                     (header.stamp = the moment it sets off)
#    /carry_plan      CRATE centre: where it was found -> DESTINATION
#                     (times count from the moment the robot has attached)
#    /r2/return_plan  robot centre: docking point at the hexagon -> START
#                     (times count from the moment it lets go)
#
#  All three are latched. Every pose carries the docking heading in its
#  orientation, so path_follower.py knows which way to face.
#
#  Frame (/r2/odom): origin at the arena's TOP-LEFT corner, x right, y DOWN.
#
#  Run:  ros2 launch hb_description task3a.launch.py
#        ros2 run task_3a path_planner
#        ros2 run task_3a path_follower
#
###############################################################################

import heapq
import threading
import math

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

################# TASK 3A: THE PAYLOAD AND WHERE IT GOES ######

PAYLOAD_NAME = "payload_beta"
DESTINATION = (1.2192, 1.2192)       # m, the hexagon's centre: the crate's centre ends here
SLOT = (0.1798, 0.0)                 # m, robot centre from crate centre when docked. Its
                                     # LENGTH is used as is; its DIRECTION only says which
                                     # side is preferred (right of the crate, as in the
                                     # repo's ReadMe): the robot docks square on to the
                                     # crate's plate nearest that direction (find_plates)
START = (1.2192, 2.2616)             # m, r2's start circle: it drives back here
CRATE_HALF = (0.1088, 0.1089)        # m, the crate's half sizes
CRATE_RADIUS = math.hypot(*CRATE_HALF)   # m, ~0.154: covers the crate at any rotation
SOLENOID_TIP = 0.070                 # m, solenoid tip ahead of the chassis centre

# find_payload(): the teal (verdigris) top is far more saturated and brighter
# than its duller teal pad, so S and V separate them
TOP_LOW, TOP_HIGH = (75, 140, 215), (100, 255, 255)   # HSV
MIN_TOP_PX = 150                     # px, smallest blob that can be the crate's top
CAMERA_HEIGHT = 2.6                  # m, overhead camera above the floor (fovy 53.42 deg)
PAYLOAD_TOP_HEIGHT = 0.05            # m, the top is raised: the camera sees it pushed out

###############################################################

################# FROM TASK 2B (remove_drivable / plan_route) ##

PAINT_S_MIN, PAINT_V_MIN = 90, 90    # HSV thresholds for "painted"
PAINT_GROW_PX = 3                    # px, cover the anti-aliased edge of paint
PALE_V_MIN, PALE_S_MAX = 165, 70     # HSV, pale paint (the cyan swirl funnel)
DARK_V_MAX = 140                     # V below this is shade: a pillar or a station
PAINT_CELL_FRACTION = 0.20           # a cell is paint if this much of it is
THIN_CELLS = 5                       # cells, obstacles thinner than this are paint outlines
MAX_FILL_CELLS = 1500                # only fill holes in blobs smaller than this
PADS = [(0.534, 0.425), (1.904, 0.425),      # m, the four coloured pads; a payload
        (0.316, 1.221), (2.126, 1.221)]      # stands on each
PAYLOAD_RADIUS = 0.16                # m, a payload on its pad (crate half-diagonal 0.154)
CRATE_CLEAR = 0.17                   # m, freed around payload_beta when it is carried
START_CLEAR = 0.13                   # m, around each start circle (+ its dots)
ROBOT_CLEAR = 0.10                   # m, around a robot
GOAL_CLEAR = 0.114                   # m, around the destination
DEBUG_IMAGE = "/tmp/task3a_plan.png" # the grid and the three plans, to look at

EXTRA_CLEAR = (0.12, 0.08, 0.04, 0.0)   # m, try extra clearance first, fall back to none
MAX_DETOUR = 1.00                    # extra clearance may make a route this much longer
MAX_END_SHIFT = 0.15                 # m, how far a start/goal may move onto a free cell
COMFORT = 0.15                       # m, the search prefers to stay this far from obstacles
COMFORT_WEIGHT = 1.0                 # how strongly (0 = shortest path only)
STRAIGHT_CLEAR = 0.04                # m, shortcuts keep this far from the grown obstacles
SLOW_SPEED = 0.70                    # m/s, speed right at the edge of the grown obstacles
SLOW_BELOW = 0.04                    # m, slow down when closer than this to them
LOOK_AHEAD = 0.15                    # m, start slowing this far before a tight spot
STEP = 0.04                          # m, the path is timed in pieces this long

# new in 3A
LINE_UP_STOP = 0.1                   # s, pause there so the heading settles
DOCK_SPEED = 0.70                    # m/s, the last few cm onto the crate
CARRY_SPEED = 0.70                   # m/s, crate speed while carrying
CARRY_SLOW_SPEED = 0.70              # m/s, carrying close to an obstacle
CARRY_SHAPES = ((False, 0.03), (False, 0.015),      # (crate as its box?, clearance in m)
                (True, 0.03), (True, 0.015))       # around robot + crate while carrying:
                                     # a disc that covers the crate at any rotation
                                     # first, its 0.218 m box only if that is too wide
PREDOCKS = (0.33, 0.29)              # m, line-up distances to try
ROBOT_MARGINS = (0.030, 0.020, 0.012)   # m, robot clearance, tightened only if
                                     # a layout leaves no route with the full margin
TRY_OTHER_SIDES = True               # if the preferred plate cannot be reached (e.g. it
                                     # faces a wall), dock on another of the crate's plates
# the crate is a hexagonal prism with three dark plates on alternate faces (120 deg
# apart). The solenoid must touch a plate (the expected-output video docks on the
# upper-right one), so the robot docks along a plate's direction, red arrow
# pointing at the crate's centre, its solenoid tip DOCK_GAP short of the crate's
# collision box (crate_extent())
DOCK_GAP = 0.001                     # m, solenoid tip to the crate when docked
CARRY_SWEEPS = tuple(math.radians(a) for a in (45.0, 30.0, 15.0, 0.0))   # rad, twist of
                                     # robot + crate to allow for while carrying, widest
                                     # first (it is told to the follower in /carry_plan)
CARRY_SWEEP_STEPS = 5
ETA_ROBOT = 0.55                     # m/s, robot alone, on average (choosing a plate)
ETA_DRAG = 0.15                      # m/s, with the crate on
PLATES_ONLY = False                  # True: never dock off a plate. False: if no plate
                                     # can be reached on a layout, dock square on to a
                                     # face of the crate's box instead (and warn)
PLATE_R = (0.085, 0.128)             # m, ring around the crate's centre the plates sit in
PLATE_V_MAX = 100                    # HSV V, the plates are dark
PLATE_MIN_STRENGTH = 0.10            # 3-fold pattern this clear, or plates not found

###############################################################


# ----------------------------------------------------- arena (given)
ROBOTS = ("r2",)
START_CIRCLES = [(0.8969, 2.2616), (1.2192, 2.2616), (1.5415, 2.2616)]
ARENA_SIZE = 2.4384                  # m
ROBOT_RADIUS = 0.084                 # m
SAFETY_MARGIN = 0.030                # m, kept clear on top of the robot radius

# ----------------------------------------------------- camera (given)
CAMERA_URL = "http://127.0.0.1:8080/stream"
ARENA_LEFT, ARENA_TOP, ARENA_PIXELS = 304, 24, 671       # floor square in the image
GRID_CELLS = 244                     # occupancy grid is GRID_CELLS x GRID_CELLS
CELL_SIZE = ARENA_SIZE / GRID_CELLS  # m, about 1 cm
SAND_LOW, SAND_HIGH = (10, 40, 170), (30, 120, 255)      # sand colour, HSV

# ----------------------------------------------------- planning (tunable)
PLAN_CELL = 0.02                     # m, planning grid cell (2B used 0.04: too coarse to
                                     # find the narrow gaps some random layouts leave)
PLAN_SPEED = 0.7                    # m/s, the speed the robot's paths are timed at
GOAL_STOP = 0.0                      # s, stop this long on a goal
START_DELAY = 0.2                    # s, from publishing to setting off

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


######################### CAMERA ##############################

def read_frame(url=CAMERA_URL):
    """One BGR frame from the overhead camera."""
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame from %s -- is task3a.launch.py running?" % url)
    return frame


def crop_arena(frame):
    """The arena floor only, ARENA_PIXELS square."""
    return frame[ARENA_TOP:ARENA_TOP + ARENA_PIXELS, ARENA_LEFT:ARENA_LEFT + ARENA_PIXELS]


def pixel_to_odom(px, py, height=0.0):
    """Arena-crop pixel -> /odom metres. A point `height` m above the floor is
    seen pushed away from the point under the camera (the arena's centre) by
    CAMERA_HEIGHT / (CAMERA_HEIGHT - height): pull it back by that much."""
    s = ARENA_SIZE / ARENA_PIXELS
    x, y = (px + 0.5) * s, (py + 0.5) * s
    c = ARENA_SIZE / 2.0
    k = (CAMERA_HEIGHT - height) / CAMERA_HEIGHT
    return c + (x - c) * k, c + (y - c) * k


def find_payload(frame):
    """Camera frame -> payload_beta's centre (x, y) in /odom metres, or None.

    1. mask the teal top by colour (high S and V: its pad is duller),
    2. take the biggest blob, fill it (the top's printed mark leaves holes)
       and get its area centroid,
    3. pixel -> metres, pulled back towards the point under the camera because
       the top is raised. (Not the pad's centre: the crate need not sit
       exactly in the middle of its pad.)"""
    crop = crop_arena(frame)
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TOP_LOW, TOP_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_TOP_PX]
    if not contours:
        return None
    top = max(contours, key=cv2.contourArea)
    m = cv2.moments(top)
    if m["m00"] < 1e-6:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    xy = pixel_to_odom(cx, cy, PAYLOAD_TOP_HEIGHT)

    return float(xy[0]), float(xy[1])


def find_plates(frame, crate):
    """Directions (rad, /odom frame) from the crate's centre to its three
    docking plates, or None if they cannot be made out.

    Dark pixels in a ring around the crate are the plates (and the body's
    grey rim, which is 6-fold symmetric). A 3-fold circular mean,
    angle(sum exp(3j*a)) / 3, picks out the plates and cancels the rim."""
    crop = crop_arena(frame)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, c = ARENA_PIXELS / ARENA_SIZE, ARENA_SIZE / 2.0
    # where the camera sees the crate's sides: half way up, pushed outwards
    k = CAMERA_HEIGHT / (CAMERA_HEIGHT - PAYLOAD_TOP_HEIGHT / 2.0)
    cx = (c + (crate[0] - c) * k) * s - 0.5
    cy = (c + (crate[1] - c) * k) * s - 0.5
    ys, xs = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    r = np.hypot(xs - cx, ys - cy) / s
    m = (r > PLATE_R[0]) & (r < PLATE_R[1]) & (hsv[:, :, 2] < PLATE_V_MAX)
    if m.sum() < 30:
        return None
    z = np.exp(3j * np.arctan2(ys[m] - cy, xs[m] - cx)).sum()
    if abs(z) / m.sum() < PLATE_MIN_STRENGTH:
        return None
    a0 = float(np.angle(z)) / 3.0
    return [wrap_angle(a0 + i * 2.0 * math.pi / 3.0) for i in range(3)]


def wrap_angle(a):
    """Angle -> (-pi, pi]."""
    return math.pi - (math.pi - a) % (2.0 * math.pi)


######################### OCCUPANCY GRID ######################

def frame_to_occupancy(frame):
    """Frame -> grid[row, col] = grid[y, x], True = obstacle. Only sand is free."""
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    obstacles = cv2.bitwise_not(cv2.inRange(hsv, SAND_LOW, SAND_HIGH))
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(obstacles, contours, -1, 255, cv2.FILLED)
    small = cv2.resize(obstacles, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA)
    return small > 0


def clear_circle(occupied, xy, radius):
    """Free every cell within `radius` m of point xy (in place)."""
    clear_circle_value(occupied, xy, radius, False)


def clear_circle_value(occupied, xy, radius, value):
    """Set every cell within `radius` m of xy to `value` (in place)."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = value


def remove_drivable(frame, occupied, robots, goals, payload_pads=PADS):
    """Task 2B. Free what a robot may drive over; return the grid.
    Keeps the pillars, the stations and a payload on each of `payload_pads`."""
    occupied = occupied.copy()

    # 1. painted floor: clearly coloured pixels are floor paint, not obstacles
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    paint = cv2.inRange(hsv, (0, PAINT_S_MIN, PAINT_V_MIN), (179, 255, 255))
    paint = cv2.dilate(paint, np.ones((PAINT_GROW_PX, PAINT_GROW_PX), np.uint8))
    frac = cv2.resize(paint.astype(np.float32) / 255.0, (GRID_CELLS, GRID_CELLS),
                      interpolation=cv2.INTER_AREA)
    occupied[frac > PAINT_CELL_FRACTION] = False

    # 1b. pale paint (the cyan swirl funnel is barely saturated): light, greyish
    #     cells with no dark pixels in them. Pillars and stations always have
    #     dark, shaded pixels, so they stay
    pale = cv2.inRange(hsv, (0, 0, PALE_V_MIN), (179, PALE_S_MAX, 255)).astype(np.float32) / 255.0
    dark = (hsv[:, :, 2] < DARK_V_MAX).astype(np.float32)
    pale_frac = cv2.resize(pale, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA)
    dark_near = cv2.dilate(cv2.resize(dark, (GRID_CELLS, GRID_CELLS),
                                      interpolation=cv2.INTER_AREA), np.ones((5, 5), np.uint8))
    occupied[(pale_frac > 0.5) & (dark_near < 0.02)] = False

    # 2. drop thin outlines, then fill the holes left where a payload's
    #    coloured top was freed (small blobs only)
    mask = cv2.morphologyEx(occupied.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            np.ones((THIN_CELLS, THIN_CELLS), np.uint8))
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < MAX_FILL_CELLS:
            cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    occupied = mask > 0

    # 3. a payload stands on each pad: put it back as a solid disc
    for xy in payload_pads:
        clear_circle_value(occupied, xy, PAYLOAD_RADIUS, True)

    # 4. start circles, robots, goals
    for xy in START_CIRCLES:
        clear_circle(occupied, xy, START_CLEAR)
    for xy in robots:
        clear_circle(occupied, xy, ROBOT_CLEAR)
    for xy in goals:
        clear_circle(occupied, xy, GOAL_CLEAR)
    return occupied


######################### PLANNING GRIDS ######################

def planning_grid(occupied, margin=None):
    """Grid -> blocked[i, j] = blocked[x, y] (x first) of PLAN_CELL cells, for
    the ROBOT's centre: obstacles grow by ROBOT_RADIUS + SAFETY_MARGIN. The
    arena's edge is a wall."""
    pad = ROBOT_RADIUS + (SAFETY_MARGIN if margin is None else margin)
    k = int(math.ceil(pad / CELL_SIZE))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    inflated = cv2.dilate(occupied.astype(np.uint8), disc) > 0

    n = int(round(ARENA_SIZE / PLAN_CELL))
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T        # [row=y, col=x] -> [i=x, j=y]
    near_wall = (centre < pad) | (centre > ARENA_SIZE - pad)
    blocked[near_wall, :] = True
    blocked[:, near_wall] = True
    return blocked


def carry_grid(occupied, slot, margin, square=False, sweep=0.0):
    """Grid -> blocked[i, j] for the CRATE's centre while the robot carries it.

    The robot holds its heading, so robot + crate are one fixed shape: the
    crate (a disc of CRATE_RADIUS, safe at any rotation; or, if `square`, its
    CRATE_HALF box, axis-aligned) at the centre and the robot
    (a disc of ROBOT_RADIUS) at `slot` from it, both grown by `margin`. A
    crate centre is blocked if that shape, placed on it, touches an obstacle
    or leaves the arena. cv2.dilate is a correlation (dst(p) = max of
    src(p + k) over the kernel's cells k), so the kernel is the shape itself."""
    rc = (CRATE_RADIUS + margin) / CELL_SIZE
    rr = (ROBOT_RADIUS + margin) / CELL_SIZE
    sx, sy = slot[0] / CELL_SIZE, slot[1] / CELL_SIZE
    k = int(math.ceil(max(rc, math.hypot(sx, sy) + rr))) + 1
    kernel = np.zeros((2 * k + 1, 2 * k + 1), np.uint8)
    if square:
        hx, hy = (CRATE_HALF[0] + margin) / CELL_SIZE, (CRATE_HALF[1] + margin) / CELL_SIZE
        cv2.rectangle(kernel, (int(round(k - hx)), int(round(k - hy))),
                      (int(round(k + hx)), int(round(k + hy))), 1, -1)
    else:
        cv2.circle(kernel, (k, k), int(round(rc)), 1, -1)
    # the robot may twist up to `sweep` either way while it carries (the
    # crate's drag turns it; the follower only fights back near the limit)
    steps = CARRY_SWEEP_STEPS if sweep > 0 else 0
    for i in range(-steps, steps + 1):
        a = sweep * i / max(1, steps)
        rx, ry = _rotate((sx, sy), a)
        cv2.circle(kernel, (int(round(k + rx)), int(round(k + ry))), int(round(rr)), 1, -1)

    # outside the arena is an obstacle too: pad the grid with a solid border
    src = np.pad(occupied.astype(np.uint8), k, constant_values=1)
    inflated = (cv2.dilate(src, kernel) > 0)[k:-k, k:-k]

    n = int(round(ARENA_SIZE / PLAN_CELL))
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    return inflated[np.ix_(fine, fine)].T


def to_cell(p):
    """Point (x, y) in m -> planning cell (i, j)."""
    n = int(round(ARENA_SIZE / PLAN_CELL))
    return (min(max(int(p[0] / PLAN_CELL), 0), n - 1), min(max(int(p[1] / PLAN_CELL), 0), n - 1))


def to_world(c):
    """Planning cell (i, j) -> its centre (x, y) in m."""
    return ((c[0] + 0.5) * PLAN_CELL, (c[1] + 0.5) * PLAN_CELL)


def nearest_free(blocked, p, max_r=0.30):
    """The free cell closest to point p, or None within max_r m."""
    c = to_cell(p)
    if not blocked[c]:
        return c
    free = [(i, j) for i in range(blocked.shape[0]) for j in range(blocked.shape[1])
            if not blocked[i, j] and abs(i - c[0]) * PLAN_CELL <= max_r
            and abs(j - c[1]) * PLAN_CELL <= max_r]
    return min(free, key=lambda q: math.dist(to_world(q), p)) if free else None


######################### SEARCH (Task 2B) ####################

def _grow(blocked, metres):
    """Blocked grid grown by `metres` (extra clearance)."""
    cells = int(round(metres / PLAN_CELL))
    if cells <= 0:
        return blocked
    k = 2 * cells + 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(blocked.astype(np.uint8), disc) > 0


def _distance_map(blocked):
    """m from each cell to the nearest blocked cell or the edge."""
    padded = np.pad(~blocked, 1, constant_values=False).astype(np.uint8)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * PLAN_CELL


def _astar(blocked, s, g):
    """8-connected A* from cell s to cell g, no corner cutting, cells next to
    obstacles cost more. List of cells or None."""
    n = blocked.shape[0]
    blk = blocked.tolist()
    away = _distance_map(blocked).tolist()
    root2 = math.sqrt(2.0)
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, root2), (1, -1, root2), (-1, 1, root2), (-1, -1, root2)]

    def h(c):
        return math.hypot(c[0] - g[0], c[1] - g[1])

    best, parent, done = {s: 0.0}, {}, set()
    queue = [(h(s), 0.0, s)]
    while queue:
        _, cost, c = heapq.heappop(queue)
        if c in done:
            continue
        if c == g:
            cells = [c]
            while c in parent:
                c = parent[c]
                cells.append(c)
            return cells[::-1]
        done.add(c)
        for di, dj, step in moves:
            ni, nj = c[0] + di, c[1] + dj
            if not (0 <= ni < n and 0 <= nj < n) or blk[ni][nj]:
                continue
            if di and dj and (blk[c[0] + di][c[1]] or blk[c[0]][c[1] + dj]):
                continue
            nc = cost + step * (1.0 + COMFORT_WEIGHT * max(0.0, COMFORT - away[ni][nj]) / COMFORT)
            if nc < best.get((ni, nj), math.inf):
                best[(ni, nj)] = nc
                parent[(ni, nj)] = c
                heapq.heappush(queue, (nc + h((ni, nj)), nc, (ni, nj)))
    return None


def _line_clear(blocked, p, q):
    """True if the straight line p -> q stays on free cells."""
    step, h = PLAN_CELL / 4.0, 0.35 * PLAN_CELL
    n = max(1, int(math.ceil(math.dist(p, q) / step)))
    for k in range(n + 1):
        u = k / n
        x, y = p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u
        for dx, dy in ((0, 0), (h, 0), (-h, 0), (0, h), (0, -h)):
            if blocked[to_cell((x + dx, y + dy))]:
                return False
    return True


def _straighten(blocked, pts):
    """Greedy shortcut: from each point jump to the farthest one it can see."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_clear(blocked, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _leg(blocked, a, b, max_shift):
    """Straightened point list from a (exactly) to b (exactly), or None."""
    ca, cb = nearest_free(blocked, a), nearest_free(blocked, b)
    if ca is None or cb is None:
        return None
    if math.dist(to_world(ca), a) > max_shift or math.dist(to_world(cb), b) > max_shift:
        return None
    cells = _astar(blocked, ca, cb)
    if cells is None:
        return None
    pts = _straighten(_grow(blocked, STRAIGHT_CLEAR), [to_world(c) for c in cells])
    return [tuple(a)] + pts + [tuple(b)]


def _best_leg(blocked, a, b):
    """_leg() with the widest extra clearance that does not make it much longer."""
    options = []
    for extra in EXTRA_CLEAR:
        pts = _leg(_grow(blocked, extra), a, b, math.inf if extra == 0 else MAX_END_SHIFT)
        if pts is not None:
            options.append((extra, pts, sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))))
    if not options:
        return None
    shortest = min(o[2] for o in options)
    return next(o for o in options if o[2] <= MAX_DETOUR * shortest)


def _time_points(pts, blocked, fast, slow, t0=0.0):
    """Polyline -> [(t, x, y), ...] without the first point, timed from t0:
    `fast` m/s in the open, down to `slow` m/s close to the obstacles."""
    away = _distance_map(blocked)

    def speed_at(p):
        k = min(1.0, away[to_cell(p)] / SLOW_BELOW)
        return slow + (fast - slow) * k

    out, t, cur = [], t0, pts[0]
    for p in pts[1:]:
        d = math.dist(cur, p)
        if d < 1e-9:
            continue
        pieces = max(1, int(math.ceil(d / STEP)))
        for k in range(1, pieces + 1):
            u0, u1 = (k - 1) / pieces, k / pieces
            q0 = (cur[0] + (p[0] - cur[0]) * u0, cur[1] + (p[1] - cur[1]) * u0)
            q1 = p if k == pieces else (cur[0] + (p[0] - cur[0]) * u1, cur[1] + (p[1] - cur[1]) * u1)
            ahead = min(1.0, u1 + LOOK_AHEAD / d)
            q2 = (cur[0] + (p[0] - cur[0]) * ahead, cur[1] + (p[1] - cur[1]) * ahead)
            v = min(speed_at(q0), speed_at(q1), speed_at(q2))
            t += math.dist(q0, q1) / v
            out.append((t, q1[0], q1[1]))
        cur = p
    return out


def plan_route(blocked, start, goal, line_up=False):
    """Task 2B's planner, now for ONE leg: start -> goal on the robot's
    `blocked` grid. [(t, x, y), ...] from (0.0, *start) to goal (no stop on
    it), and the route's length; or (None, None)."""
    start, goal = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
    best = _best_leg(blocked, start, goal)
    if best is None:
        return None, None
    _, pts, length = best

    # leaving the start circle: sideways first, then straight on (Task 2B)
    if line_up:
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            if math.dist(pts[i], pts[i + 1]) < 0.1:
                continue
            if abs(x1 - x0) > 0.05 and abs(y1 - y0) > 0.05:
                tight = _grow(blocked, 0.04)
                if _line_clear(tight, (x0, y0), (x1, y0)) and _line_clear(tight, (x1, y0), (x1, y1)):
                    pts = pts[:i + 1] + [(x1, y0)] + pts[i + 1:]
            break

    return [(0.0, *start)] + _time_points(pts, blocked, PLAN_SPEED, SLOW_SPEED), length


def plan_carry(cblocked, crate, destination=DESTINATION):
    """The crate's centre from where find_payload() found it to `destination`
    on the carry grid: A*, straightened, timed at CARRY_SPEED, and a stop on
    the destination. [(t, x, y), ...] from (0.0, *crate), or None."""
    crate = (float(crate[0]), float(crate[1]))
    best = _best_leg(cblocked, crate, destination)
    if best is None:
        return None
    _, pts, _ = best
    plan = [(0.0, *crate)] + _time_points(pts, cblocked, CARRY_SPEED, CARRY_SLOW_SPEED)
    plan.append((plan[-1][0] + GOAL_STOP, *plan[-1][1:]))
    return plan


def _shift(plan, dt):
    return [(t + dt, x, y) for t, x, y in plan]


def _rotate(v, a):
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def _straight(a, b, speed, t0):
    """a -> b at `speed`, timed from t0, without a."""
    return [(t0 + math.dist(a, b) / speed, float(b[0]), float(b[1]))]


def docking_slots(plates):
    """Candidate docking slots (robot centre - crate centre), best first:
    one per plate, the plate nearest SLOT's direction first, each with the
    solenoid's tip DOCK_GAP short of the crate's collision box along it.
    Without plates, SLOT and its 90-degree turns."""
    if plates is None:
        turns = [0.0] + ([math.pi / 2, -math.pi / 2, math.pi] if TRY_OTHER_SIDES else [])
        return [_rotate(SLOT, t) for t in turns]
    want = math.atan2(SLOT[1], SLOT[0])
    order = sorted(plates, key=lambda a: abs(wrap_angle(a - want)))
    if not TRY_OTHER_SIDES:
        order = order[:1]
    slots = [(d * math.cos(a), d * math.sin(a)) for a in order
             for d in [crate_extent(a) + SOLENOID_TIP + DOCK_GAP]]
    if TRY_OTHER_SIDES and not PLATES_ONLY:     # last resort: square on to a box face
        slots += [_rotate(SLOT, t) for t in (0.0, math.pi / 2, -math.pi / 2, math.pi)]
    return slots


def crate_extent(a):
    """m from the crate's centre to the edge of its collision box along
    direction a. The simulator's box is axis-aligned, CRATE_HALF in size
    (measured: 0.108 m square on, 0.150 m towards a corner), whatever the
    hexagonal look of the crate. The solenoid's tip has to stop on it."""
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return min(CRATE_HALF[0] / c if c > 1e-9 else math.inf,
               CRATE_HALF[1] / s if s > 1e-9 else math.inf)


def plan_mission(occupied_world, crate, robot_xy, log=print, plates=None):
    """All three plans. occupied_world is the grid with payload_beta REMOVED.
    Uses the full robot clearance if any docking side works with it, else
    tightens it step by step (ROBOT_MARGINS). A dict or None."""
    slots = docking_slots(plates)
    n_plates = len(slots) if plates is None else min(len(plates), len(slots))
    for margin in ROBOT_MARGINS:
        log("planning with %.0f mm robot clearance" % (margin * 1000))
        plan = _plan_sides(occupied_world, crate, robot_xy, margin, log, slots, n_plates)
        if plan is not None:
            plan["margin"] = margin
            return plan
    return None


def _eta(l1, l2, l3):
    """Rough mission time (what the robot manages in the simulator: the crate
    drags and twists it, so carrying is slow)."""
    return (l1 + l3) / ETA_ROBOT + l2 / ETA_DRAG


def _plan_sides(occupied_world, crate, robot_xy, margin, log, slots, n_plates):
    """Plans every docking slot and keeps the quickest (shortest in all);
    None if none works."""
    # robot's grid, crate still on its pad (approach)
    occ_app = occupied_world.copy()
    clear_circle_value(occ_app, crate, CRATE_RADIUS, True)
    blk_app = planning_grid(occ_app, margin)
    blk_free = planning_grid(occupied_world, margin)    # crate removed: for the last cm
    # robot's grid, crate on the hexagon (return)
    occ_ret = occupied_world.copy()
    clear_circle_value(occ_ret, DESTINATION, CRATE_RADIUS, True)
    blk_ret = planning_grid(occ_ret, margin)

    results = []
    for rank, slot in enumerate(slots):
        unit = (slot[0] / math.hypot(*slot), slot[1] / math.hypot(*slot))
        yaw = math.atan2(-slot[1], -slot[0])            # red arrow towards the crate
        dock = (crate[0] + slot[0], crate[1] + slot[1])
        tag = "slot (%+.3f, %+.3f), yaw %+.0f deg" % (slot[0], slot[1], math.degrees(yaw))

        # 1. approach: route to the line-up point, pause, straight in
        pres = [(crate[0] + unit[0] * d, crate[1] + unit[1] * d) for d in PREDOCKS]
        pres = [p for p in pres if not blk_app[to_cell(p)] and _line_clear(blk_free, p, dock)]
        if not pres:
            log("  %s: docking point not reachable" % tag)
            continue
        pre = pres[0]
        route, l1 = plan_route(blk_app, robot_xy, pre, line_up=True)
        if route is None:
            log("  %s: no route to the docking point" % tag)
            continue
        t = route[-1][0] + LINE_UP_STOP
        approach = route + [(t, *pre)] + _straight(pre, dock, DOCK_SPEED, t)

        # 2. carry: crate centre on the carry grid
        carry = None
        sweep = 0.0
        for sweep in CARRY_SWEEPS:                     # the widest twist allowance first
            for square, cm in CARRY_SHAPES:
                carry = plan_carry(carry_grid(occupied_world, slot, min(cm, margin),
                                              square, sweep), crate)
                if carry is not None:
                    break
            if carry is not None:
                break
        if carry is None:
            log("  %s: no way to carry the crate to the hexagon" % tag)
            continue
        l2 = sum(math.dist(a[1:], b[1:]) for a, b in zip(carry, carry[1:]))

        # 3. return: back straight off the crate, then route to START
        release = (DESTINATION[0] + slot[0], DESTINATION[1] + slot[1])
        backs = [(DESTINATION[0] + unit[0] * d, DESTINATION[1] + unit[1] * d) for d in PREDOCKS]
        backs = [b for b in backs if not blk_ret[to_cell(b)] and _line_clear(blk_free, release, b)]
        back = backs[0] if backs else None
        if back is None:
            log("  %s: no room to back off the crate at the hexagon" % tag)
            continue
        route, l3 = plan_route(blk_ret, back, START)
        if route is None:
            log("  %s: no route back to the start circle" % tag)
            continue
        t = math.dist(release, back) / DOCK_SPEED
        ret = [(0.0, *release), (t, *back)] + _shift(route[1:], t)
        ret.append((ret[-1][0] + GOAL_STOP, *ret[-1][1:]))

        results.append(dict(slot=slot, yaw=yaw, dock=dock, approach=approach, carry=carry,
                            ret=ret, length=l1 + l2 + l3, rank=rank, sweep=sweep,
                            eta=_eta(l1, l2, l3)))
        log("  %s: %.2f m in all" % (tag, l1 + l2 + l3))
    if not results:
        return None
    # the quickest mission: shortest total distance, docking on a plate if any
    # plate works (the box-face slots, if present, come after the plates)
    on_plate = [r for r in results if r["rank"] < n_plates] or results
    return min(on_plate, key=lambda r: r["eta"])


def _save_debug(frame, plan, crate):
    """The camera crop with the three plans drawn on it."""
    try:
        img = crop_arena(frame).copy()
        s = ARENA_PIXELS / ARENA_SIZE

        def px(p):
            return int(round(p[0] * s)), int(round(p[1] * s))

        for key, colour in (("approach", (0, 0, 255)), ("carry", (0, 200, 0)), ("ret", (255, 0, 0))):
            pts = np.array([px(p[1:]) for p in plan[key]], np.int32)
            cv2.polylines(img, [pts], False, colour, 2)
        cv2.circle(img, px(crate), int(CRATE_RADIUS * s), (0, 255, 255), 2)
        cv2.circle(img, px(plan["dock"]), int(ROBOT_RADIUS * s), (0, 0, 255), 2)
        cv2.imwrite(DEBUG_IMAGE, img)
    except Exception:
        pass


######################### NODE ################################

class PathPlanner(Node):
    """Waits for /r2/odom, plans once from one camera frame, and publishes
    /r2/plan, /carry_plan and /r2/return_plan (all latched)."""

    def __init__(self):
        super().__init__("task_3a_path_planner")
        self.pose = {}
        self.create_subscription(Odometry, "/r2/odom", self._odom_cb("r2"), 10)
        self.pub_plan = self.create_publisher(Path, "/r2/plan", LATCHED)
        self.pub_carry = self.create_publisher(Path, "/carry_plan", LATCHED)
        self.pub_return = self.create_publisher(Path, "/r2/return_plan", LATCHED)
        self.planned, self.tries = False, 0
        # keep the camera stream open and the newest frame at hand, so the
        # plan does not wait for a stream to connect once odom is in
        self.frame, self.frame_lock = None, threading.Lock()
        threading.Thread(target=self._grab, daemon=True).start()
        self.create_timer(0.05, self.tick)
        self.get_logger().info("task 3A planner: r2 takes %s to %s" % (PAYLOAD_NAME, DESTINATION))

    def _grab(self):
        """Background: read the camera stream, keep the newest frame."""
        while rclpy.ok() and not self.planned:
            cap = cv2.VideoCapture(CAMERA_URL)
            while rclpy.ok() and not self.planned:
                ok, f = cap.read()
                if not ok:
                    break
                with self.frame_lock:
                    self.frame = f
            cap.release()

    def _odom_cb(self, robot):
        def cb(msg):
            p = msg.pose.pose.position
            self.pose[robot] = (p.x, p.y)
        return cb

    def to_path_msg(self, plan, start, yaw=0.0, z=0.0):
        """[(t, x, y), ...] and start (rclpy Time) -> nav_msgs/Path. Every pose
        carries `yaw` (the heading to hold) in its orientation."""
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = start.to_msg()
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        for t, x, y in plan:
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.header.stamp = (start + Duration(seconds=float(t))).to_msg()
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)        # /carry_plan: the twist allowed
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            msg.poses.append(ps)
        return msg

    def tick(self):
        if self.planned or "r2" not in self.pose:
            return                                      # wait for the robot's odom
        self.tries += 1
        log = self.get_logger()
        with self.frame_lock:
            frame = self.frame
        if frame is None:
            try:
                frame = read_frame()
            except RuntimeError as e:
                log.warn(str(e))
                return

        crate = find_payload(frame)
        if crate is None:
            log.warn("%s not found in the camera frame (try %d)" % (PAYLOAD_NAME, self.tries))
            return
        plates = find_plates(frame, crate)
        log.info("found %s at (%.3f, %.3f), docking plates at %s deg"
                 % (PAYLOAD_NAME, *crate, "?" if plates is None
                    else ", ".join("%.0f" % math.degrees(a) for a in plates)))
        if plates is None:
            log.warn("could not make out the crate's plates: docking square to SLOT")

        # the world without payload_beta: the other payloads stay on their pads
        others = [p for p in PADS if math.dist(p, crate) > 0.15]
        occupied = remove_drivable(frame, frame_to_occupancy(frame), robots=[self.pose["r2"]],
                                   goals=[DESTINATION], payload_pads=others)
        clear_circle(occupied, crate, CRATE_CLEAR)
        log.info("occupancy grid: %.1f%% obstacle" % (100.0 * occupied.mean()))

        plan = plan_mission(occupied, crate, self.pose["r2"], log=log.info, plates=plates)
        if plan is None:
            log.error("no route / no way to carry for any docking side; relaunch or check "
                      "the grid in %s" % DEBUG_IMAGE)
            if self.tries >= 5:
                self.planned = True
            return
        self.planned = True
        if plates is not None and plan["rank"] >= len(plates):
            log.warn("no plate of the crate can be reached on this layout: docking OFF the "
                     "plates, square on to a face, at (%+.3f, %+.3f)" % plan["slot"])
        elif plan["rank"] != 0:
            log.warn("the preferred plate cannot be reached on this layout: docking at "
                     "(%+.3f, %+.3f) instead" % plan["slot"])
        _save_debug(frame, plan, crate)

        now = self.get_clock().now()
        start = now + Duration(seconds=START_DELAY)
        yaw = plan["yaw"]
        self.pub_plan.publish(self.to_path_msg(plan["approach"], start, yaw))
        log.info("about %.1f s in all" % plan["eta"])
        log.info("published /r2/plan: %d poses to the docking point (%.3f, %.3f), "
                 "facing %.0f deg, arrives at t=%.1f s"
                 % (len(plan["approach"]), *plan["dock"], math.degrees(yaw), plan["approach"][-1][0]))
        self.pub_carry.publish(self.to_path_msg(plan["carry"], now, yaw, plan["sweep"]))
        log.info("published /carry_plan: %d poses, crate onto (%.4f, %.4f) in %.1f s"
                 % (len(plan["carry"]), *DESTINATION, plan["carry"][-1][0]))
        self.pub_return.publish(self.to_path_msg(plan["ret"], now, yaw))
        log.info("published /r2/return_plan: %d poses, back to (%.4f, %.4f) in %.1f s"
                 % (len(plan["ret"]), *START, plan["ret"][-1][0]))


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)                                # keeps the latched paths available
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

