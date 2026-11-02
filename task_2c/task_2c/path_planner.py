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
*  This script is to implement Task 2C of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ 4181 ]
# Author List:      [ Names of team members who worked on this file, separated by comma ]
# Filename:         path_planner.py
# Functions:        clear_circle_value, clear_disc, remove_drivable, _grow, _astar,
#                   _line_clear, _straighten, _route_points, _sweep, _schedule,
#                   _reserve, _plan_from_slots, _plan_robot, plan_all, to_path_msg
# Global variables: PAINT_S_MIN, PAINT_V_MIN, PAINT_GROW_PX, PAINT_CELL_FRACTION,
#                   FLOOR_MIN_VAL, FLOOR_CELL_FRACTION, MIN_BLOB_CELLS, PILLAR_CORE_PX,
#                   PADS, PAYLOAD_RADIUS, START_CLEAR, ROBOT_CLEAR, GOAL_CLEAR, DEBUG_IMAGE,
#                   SLOT, HOLD_CHUNK, GOAL_STOP, MIN_PLAN_SEP, MAX_JOINT_TIME,
#                   MAX_START_SHIFT, PARK_RADIUS, COMFORT, COMFORT_WEIGHT,
#                   STRAIGHT_CELLS, SLOW_SPEED, SLOW_BELOW


############################ WHAT YOU HAVE TO DO ##############################
#
#  r1, r2 and r3 must each reach their goal without touching an obstacle or
#  each other. A wall with ONE gap crosses the arena and the goals cross
#  sides, so the robots must take the gap one at a time. This node plans all
#  three ONCE and publishes each on /rN/plan; path_follower.py drives them.
#
#    camera frame --frame_to_occupancy()--> grid       (given)
#                 --remove_drivable()-----> grid       (from Task 2B)
#                 --planning_grid()-------> blocked    (given)
#                 --plan_all()------------> plans      (YOURS)
#                 --to_path_msg()---------> /rN/plan   (from Task 2B)
#
#  A plan is [(t, x, y), ...]: be at (x, y) t seconds after setting off. All
#  three share one start time, so the TIMES are what keep the robots apart.
#
#  Frame (/rN/odom): origin at the arena's TOP-LEFT corner, x right, y DOWN.
#
#  Run:  ros2 launch hb_description task2c.launch.py
#        ros2 run task_2c path_planner
#        ros2 run task_2c path_follower
#
###############################################################################

import heapq
import math
from itertools import permutations

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

################# ADD EXTRA IMPORTS / GLOBALS HERE ############

# remove_drivable(): free ONLY what is provably floor.  "Not sand" is not enough
# on its own: a shadow, a pad's ring and an antialiased outline all come out of
# the given sand range, and any of them left occupied becomes a wall across a
# start or a goal once planning_grid() grows it by the robot's clearance.
PAINT_S_MIN, PAINT_V_MIN = 90, 60    # HSV thresholds for "painted floor".  The
                                     # value floor is low on purpose: the
                                     # platform's dark blue rim and its stripes
                                     # are paint on a drivable surface, and a
                                     # saturated pixel is not a pillar.  Every
                                     # obstacle in this arena -- the wall's
                                     # pillars, the toppled pillars, the rocks,
                                     # the platform's pillar -- is grey, S under
                                     # 70, and the pads' payloads are put back by
                                     # geometry in step 2
PAINT_GROW_PX = 3                    # px, cover the anti-aliased edge of paint
PAINT_CELL_FRACTION = 0.20           # a cell is paint if this much of it is
FLOOR_MIN_VAL = 150                  # V floor for "this is floor, shadowed or
                                     # not".  Measured on the reference frame of
                                     # the run: the floor, its shadows and its
                                     # paint all sit at V 185 and up, while every
                                     # obstacle -- the wall's pillars, the
                                     # toppled pillars, the rocks, the platform's
                                     # pillar and the pads' payloads -- sits at
                                     # V 102 or below.  So the shadows are freed
                                     # (the task says the painted floor comes out
                                     # black too) and the pillars stay obstacles.
FLOOR_CELL_FRACTION = 0.50           # a cell is floor if this much of it is
MIN_BLOB_CELLS = 12                  # obstacles smaller than this are specks
PILLAR_CORE_PX = 7                   # cells, the opening that decides what is
                                     # thick enough to be a pillar: anything
                                     # thinner than this (a goal dot and its
                                     # outline, a start circle's dots, a pad's
                                     # ring, an antialiased edge) is flat paint
                                     # and a disc may free it, while a pillar
                                     # (10 cm and up) may not
PADS = [(0.534, 0.425), (1.904, 0.425),      # m, the four coloured pads (given in
        (0.316, 1.221), (2.126, 1.221)]      # Task 2B; both arenas share them)
PAYLOAD_RADIUS = 0.12                # m, the payload standing in the middle of each pad
START_CLEAR = 0.18                   # m, around each start circle (+ its dots)
ROBOT_CLEAR = 0.14                   # m, around a robot: its body AND the arms it
                                     # carries, which stick out about this far
GOAL_CLEAR = 0.30                    # m, around a goal: the dot the camera sees,
                                     # its outline and its shadow all have to go,
                                     # or the goal's own outline is inflated into
                                     # a wall across the goal.  Guarded by the
                                     # pillar core below, so it cannot punch a
                                     # hole in anything solid.  plan_all() checks
                                     # every start and every goal first and
                                     # reports the ones still blocked.
DEBUG_IMAGE = "/tmp/task2c_occupancy.png"   # grid after remove_drivable(), to look at

SLOT = None                          # = PLAN_CELL / PLAN_SPEED, set with the
                                     # planning constants below
HOLD_CHUNK = 2                       # slots, the longest an in-place wait may
                                     # repeat a point: the follower reads a
                                     # repeat of MIN_STOP = 0.5 s (three slots)
                                     # or more as a goal, so a wait is cut into
                                     # SLOT * HOLD_CHUNK pieces, which are not
GOAL_STOP = 1.5                      # s, sit on the goal this long.  The repeat
                                     # is what stops_in() reads as the stop
MIN_PLAN_SEP = 0.20                  # m, the floor on ROBOT_CLEARANCE when the
                                     # robots simply start closer together
MAX_JOINT_TIME = 120.0               # s, give up if the team cannot be in by then
MAX_START_SHIFT = 0.06               # m, how far a start or a goal may be nudged
                                     # onto a free cell (odometry noise on a
                                     # start must not fail the whole team)
PARK_RADIUS = 0.24                   # m, a robot that has arrived parks on its
                                     # goal for the rest of the run, so later
                                     # routes keep out of that cell

COMFORT = 0.20                       # m, the search prefers to stay this far from obstacles
COMFORT_WEIGHT = 2.0                 # how strongly (0 = shortest path only)
STRAIGHT_CELLS = 3                   # cells, a shortcut may not pass closer than this
                                     # (on top of the grown obstacles): keeps the bends
                                     # of the path away from obstacles
SLOW_SPEED = 0.35                    # m/s, speed right at the edge of the grown obstacles
SLOW_BELOW = 1.0                    # m, slow down when closer than this to them

###############################################################


# ----------------------------------------------------- arena (given, do not edit)
ROBOTS = ("r1", "r2", "r3")          # also the planning order
START_CIRCLES = [(0.8969, 2.2616), (1.2192, 2.2616), (1.5415, 2.2616)]
GOALS = {"r1": (1.9992, 0.7992),     # the goals cross sides
         "r2": (1.2192, 0.5392),
         "r3": (0.4392, 0.7992)}
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
PLAN_CELL = 0.04                     # m, planning grid cell
PLAN_SPEED = 0.30                    # m/s, the speed the paths are timed at
SLOT = PLAN_CELL / PLAN_SPEED        # s, one planning cell of travel: the clock the
                                     # reservation table and every wait runs on
ROBOT_CLEARANCE = 0.26               # m, keep two robots' plans this far apart
START_DELAY = 2.0                    # s, from publishing to setting off

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


######################### OCCUPANCY GRID ######################

def read_frame(url=CAMERA_URL):
    """Given. One BGR frame from the overhead camera."""
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame from %s -- is task2c.launch.py running?" % url)
    return frame


def crop_arena(frame):
    """Given. The arena floor only, ARENA_PIXELS square."""
    return frame[ARENA_TOP:ARENA_TOP + ARENA_PIXELS, ARENA_LEFT:ARENA_LEFT + ARENA_PIXELS]


def frame_to_occupancy(frame):
    """Given. Frame -> grid[row, col] = grid[y, x], True = obstacle.
    Only sand is free. To view: cv2.imshow("g", np.where(grid, 0, 255).astype(np.uint8))"""
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    obstacles = cv2.bitwise_not(cv2.inRange(hsv, SAND_LOW, SAND_HIGH))
    # drop specks and thin painted lines, then fill the holes inside each obstacle
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(obstacles, contours, -1, 255, cv2.FILLED)
    # one pixel per cell; a cell is an obstacle if any part of it is
    small = cv2.resize(obstacles, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA)
    return small > 0


def clear_circle(occupied, xy, radius):
    """Given. Free every cell within `radius` m of point xy (in place)."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = False


def clear_circle_value(occupied, xy, radius, value):
    """Like clear_circle(), but sets the cells within `radius` m of xy to `value`.
    A pad's payload has to be put back, and a parked robot's goal cell blocked."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = value


def clear_disc(occupied, protected, xy, radius):
    """Free every cell within `radius` m of xy that `protected` does not claim.

    The guard is what lets a start's and a goal's disc be generous: the pad, the
    dots, the goal dot's outline and the shadow under them are all flat paint,
    but a pillar standing in the same disc is not, and a hole punched in a
    pillar is a hole in the wall the three robots have to queue at."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    disc = (x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2
    occupied[disc & ~protected] = False


def remove_drivable(frame, occupied, robots, goals):
    """Free what a robot may drive over; return the grid.

    frame_to_occupancy() marks everything that is not sand as an obstacle --
    including the painted floor, the robots and the goal dots, which leaves
    NO path. Keep the wall and pillars (and each pad's payload)."""
    ##############  ADD YOUR CODE HERE  ##############
    occupied = occupied.copy()
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]

    # 1. the floor.  A pixel is floor if it is sand, or bright enough to be the
    #    sand's own shadows, paler patches and the arena's trim, or clearly
    #    painted.  Everything else -- the wall's pillars, the toppled pillars,
    #    the rocks, the platform's pillar, the pads' payloads -- is dark and
    #    stays an obstacle.  Getting this wrong is not cosmetic: a shadow left
    #    in the grid is a 0.23 m disc once planning_grid() has grown it, and one
    #    of those across the gap is the whole task.
    sand = cv2.inRange(hsv, SAND_LOW, SAND_HIGH) > 0
    bright = val >= FLOOR_MIN_VAL
    paint = (sat >= PAINT_S_MIN) & (val >= PAINT_V_MIN)
    floor = (sand | bright | paint).astype(np.uint8) * 255
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE,
                             np.ones((PAINT_GROW_PX, PAINT_GROW_PX), np.uint8))
    frac = cv2.resize(floor.astype(np.float32) / 255.0, (GRID_CELLS, GRID_CELLS),
                      interpolation=cv2.INTER_AREA)
    occupied[frac > FLOOR_CELL_FRACTION] = False

    # 2. a payload stands in the middle of each pad: the pad is floor, the
    #    payload is an obstacle, and its own bright colour made the cell floor
    #    in step 1, so put it back by geometry
    for xy in PADS:
        clear_circle_value(occupied, xy, PAYLOAD_RADIUS, True)

    # 3. the start circles, the robots and the goal dots are flat paint lying on
    #    the floor, whatever colour the simulation renders them, so free them by
    #    geometry as well: a pad the camera sees pale must not fence a robot in
    #    on its own start, and a goal dot the camera sees a few centimetres off
    #    the goal's own coordinates must not leave the goal walled off.  The
    #    discs are guarded by `pillar`, an opening that keeps only obstacles at
    #    least PILLAR_CORE_PX cells thick.
    core = cv2.erode((occupied.astype(np.uint8) * 255),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (PILLAR_CORE_PX, PILLAR_CORE_PX)))
    pillar = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (PILLAR_CORE_PX, PILLAR_CORE_PX))) > 0
    for xy in START_CIRCLES:
        clear_disc(occupied, pillar, xy, START_CLEAR)
    for xy in robots:
        # a robot's own body is thick and dark, but it is where the robot is:
        # free it outright, or the robot starts inside an obstacle
        clear_circle(occupied, xy, ROBOT_CLEAR)
    for xy in goals:
        clear_disc(occupied, pillar, xy, GOAL_CLEAR)

    # 4. clean up.  A one-cell-thick dark line -- a painted outline, a stripe on
    #    the platform, an antialiased edge -- is not an obstacle, but it seals
    #    the grid completely: planning_grid() grows it into a 0.23 m band, and
    #    A* may not step diagonally through a blocked line, so a single cell
    #    across the gap stops every robot.  A 3x3 opening takes exactly those
    #    lines out and leaves anything a robot could actually hit (a pillar is
    #    10 cells and up) standing.  The speck filter then drops what is left of
    #    the noise, by area rather than by an opening that would eat a pillar.
    thin = cv2.morphologyEx((occupied.astype(np.uint8) * 255), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_CELLS:
            thin[lab == i] = False
    occupied = thin

    try:                                 # a picture to check the result by eye
        cv2.imwrite(DEBUG_IMAGE, np.where(occupied, 0, 255).astype(np.uint8))
    except Exception:
        pass
    return occupied
    ##################################################


######################### PLANNING GRID #######################

def planning_grid(occupied):
    """Given. Grid -> blocked[i, j] = blocked[x, y] (NOTE: x first) of PLAN_CELL
    cells. Obstacles grow by ROBOT_RADIUS + SAFETY_MARGIN, so plan the robot's
    centre as a point. The arena's edge is a wall."""
    pad = ROBOT_RADIUS + SAFETY_MARGIN
    k = int(math.ceil(pad / CELL_SIZE))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    inflated = cv2.dilate(occupied.astype(np.uint8), disc) > 0

    # a planning cell is blocked if the robot's centre, standing on the cell's
    # centre, would be too close to an obstacle
    n = int(round(ARENA_SIZE / PLAN_CELL))
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T        # [row=y, col=x] -> [i=x, j=y]
    near_wall = (centre < pad) | (centre > ARENA_SIZE - pad)
    blocked[near_wall, :] = True
    blocked[:, near_wall] = True
    return blocked


def to_cell(p):
    """Given. Point (x, y) in m -> planning cell (i, j)."""
    n = int(round(ARENA_SIZE / PLAN_CELL))
    return (min(max(int(p[0] / PLAN_CELL), 0), n - 1), min(max(int(p[1] / PLAN_CELL), 0), n - 1))


def to_world(c):
    """Given. Planning cell (i, j) -> its centre (x, y) in m."""
    return ((c[0] + 0.5) * PLAN_CELL, (c[1] + 0.5) * PLAN_CELL)


def nearest_free(blocked, p, max_r=0.30):
    """Given. The free cell closest to point p, or None within max_r m."""
    c = to_cell(p)
    if not blocked[c]:
        return c
    free = [(i, j) for i in range(blocked.shape[0]) for j in range(blocked.shape[1])
            if not blocked[i, j] and abs(i - c[0]) * PLAN_CELL <= max_r
            and abs(j - c[1]) * PLAN_CELL <= max_r]
    return min(free, key=lambda q: math.dist(to_world(q), p)) if free else None


def _grow(blocked, cells):
    """Blocked grid grown by `cells` planning cells (extra clearance)."""
    if cells <= 0:
        return blocked
    k = 2 * cells + 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(blocked.astype(np.uint8), disc) > 0


def _astar(blocked, s, g):
    """8-connected A* from cell s to cell g. No corner cutting. Cells next to
    obstacles cost more, so the path keeps to the middle of the free space.
    List of cells or None."""
    n = blocked.shape[0]
    blk = blocked.tolist()
    # distance (m) from each free cell to the nearest blocked cell or the edge
    padded = np.pad(~blocked, 1, constant_values=False).astype(np.uint8)
    away = (cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * PLAN_CELL).tolist()
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
                continue                        # would cut a blocked corner
            nc = cost + step * (1.0 + COMFORT_WEIGHT * max(0.0, COMFORT - away[ni][nj]) / COMFORT)
            if nc < best.get((ni, nj), math.inf):
                best[(ni, nj)] = nc
                parent[(ni, nj)] = c
                heapq.heappush(queue, (nc + h((ni, nj)), nc, (ni, nj)))
    return None


def _line_clear(blocked, p, q):
    """True if the straight line p -> q stays on free cells (with a small
    sideways tolerance, because a cell is only known at its centre)."""
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


######################### PLANNING THREE ROBOTS ################

def _route_points(blocked, a, b, max_shift=MAX_START_SHIFT):
    """One leg, a -> b, as a straightened polyline running from a to b exactly,
    or None if either end is walled in.  A start or a goal may be nudged onto a
    free cell by at most `max_shift`, so a centimetre of odometry noise on a
    start cannot fail the whole team."""
    ca, cb = nearest_free(blocked, a), nearest_free(blocked, b)
    if ca is None or cb is None:
        return None
    if math.dist(to_world(ca), a) > max_shift or math.dist(to_world(cb), b) > max_shift:
        return None
    cells = _astar(blocked, ca, cb)
    if cells is None:
        return None
    pts = [tuple(a)] + _straighten(_grow(blocked, STRAIGHT_CELLS),
                                   [to_world(c) for c in cells]) + [tuple(b)]
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, out[-1]) > 1e-9:
            out.append(p)
    return out


def _sweep(leg, blocked):
    """Where the robot is at every SLOT of its own travel, before any waiting.

    One entry per slot, and consecutive entries are at most PLAN_CELL apart:
    the robot covers speed(p) * SLOT, so it slows where the path is tight -- in
    the gap, that is what keeps it off the pillars."""
    away = cv2.distanceTransform(np.pad(~blocked, 1, constant_values=False).astype(np.uint8),
                                 cv2.DIST_L2, 5)[1:-1, 1:-1] * PLAN_CELL

    def speed_at(p):
        k = min(1.0, away[to_cell(p)] / SLOW_BELOW)
        return SLOW_SPEED + (PLAN_SPEED - SLOW_SPEED) * k

    pos, cur, i = [tuple(leg[0])], tuple(leg[0]), 1
    while i < len(leg):
        nxt = tuple(leg[i])
        d = math.dist(cur, nxt)
        if d < 1e-9:
            i += 1
            continue
        step = speed_at(cur) * SLOT
        if step >= d:
            cur, i = nxt, i + 1          # the last piece takes the rest of the slot
        else:
            u = step / d
            cur = (cur[0] + (nxt[0] - cur[0]) * u, cur[1] + (nxt[1] - cur[1]) * u)
        pos.append(cur)
    return pos


def _schedule(sweep, reserved, horizon, sep):
    """When the robot is at each point of its sweep, so that it is never within
    `sep` of a robot planned before it at the same instant.

    `reserved` is one (xs, ys) pair per already-planned robot: its centre at
    every slot of the run, and its goal for every slot after it has parked.
    Waiting in place is a move of its own, so this finds a schedule whenever one
    exists on this route.

    Two robots can never swap places either: a swap needs both of them to be
    within two cells at the same slot, and `sep` is many cells, so the slot they
    would swap in is already forbidden.  Returns (times, idx) -- the slot the
    robot arrives at each point of its sweep, and the point it is at each slot
    -- or None if the horizon is not long enough."""
    A = len(sweep) - 1
    if A < 1:
        return None
    xs = np.array([p[0] for p in sweep])
    ys = np.array([p[1] for p in sweep])
    allowed = np.ones((A + 1, horizon + 1), dtype=bool)
    for rx, ry in reserved:
        for i in range(A + 1):
            allowed[i] &= np.hypot(xs[i] - rx, ys[i] - ry) >= sep

    # reach[i, t]: the robot can be at point i at slot t.  Two moves get there:
    # it drove from point i-1 (slot t-1), or it waited at point i (slot t-1).
    reach = np.zeros((A + 1, horizon + 1), dtype=bool)
    reach[0, 0] = allowed[0, 0]
    for t in range(1, horizon + 1):
        moved = np.concatenate(([False], reach[:-1, t - 1]))
        waited = reach[:, t - 1]
        reach[:, t] = (moved | waited) & allowed[:, t]

    found = np.nonzero(reach[A])[0]
    if found.size == 0:
        return None
    t_end = int(found[0])                # the earliest it can finish
    times, idx = [0] * (A + 1), [0] * (t_end + 1)
    i, t = A, t_end
    times[A], idx[t_end] = t_end, A
    while t > 0:                         # walk the schedule back to slot 0
        if i > 0 and reach[i - 1, t - 1]:
            i -= 1                       # it drove from the point before
        t -= 1
        times[i], idx[t] = t, i
    return times, idx


def _reserve(sweep, idx, horizon, goal):
    """The reservation table one robot leaves behind: its centre at every slot of
    the run, and its goal for every slot after it has arrived -- a robot that
    has arrived stays on the goal."""
    last = len(idx) - 1
    xs, ys = np.empty(horizon + 1), np.empty(horizon + 1)
    for t in range(horizon + 1):
        p = sweep[idx[t]] if t <= last else goal
        xs[t], ys[t] = p
    return xs, ys


def _plan_from_slots(sweep, idx, goal):
    """[(t, x, y), ...] from one scheduled run: a pose at the start, a pose at
    every change of position, and a pose every HOLD_CHUNK slots while it waits.

    A wait is written as repeats of at most HOLD_CHUNK * SLOT = 0.33 s, because
    the follower reads any repeat of MIN_STOP = 0.5 s or more as a goal: only
    the goal itself is a long repeat -- the same point again GOAL_STOP later."""
    pts = [sweep[i] for i in idx]
    plan = [(0.0, pts[0][0], pts[0][1])]
    k = 1
    while k < len(pts):
        if pts[k] != pts[k - 1]:
            plan.append((k * SLOT, pts[k][0], pts[k][1]))
            k += 1
            continue
        run = k                                   # the wait holds k-1 .. run-1
        while run < len(pts) and pts[run] == pts[k - 1]:
            run += 1
        j = k - 1 + HOLD_CHUNK
        while j < run - 1:
            plan.append((j * SLOT, pts[k - 1][0], pts[k - 1][1]))
            j += HOLD_CHUNK
        plan.append(((run - 1) * SLOT, pts[k - 1][0], pts[k - 1][1]))
        k = run
    t_end = (len(pts) - 1) * SLOT
    plan.append((t_end + GOAL_STOP, goal[0], goal[1]))   # the stop on the goal
    return plan


def _plan_robot(blocked, start, goal, reserved, horizon, sep, park):
    """One robot's timed path around the robots already planned, or None."""
    # 1. the route, with the goals those robots will park on kept out of the way
    grid = blocked.copy()
    for xy in park:
        clear_circle_value(grid, xy, PARK_RADIUS, True)
    leg = _route_points(grid, start, goal)
    if leg is None:                          # a tight arena: time it instead
        leg = _route_points(blocked, start, goal)
    if leg is None:
        return None

    # 2. the timing: wait wherever it must to stay `sep` clear of all of them
    sweep = _sweep(leg, blocked)
    sched = _schedule(sweep, reserved, horizon, sep)
    if sched is None:
        # the route is fine but it cannot be timed around them: take a detour
        # that keeps clear of everywhere they go
        wide = blocked.copy()
        for rx, ry in reserved:
            for t in range(0, horizon + 1, max(1, int(0.5 / SLOT))):
                clear_circle_value(wide, (rx[t], ry[t]), sep, True)
        leg = _route_points(wide, start, goal)
        if leg is None:
            return None
        sweep = _sweep(leg, blocked)
        sched = _schedule(sweep, reserved, horizon, sep)
        if sched is None:
            return None
    return sched, sweep


def _plan_order(blocked, starts, goals, order, horizon, sep):
    """One prioritised pass: the robots in `order`, each planned around the ones
    already planned.  Returns {robot: plan} or None."""
    plans, reserved, park = {}, [], []
    for r in order:
        out = _plan_robot(blocked, starts[r], goals[r], reserved, horizon, sep, park)
        if out is None:
            print("   %s cannot be fitted after %s" % (r, ", ".join(plans) or "nobody"),
                  flush=True)
            return None
        (times, idx), sweep = out
        plans[r] = _plan_from_slots(sweep, idx, goals[r])
        reserved.append(_reserve(sweep, idx, horizon, goals[r]))
        park.append(goals[r])
    return plans


def plan_all(blocked, starts):
    """starts {robot: (x, y)} -> {robot: [(t, x, y), ...]}, every robot from
    (0.0, *starts[robot]) to exactly GOALS[robot], with no collisions; or None."""
    ##############  ADD YOUR CODE HERE  ##############
    starts = {r: (float(starts[r][0]), float(starts[r][1])) for r in ROBOTS}
    goals = {r: (float(GOALS[r][0]), float(GOALS[r][1])) for r in ROBOTS}

    # every start and every goal must be free on the grid.  Report all of them,
    # not just the first: the fix is nearly always remove_drivable(), and the
    # gap, each start and each goal has to come out free for the team to move.
    bad = ["%s %s" % (name, p) for name, p in
           [("start " + r, starts[r]) for r in ROBOTS] +
           [("goal " + r, goals[r]) for r in ROBOTS] if blocked[to_cell(p)]]
    if bad:
        print("plan_all: blocked on the grid: %s" % ", ".join(bad), flush=True)
        return None

    horizon = int(round(MAX_JOINT_TIME / SLOT))
    # the robots start where they start: if two of them begin closer together
    # than ROBOT_CLEARANCE, keep them that far apart rather than refuse to plan
    closest = min(math.dist(starts[a], starts[b]) for a in ROBOTS for b in ROBOTS if a != b)
    sep = max(MIN_PLAN_SEP, min(ROBOT_CLEARANCE, closest - 0.02))

    # Prioritised planning: plan the first robot of the order and record where it
    # is at every slot -- a reservation table -- then plan the next robot around
    # it and the last around both.  A robot that has arrived stays on its goal,
    # and a robot may always wait, so the single gap is taken one robot at a
    # time.  ROBOTS is the given priority and is used whenever it works; when it
    # deadlocks -- with one 27 cm gap the robot that goes first can leave another
    # with nowhere to wait that is also clear when the gap comes free -- the
    # planner picks the order that gets the team in soonest instead.  Which
    # robot takes the gap first is decision this task leaves to the planner.
    best = None
    for order in permutations(ROBOTS):
        plans = _plan_order(blocked, starts, goals, order, horizon, sep)
        if plans is None:
            continue
        finish = max(plans[r][-1][0] for r in ROBOTS)
        if best is None or finish < best[0]:
            best = (finish, plans, order)
        if order == ROBOTS:
            pass                        # the given priority works: keep it
    if best is None:
        print("plan_all: no order of the robots gives a conflict-free joint plan",
              flush=True)
        return None

    finish, plans, order = best
    print("plan_all: gap order %s, everyone in by t=%.1f s, never closer than %.2f m"
          % (", ".join(order), finish, sep), flush=True)
    return plans
    ##################################################


######################### NODE ################################

class PathPlanner(Node):
    """Given, except to_path_msg(). Waits for every /rN/odom, plans once, and
    publishes each path on /rN/plan (latched) with one shared start time."""

    def __init__(self):
        super().__init__("task_2c_path_planner")
        self.pose, self.pubs = {}, {}
        for r in ROBOTS:
            self.create_subscription(Odometry, f"/{r}/odom", self._odom_cb(r), 10)
            self.pubs[r] = self.create_publisher(Path, f"/{r}/plan", LATCHED)
        self.planned = False
        self.create_timer(0.2, self.tick)
        self.get_logger().info("task 2C planner: goals %s" % GOALS)

    def _odom_cb(self, robot):
        def cb(msg):
            p = msg.pose.pose.position
            self.pose[robot] = (p.x, p.y)
        return cb

    def to_path_msg(self, plan, start):
        """[(t, x, y), ...] and start (rclpy Time) -> nav_msgs/Path."""
        ##############  ADD YOUR CODE HERE  ##############
        # The path's stamp is the moment the robots set off -- the SAME stamp for
        # all three, which is what keeps the followers on one clock.  Each pose's
        # stamp is that plus the time the robot should be there.
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = start.to_msg()
        for t, x, y in plan:
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.header.stamp = (start + Duration(seconds=float(t))).to_msg()
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg
        ##################################################

    def tick(self):
        if self.planned or len(self.pose) < len(ROBOTS):
            return                              # wait for every robot's odom
        self.planned = True

        frame = read_frame()
        occupied = remove_drivable(frame, frame_to_occupancy(frame),
                                   robots=list(self.pose.values()), goals=list(GOALS.values()))
        blocked = planning_grid(occupied)
        self.get_logger().info("occupancy grid: %.1f%% obstacle; planning grid %d x %d"
                               % (100.0 * occupied.mean(), *blocked.shape))

        plans = plan_all(blocked, self.pose)
        if plans is None:
            self.get_logger().error("no conflict-free joint plan. Starts and goals, as the planner saw them:")
            for r in ROBOTS:
                self.get_logger().error("  %s start %s %s -> goal %s %s" % (
                    r, self.pose[r], "BLOCKED" if blocked[to_cell(self.pose[r])] else "free",
                    GOALS[r], "BLOCKED" if blocked[to_cell(GOALS[r])] else "free"))
            return

        start = self.get_clock().now() + Duration(seconds=START_DELAY)   # shared by all three
        for r in ROBOTS:
            self.pubs[r].publish(self.to_path_msg(plans[r], start))
            length = sum(math.dist(a[1:], b[1:]) for a, b in zip(plans[r], plans[r][1:]))
            self.get_logger().info("%s: published /%s/plan, %d poses, %.2f m, arrives at t=%.1f s"
                                   % (r, r, len(plans[r]), length, plans[r][-1][0]))


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)                        # keeps the latched paths available
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
