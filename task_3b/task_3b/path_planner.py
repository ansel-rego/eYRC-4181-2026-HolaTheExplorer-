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
# Filename:         path_planner.py
# Functions:        clear_circle_value, fill_rect, clear_disc, remove_drivable, find_payload,
#                   _block_disc, _grow, _away, _astar, _line_clear, _straighten,
#                   _route_points, _standoff, _sweep, _schedule, _reserve, _plan_from_slots,
#                   _plan_robot, _plan_order, plan_all, _time_polyline, plan_carry,
#                   PathPlanner.to_path_msg
# Global variables: PAINT_S_MIN, PAINT_V_MIN, PAINT_GROW_PX, FLOOR_MIN_VAL,
#                   FLOOR_CELL_FRACTION, MIN_BLOB_CELLS, MAX_HOLE_CELLS, PILLAR_CORE_PX,
#                   PADS, PAYLOAD_RADIUS, PAYLOAD_MARGIN, START_CLEAR, ROBOT_CLEAR, GOAL_CLEAR,
#                   DEBUG_IMAGE, VIOLET_H, VIOLET_THRESHOLDS, TOP_MIN_PX, TOP_MAX_PX,
#                   TOP_PARALLAX, PAYLOAD_FRAMES, DOCK_STANDOFFS, DOCK_SPEED, MIN_PLAN_SEP,
#                   MAX_JOINT_TIME, MAX_START_SHIFT, PARK_RADIUS, COMFORT, COMFORT_WEIGHT,
#                   STRAIGHT_CELLS, SLOW_SPEED, SLOW_BELOW, EXTRA_CELLS, MAX_DETOUR,
#                   CARRY_SLOW, CARRY_SLOW_BELOW, LOOK_AHEAD, STEP




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

# remove_drivable(): free ONLY what is provably floor (Task 2C, plus payload_gamma
# put back where find_payload() saw it and the stations' holes filled)
PAINT_S_MIN, PAINT_V_MIN = 90, 60    # HSV, "painted floor" (pads, hexagon, stripes)
PAINT_GROW_PX = 3                    # px, close the anti-aliased edge of paint
FLOOR_MIN_VAL = 150                  # V, floor / its shadows are brighter than this;
                                     # every pillar and crate is darker
FLOOR_CELL_FRACTION = 0.50           # a cell is floor if this much of it is
MIN_BLOB_CELLS = 12                  # cells, obstacles smaller than this are specks
MAX_HOLE_CELLS = 600                 # cells, a hole this small inside an obstacle (a
                                     # station's coloured cargo, freed as paint) is filled
PILLAR_CORE_PX = 7                   # cells, thinner than this is flat paint
PADS = [(0.534, 0.425), (1.904, 0.425),      # m, the four pads; a payload stands on
        (0.316, 1.221), (2.126, 1.221)]      # each (payload_gamma is found, not assumed)
PAYLOAD_RADIUS = 0.12                # m, the other payloads, put back by geometry
PAYLOAD_MARGIN = 0.005               # m, payload_gamma's footprint, put back a bit fat
START_CLEAR = 0.18                   # m, around each start circle (+ its dots)
ROBOT_CLEAR = 0.14                   # m, around a robot (body + arms)
GOAL_CLEAR = 0.20                    # m, around the destination marker
DEBUG_IMAGE = "/tmp/task3b_occupancy.png"   # grid after remove_drivable(), to look at

# find_payload(): the violet TOP is far more saturated and brighter than its pad.
# Measured on the reference frame: top H 133-141, S 130+, V 170+; the pad sits at
# S ~80-120 / V ~90-190. Thresholds are tried strictest first.
VIOLET_H = (118, 160)                # OpenCV hue band for violet
VIOLET_THRESHOLDS = ((150, 170), (130, 170), (120, 150), (100, 140))   # (S min, V min)
TOP_MIN_PX, TOP_MAX_PX = 450, 2600   # px area of the top as the camera sees it (~1000)
TOP_PARALLAX = 0.024                 # the top is raised, so the camera sees it pushed
                                     # outwards by this fraction of its distance from the
                                     # point under the camera (reference frame: seen
                                     # 22 mm out at 0.93 m from the centre)
PAYLOAD_FRAMES = 5                   # frames to take the median over

# plan_all(): each robot is routed to a STAND-OFF point straight behind its slot,
# already facing the payload, and then drives the last few cm straight in
DOCK_STANDOFFS = (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.24)   # m, nearest free one wins
DOCK_SPEED = 0.20                    # m/s, the straight push onto the payload's face
MIN_PLAN_SEP = 0.175                  # m, floor on the robot-robot separation
MAX_JOINT_TIME = 60.0                # s, give up if the team cannot be docked by then
MAX_START_SHIFT = 0.06               # m, how far a start / stand-off may be nudged
PARK_RADIUS = 0.22                   # m, keep later routes off a docked robot

# A* / straightening (Task 2B)
COMFORT = 0.10                       # m, prefer to stay this far from obstacles
COMFORT_WEIGHT = 1.5                 # how strongly (0 = shortest path only)
STRAIGHT_CELLS = 1                   # cells, a shortcut keeps this far off the obstacles
SLOW_SPEED = 0.4                    # m/s, approach speed at the edge of the grown obstacles
SLOW_BELOW = 0.04                    # m, slow down when closer than this to them

# plan_carry(): try extra clearance first, fall back to none (Task 2B)
EXTRA_CELLS = (2, 1, 0)
MAX_DETOUR = 1.10                    # extra clearance may make the carry this much longer
CARRY_SLOW = 0.4                    # m/s, carry speed at the edge of the grown obstacles
CARRY_SLOW_BELOW = 0.04              # m, slow the carry when closer than this
LOOK_AHEAD = 0.09                    # m, start slowing this far before a tight spot
STEP = 0.04                          # m, the carry is timed in pieces this long

###############################################################


# ----------------------------------------------------- arena (given, do not edit)
ROBOTS = ("r1", "r2", "r3")          # also the default planning order
START_CIRCLES = [(0.8969, 2.2616), (1.2192, 2.2616), (1.5415, 2.2616)]
PAYLOAD_NAME = "payload_gamma"
PAYLOAD_HALF = (0.1066, 0.1097)      # m, half sizes of the crate's footprint (x, y)
DESTINATION = (1.2192, 1.2192)       # m, the hexagon's centre: the payload ends here
SLOTS = {"r1": (0.0, -0.1827, math.pi / 2),    # robot centre from payload centre,
         "r2": (-0.1796, 0.0, 0.0),            # and the heading it docks with
         "r3": (0.0, 0.1827, -math.pi / 2)}
ARENA_SIZE = 2.4384                  # m
ROBOT_RADIUS = 0.084                 # m
SAFETY_MARGIN = 0.030                # m, kept clear on top of the robot radius

# ----------------------------------------------------- camera (given)
CAMERA_URL = "http://127.0.0.1:8080/stream"
FRAME_W, FRAME_H = 1280, 720
ARENA_LEFT, ARENA_TOP, ARENA_PIXELS = 304, 24, 671       # floor square in the image
GRID_CELLS = 244                     # occupancy grid is GRID_CELLS x GRID_CELLS
CELL_SIZE = ARENA_SIZE / GRID_CELLS  # m, about 1 cm
SAND_LOW, SAND_HIGH = (10, 40, 170), (30, 120, 255)      # sand colour, HSV

# ----------------------------------------------------- planning (tunable)
PLAN_CELL = 0.04                     # m, planning grid cell
PLAN_SPEED = 0.30                    # m/s, the speed the approaches are timed at
SLOT = PLAN_CELL / PLAN_SPEED        # s, one planning cell of travel
ROBOT_CLEARANCE = 0.30               # m, keep two robots' plans this far apart (the
                                     # follower lags a few cm, the arms stick out)
DOCK_CLEARANCE = 0.23                # m, ... except when BOTH are at the payload: the
                                     # given slots are only 0.255 m apart
DOCK_ZONE = 0.31                     # m, "at the payload": this close to its centre
CARRY_SPEED = 0.45                   # m/s, the speed the carry is timed at
START_DELAY = 0.9                    # s, from publishing to setting off

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


######################### OCCUPANCY GRID ######################

def read_frames(n=1, url=CAMERA_URL):
    """Given. n BGR frames from the overhead camera."""
    cap = cv2.VideoCapture(url)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("no frame from %s -- is task3b.launch.py running?" % url)
    return frames


def crop_arena(frame):
    """Given. The arena floor only, ARENA_PIXELS square."""
    return frame[ARENA_TOP:ARENA_TOP + ARENA_PIXELS, ARENA_LEFT:ARENA_LEFT + ARENA_PIXELS]


def frame_to_occupancy(frame):
    """Given. Frame -> grid[row, col] = grid[y, x], True = obstacle. Only sand is free."""
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    obstacles = cv2.bitwise_not(cv2.inRange(hsv, SAND_LOW, SAND_HIGH))
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(obstacles, contours, -1, 255, cv2.FILLED)
    small = cv2.resize(obstacles, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA)
    return small > 0


def _fine_xy(shape):
    """Given. Cell-centre coordinates (x, y) in m of an occupancy grid."""
    rows, cols = np.indices(shape)
    return (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE


def clear_circle(occupied, xy, radius):
    """Given. Free every cell within `radius` m of point xy (in place)."""
    x, y = _fine_xy(occupied.shape)
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = False


def clear_circle_value(occupied, xy, radius, value):
    """Like clear_circle(), but sets the cells within `radius` m of xy to `value`."""
    x, y = _fine_xy(occupied.shape)
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = value


def fill_rect(occupied, xy, half, value):
    """Set the cells of the axis-aligned rectangle of half sizes `half` around xy."""
    x, y = _fine_xy(occupied.shape)
    occupied[(np.abs(x - xy[0]) <= half[0]) & (np.abs(y - xy[1]) <= half[1])] = value


def clear_disc(occupied, protected, xy, radius):
    """Free every cell within `radius` m of xy that `protected` does not claim."""
    x, y = _fine_xy(occupied.shape)
    disc = (x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2
    occupied[disc & ~protected] = False


def remove_drivable(frame, occupied, robots, goals, payload=None):
    """Free what a robot may drive over; return the grid.

    Keeps the pillars, the stations, the other payloads and -- if `payload`
    is given -- payload_gamma's footprint where find_payload() saw it."""
    ##############  ADD YOUR CODE HERE  ##############
    occupied = occupied.copy()
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]

    # 1. the floor: sand, its bright shadows / trim, or clearly painted
    sand = cv2.inRange(hsv, SAND_LOW, SAND_HIGH) > 0
    floor = (sand | (val >= FLOOR_MIN_VAL) | ((sat >= PAINT_S_MIN) & (val >= PAINT_V_MIN)))
    floor = cv2.morphologyEx(floor.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                             np.ones((PAINT_GROW_PX, PAINT_GROW_PX), np.uint8))
    frac = cv2.resize(floor.astype(np.float32) / 255.0, (GRID_CELLS, GRID_CELLS),
                      interpolation=cv2.INTER_AREA)
    occupied[frac > FLOOR_CELL_FRACTION] = False

    # 2. the payloads: their coloured tops came out as paint, put them back by
    #    geometry. payload_gamma goes back where it really is, not on its pad
    for xy in PADS:
        if payload is None or math.dist(xy, payload) > PAYLOAD_RADIUS + 0.05:
            clear_circle_value(occupied, xy, PAYLOAD_RADIUS, True)
        else:
            clear_circle(occupied, xy, PAYLOAD_RADIUS + 0.04)   # gamma's pad is floor
    if payload is not None:
        fill_rect(occupied, payload, (PAYLOAD_HALF[0] + PAYLOAD_MARGIN,
                                      PAYLOAD_HALF[1] + PAYLOAD_MARGIN), True)

    # 3. start circles, robots, destination marker: flat, freed by geometry,
    #    but never through anything thick enough to be a pillar or a crate
    core = cv2.erode(occupied.astype(np.uint8) * 255,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PILLAR_CORE_PX,) * 2))
    pillar = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (PILLAR_CORE_PX,) * 2)) > 0
    for xy in START_CIRCLES:
        clear_disc(occupied, pillar, xy, START_CLEAR)
    for xy in robots:
        clear_circle(occupied, xy, ROBOT_CLEAR)
    for xy in goals:
        clear_disc(occupied, pillar, xy, GOAL_CLEAR)

    # 4. clean up: one-cell lines out (3x3 opening), specks out, small holes in
    #    (a station's cargo is saturated and was freed as paint in step 1)
    mask = cv2.morphologyEx(occupied.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    small = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_CELLS])
    mask[small] = 0
    contours, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for c, h in zip(contours, hier[0]):
            if h[3] != -1 and cv2.contourArea(c) < MAX_HOLE_CELLS:
                cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    occupied = mask > 0
    if payload is not None:              # keep the crate whole after the clean-up
        fill_rect(occupied, payload, (PAYLOAD_HALF[0] + PAYLOAD_MARGIN,
                                      PAYLOAD_HALF[1] + PAYLOAD_MARGIN), True)

    try:
        cv2.imwrite(DEBUG_IMAGE, np.where(occupied, 0, 255).astype(np.uint8))
    except Exception:
        pass
    return occupied
    ##################################################


def find_payload(frames):
    """Frame(s) -> payload_gamma's centre (x, y) in m, or None.

    Mask the violet top by colour, take the blob's centroid, turn the pixel
    into metres and pull it back towards the point under the camera."""
    ##############  ADD YOUR CODE HERE  ##############
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    m_per_px = ARENA_SIZE / ARENA_PIXELS
    under = ((FRAME_W / 2.0 - ARENA_LEFT) * m_per_px, (FRAME_H / 2.0 - ARENA_TOP) * m_per_px)
    seen = []
    for frame in frames:
        hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
        for s_min, v_min in VIOLET_THRESHOLDS:
            mask = cv2.inRange(hsv, (VIOLET_H[0], s_min, v_min), (VIOLET_H[1], 255, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
            ok = [i for i in range(1, n)
                  if TOP_MIN_PX <= stats[i, cv2.CC_STAT_AREA] <= TOP_MAX_PX]
            if ok:
                i = max(ok, key=lambda k: stats[k, cv2.CC_STAT_AREA])
                px, py = cents[i]
                x, y = (px + 0.5) * m_per_px, (py + 0.5) * m_per_px
                k = 1.0 - TOP_PARALLAX
                seen.append((under[0] + (x - under[0]) * k, under[1] + (y - under[1]) * k))
                break
    if not seen:
        return None
    xs, ys = zip(*seen)
    return float(np.median(xs)), float(np.median(ys))
    ##################################################


######################### PLANNING GRIDS ######################

def planning_grid(occupied):
    """Given. Grid -> blocked[i, j] = blocked[x, y] (NOTE: x first) of PLAN_CELL
    cells. Obstacles grow by ROBOT_RADIUS + SAFETY_MARGIN, so plan the robot's
    centre as a point. The arena's edge is a wall."""
    pad = ROBOT_RADIUS + SAFETY_MARGIN
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


def carry_grid(occupied):
    """Given. Grid -> blocked[i, j] = blocked[x, y] of PLAN_CELL cells for the
    PAYLOAD'S CENTRE while it is carried: every obstacle grows by the whole
    shape (crate + the three docked robots, each + SAFETY_MARGIN), and the
    whole shape has to stay inside the arena. `occupied` must not contain
    payload_gamma or the robots."""
    m, r = SAFETY_MARGIN, ROBOT_RADIUS + SAFETY_MARGIN
    reach = max(max(abs(sx), abs(sy)) for sx, sy, _ in SLOTS.values()) + r
    k = int(math.ceil(reach / CELL_SIZE))
    off = (np.arange(2 * k + 1) - k) * CELL_SIZE
    dx, dy = np.meshgrid(off, off)                  # dx along columns (x), dy along rows (y)
    shape = (np.abs(dx) <= PAYLOAD_HALF[0] + m) & (np.abs(dy) <= PAYLOAD_HALF[1] + m)
    for sx, sy, _ in SLOTS.values():
        shape |= (dx - sx) ** 2 + (dy - sy) ** 2 <= r ** 2
    inflated = cv2.dilate(occupied.astype(np.uint8), shape.astype(np.uint8)) > 0

    n = int(round(ARENA_SIZE / PLAN_CELL))
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T
    xs, ys = dx[shape], dy[shape]
    blocked[(centre < -xs.min()) | (centre > ARENA_SIZE - xs.max()), :] = True
    blocked[:, (centre < -ys.min()) | (centre > ARENA_SIZE - ys.max())] = True
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


######################### A* (from Task 2B) ###################

def _block_disc(blocked, xy, radius):
    """A copy of the PLANNING grid with every cell within `radius` m of xy blocked."""
    n = blocked.shape[0]
    c = (np.arange(n) + 0.5) * PLAN_CELL
    out = blocked.copy()
    out[(c[:, None] - xy[0]) ** 2 + (c[None, :] - xy[1]) ** 2 <= radius ** 2] = True
    return out


def _grow(blocked, cells):
    """Blocked grid grown by `cells` planning cells (extra clearance)."""
    if cells <= 0:
        return blocked
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cells + 1, 2 * cells + 1))
    return cv2.dilate(blocked.astype(np.uint8), disc) > 0


def _away(blocked):
    """Distance (m) from each cell to the nearest blocked cell or the edge."""
    padded = np.pad(~blocked, 1, constant_values=False).astype(np.uint8)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1] * PLAN_CELL


def _astar(blocked, s, g):
    """8-connected A* from cell s to cell g, no corner cutting, cells near
    obstacles cost more (keeps to the middle of the free space). Octile
    heuristic: admissible and much tighter than Euclidean on an 8-grid, so
    far fewer nodes are expanded. List of cells or None."""
    n = blocked.shape[0]
    blk = blocked.tolist()
    away = _away(blocked)
    penalty = (1.0 + COMFORT_WEIGHT * np.maximum(0.0, COMFORT - away) / COMFORT).tolist()
    root2 = math.sqrt(2.0)
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, root2), (1, -1, root2), (-1, 1, root2), (-1, -1, root2)]
    gi, gj = g

    def h(i, j):
        a, b = abs(i - gi), abs(j - gj)
        return (a + b) + (root2 - 2.0) * min(a, b)

    best, parent = {s: 0.0}, {}
    queue = [(h(*s), 0.0, s)]
    while queue:
        _, cost, c = heapq.heappop(queue)
        if cost > best.get(c, math.inf):
            continue                            # stale entry
        if c == g:
            cells = [c]
            while c in parent:
                c = parent[c]
                cells.append(c)
            return cells[::-1]
        ci, cj = c
        for di, dj, step in moves:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n and 0 <= nj < n) or blk[ni][nj]:
                continue
            if di and dj and (blk[ci + di][cj] or blk[ci][cj + dj]):
                continue                        # would cut a blocked corner
            nc = cost + step * penalty[ni][nj]
            if nc < best.get((ni, nj), math.inf):
                best[(ni, nj)] = nc
                parent[(ni, nj)] = c
                heapq.heappush(queue, (nc + h(ni, nj), nc, (ni, nj)))
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


def _route_points(blocked, a, b, max_shift=MAX_START_SHIFT):
    """One leg a -> b as a straightened polyline from a to b exactly, or None."""
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


######################### THREE APPROACHES (from Task 2C) ######

def _standoff(blocked, slot, yaw):
    """The nearest free point straight behind a docking slot (the robot backs
    out along its heading), or None. Short = less time spent docking."""
    ux, uy = math.cos(yaw), math.sin(yaw)
    for d in DOCK_STANDOFFS:
        p = (slot[0] - ux * d, slot[1] - uy * d)
        if 0.0 < p[0] < ARENA_SIZE and 0.0 < p[1] < ARENA_SIZE and not blocked[to_cell(p)]:
            return p
    return None


def _sweep(leg, away, dock_from=None):
    """Where the robot is at every SLOT of its own travel, before any waiting.
    Consecutive entries are at most PLAN_CELL apart. It slows near obstacles,
    and drives the legs from index `dock_from` on at DOCK_SPEED."""
    def speed_at(p, i):
        if dock_from is not None and i > dock_from:
            return DOCK_SPEED
        k = min(1.0, away[to_cell(p)] / SLOW_BELOW)
        return SLOW_SPEED + (PLAN_SPEED - SLOW_SPEED) * k

    pos, cur, i = [tuple(leg[0])], tuple(leg[0]), 1
    while i < len(leg):
        nxt = tuple(leg[i])
        d = math.dist(cur, nxt)
        if d < 1e-9:
            i += 1
            continue
        step = speed_at(cur, i) * SLOT
        if step >= d:
            cur, i = nxt, i + 1
        else:
            u = step / d
            cur = (cur[0] + (nxt[0] - cur[0]) * u, cur[1] + (nxt[1] - cur[1]) * u)
        pos.append(cur)
    return pos


def _schedule(sweep, reserved, horizon, sep):
    """The slot the robot is at each point of its sweep so that it is never
    too close to a reserved robot at the same instant (it may wait in place).
    `sep` = (far, near, centre, zone): two robots keep `far` apart, or `near`
    when both are within `zone` of the payload's `centre` (the docking slots).
    Earliest finish. Returns (times, idx) or None."""
    A = len(sweep) - 1
    if A < 1:
        return None
    xs = np.array([p[0] for p in sweep])[:, None]
    ys = np.array([p[1] for p in sweep])[:, None]
    far, near, (cx, cy), zone = sep
    mine_near = np.hypot(xs - cx, ys - cy) < zone                  # (A+1, 1)
    allowed = np.ones((A + 1, horizon + 1), dtype=bool)
    for rx, ry in reserved:                       # vectorised over points x slots
        d = np.hypot(xs - rx[None, :], ys - ry[None, :])
        theirs_near = (np.hypot(rx - cx, ry - cy) < zone)[None, :]
        allowed &= (d >= far) | (mine_near & theirs_near & (d >= near))

    reach = np.zeros((A + 1, horizon + 1), dtype=bool)
    reach[0, 0] = allowed[0, 0]
    for t in range(1, horizon + 1):
        reach[1:, t] = reach[:-1, t - 1] | reach[1:, t - 1]
        reach[0, t] = reach[0, t - 1]
        reach[:, t] &= allowed[:, t]
        if reach[A, t]:
            break                                 # the earliest finish is all we need
    found = np.nonzero(reach[A])[0]
    if found.size == 0:
        return None
    t_end = int(found[0])
    times, idx = [0] * (A + 1), [0] * (t_end + 1)
    i, t = A, t_end
    times[A], idx[t_end] = t_end, A
    while t > 0:
        if i > 0 and reach[i - 1, t - 1]:
            i -= 1
        t -= 1
        times[i], idx[t] = t, i
    return times, idx


def _reserve(sweep, idx, horizon):
    """One robot's centre at every slot of the run; after it arrives it stays."""
    last = len(idx) - 1
    pts = np.array([sweep[idx[min(t, last)]] for t in range(horizon + 1)])
    return pts[:, 0], pts[:, 1]


def _plan_from_slots(sweep, idx):
    """[(t, x, y), ...] from one scheduled run: a pose whenever the robot
    starts or stops moving and at every move -- position_at() interpolates,
    so a wait is just the same point at two times."""
    pts = [sweep[i] for i in idx]
    plan = [(0.0, pts[0][0], pts[0][1])]
    for k in range(1, len(pts)):
        if pts[k] != pts[k - 1]:
            if plan[-1][0] < (k - 1) * SLOT - 1e-9:     # a wait just ended
                plan.append(((k - 1) * SLOT, pts[k - 1][0], pts[k - 1][1]))
            plan.append((k * SLOT, pts[k][0], pts[k][1]))
    return plan


def _plan_robot(blocked, away, start, goal, yaw, reserved, horizon, sep, park):
    """One robot's timed approach (route to the stand-off, straight in to the
    slot) around the robots already planned, or None."""
    standoff = _standoff(blocked, goal, yaw)
    if standoff is None:
        return None
    grid = blocked
    for xy in park:
        grid = _block_disc(grid, xy, PARK_RADIUS)
    leg = _route_points(grid, start, standoff) or _route_points(blocked, start, standoff)
    if leg is None:
        return None
    dock_from = len(leg) - 1
    sweep = _sweep(leg + [tuple(goal)], away, dock_from)
    sched = _schedule(sweep, reserved, horizon, sep)
    if sched is None:
        # cannot be timed around them on this route: detour around where they go
        wide = blocked
        for rx, ry in reserved:
            for t in range(0, horizon + 1, max(1, int(0.5 / SLOT))):
                if math.hypot(rx[t] - sep[2][0], ry[t] - sep[2][1]) >= sep[3]:
                    wide = _block_disc(wide, (rx[t], ry[t]), sep[0])
        leg = _route_points(wide, start, standoff)
        if leg is None:
            return None
        dock_from = len(leg) - 1
        sweep = _sweep(leg + [tuple(goal)], away, dock_from)
        sched = _schedule(sweep, reserved, horizon, sep)
        if sched is None:
            return None
    return sched, sweep


def _plan_order(blocked, away, starts, goals, order, horizon, sep):
    """One prioritised pass. A robot not planned yet is assumed to sit on its
    start, so an earlier robot never drives through it. {robot: plan} or None."""
    plans, reserved, park = {}, [], []
    for k, r in enumerate(order):
        waiting = [(np.full(horizon + 1, starts[o][0]), np.full(horizon + 1, starts[o][1]))
                   for o in order[k + 1:]]
        out = _plan_robot(blocked, away, starts[r], goals[r], SLOTS[r][2],
                          reserved + waiting, horizon, sep, park)
        if out is None:
            return None
        (_, idx), sweep = out
        plans[r] = _plan_from_slots(sweep, idx)
        reserved.append(_reserve(sweep, idx, horizon))
        park.append(goals[r])
    return plans


def plan_all(blocked, starts, goals):
    """starts {robot: (x, y)}, goals {robot: docking slot (x, y)} ->
    {robot: [(t, x, y), ...]} from (0.0, *start) to exactly the slot, with
    no collisions; or None."""
    ##############  ADD YOUR CODE HERE  ##############
    starts = {r: (float(starts[r][0]), float(starts[r][1])) for r in ROBOTS}
    goals = {r: (float(goals[r][0]), float(goals[r][1])) for r in ROBOTS}
    bad = ["start %s %s" % (r, starts[r]) for r in ROBOTS if blocked[to_cell(starts[r])]]
    bad += ["stand-off of %s" % r for r in ROBOTS
            if _standoff(blocked, goals[r], SLOTS[r][2]) is None]
    if bad:
        print("plan_all: blocked on the grid: %s" % ", ".join(bad), flush=True)
        return None

    horizon = int(round(MAX_JOINT_TIME / SLOT))
    # robots start where they start: never ask for more room than they begin with
    closest = min(math.dist(starts[a], starts[b]) for a in ROBOTS for b in ROBOTS if a < b)
    far = max(MIN_PLAN_SEP, min(ROBOT_CLEARANCE, closest - 0.01))
    centre = tuple(np.mean([(goals[r][0] - SLOTS[r][0], goals[r][1] - SLOTS[r][1])
                            for r in ROBOTS], axis=0))
    sep = (far, min(far, DOCK_CLEARANCE), centre, DOCK_ZONE)
    away = _away(blocked)

    # prioritised planning in every order; keep the one where the LAST robot
    # docks soonest, because nobody attaches until all three are docked
    best = None
    for order in permutations(ROBOTS):
        plans = _plan_order(blocked, away, starts, goals, order, horizon, sep)
        if plans is None:
            continue
        finish = max(plans[r][-1][0] for r in ROBOTS)
        if best is None or finish < best[0] - 1e-6:
            best = (finish, plans, order)
    if best is None:
        print("plan_all: no order of the robots gives a conflict-free approach", flush=True)
        return None
    finish, plans, order = best
    print("plan_all: order %s, all docked by t=%.1f s, robots kept %.2f m apart "
          "(%.2f m on the slots)" % (", ".join(order), finish, sep[0], sep[1]), flush=True)
    return plans
    ##################################################


######################### THE CARRY ###########################

def _time_polyline(pts, away, speed, slow, slow_below):
    """[(t, x, y), ...] along a polyline at `speed`, dropping to `slow` where
    the path comes within `slow_below` of the grown obstacles (looking ahead)."""
    def speed_at(p):
        return slow + (speed - slow) * min(1.0, away[to_cell(p)] / slow_below)

    plan, t, cur = [(0.0, pts[0][0], pts[0][1])], 0.0, pts[0]
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
            plan.append((t, q1[0], q1[1]))
        cur = p
    return plan


def plan_carry(blocked, start, goal=DESTINATION):
    """The payload's centre from `start` (find_payload()) to `goal` on the carry
    grid: [(t, x, y), ...] starting with (0.0, *start), or None."""
    ##############  ADD YOUR CODE HERE  ##############
    start, goal = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
    # widest berth that does not make the carry much longer (the shape is big:
    # extra clearance is cheap insurance against the follower's lag)
    options = []
    for extra in EXTRA_CELLS:
        pts = _route_points(_grow(blocked, extra), start, goal,
                            max_shift=math.inf if extra == 0 else 0.10)
        if pts is not None:
            options.append((extra, pts, sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))))
    if not options:
        return None
    shortest = min(o[2] for o in options)
    extra, pts, length = next(o for o in options if o[2] <= MAX_DETOUR * shortest)
    print("plan_carry: extra clearance %.2f m, %d legs, %.2f m"
          % (extra * PLAN_CELL, len(pts) - 1, length), flush=True)
    return _time_polyline(pts, _away(blocked), CARRY_SPEED, CARRY_SLOW, CARRY_SLOW_BELOW)
    ##################################################


######################### NODE ################################

class PathPlanner(Node):
    """Given, except to_path_msg(). Waits for every /rN/odom, finds the payload,
    plans once, and publishes /rN/plan (one shared start) and /carry_plan."""

    def __init__(self):
        super().__init__("task_3b_path_planner")
        self.pose, self.pubs = {}, {}
        for r in ROBOTS:
            self.create_subscription(Odometry, f"/{r}/odom", self._odom_cb(r), 10)
            self.pubs[r] = self.create_publisher(Path, f"/{r}/plan", LATCHED)
        self.carry_pub = self.create_publisher(Path, "/carry_plan", LATCHED)
        self.planned = False
        self.create_timer(0.2, self.tick)
        self.get_logger().info("task 3B planner: %s -> %s" % (PAYLOAD_NAME, DESTINATION))

    def _odom_cb(self, robot):
        def cb(msg):
            p = msg.pose.pose.position
            self.pose[robot] = (p.x, p.y)
        return cb

    def to_path_msg(self, plan, start):
        """[(t, x, y), ...] and start (rclpy Time) -> nav_msgs/Path."""
        ##############  ADD YOUR CODE HERE  ##############
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = start.to_msg()
        for t, x, y in plan:
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.header.stamp = (start + Duration(seconds=float(t))).to_msg()
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg
        ##################################################

    def tick(self):
        if self.planned or len(self.pose) < len(ROBOTS):
            return
        self.planned = True
        log = self.get_logger()

        frames = read_frames(PAYLOAD_FRAMES)
        payload = find_payload(frames)
        if payload is None:
            log.error("payload_gamma not found in the camera")
            return
        log.info("%s found at (%.3f, %.3f)" % (PAYLOAD_NAME, *payload))
        frame = frames[-1]
        robots = list(self.pose.values())

        # the approaches: payload_gamma is an obstacle until it is docked on
        occupied = remove_drivable(frame, frame_to_occupancy(frame), robots=robots,
                                   goals=[DESTINATION], payload=payload)
        blocked = planning_grid(occupied)
        goals = {r: (payload[0] + SLOTS[r][0], payload[1] + SLOTS[r][1]) for r in ROBOTS}
        plans = plan_all(blocked, self.pose, goals)
        if plans is None:
            log.error("no conflict-free approach. Check %s" % DEBUG_IMAGE)
            return

        # the carry: payload_gamma (and the robots) are the moving shape itself
        free = occupied.copy()
        fill_rect(free, payload, (PAYLOAD_HALF[0] + 0.03, PAYLOAD_HALF[1] + 0.03), False)
        carry = plan_carry(carry_grid(free), payload)
        if carry is None:
            log.error("no way to carry %s to %s" % (PAYLOAD_NAME, DESTINATION))
            return

        start = self.get_clock().now() + Duration(seconds=START_DELAY)
        for r in ROBOTS:
            self.pubs[r].publish(self.to_path_msg(plans[r], start))
            length = sum(math.dist(a[1:], b[1:]) for a, b in zip(plans[r], plans[r][1:]))
            log.info("%s: published /%s/plan, %d poses, %.2f m, docks at t=%.1f s"
                     % (r, r, len(plans[r]), length, plans[r][-1][0]))
        self.carry_pub.publish(self.to_path_msg(carry, start))
        length = sum(math.dist(a[1:], b[1:]) for a, b in zip(carry, carry[1:]))
        log.info("published /carry_plan, %d poses, %.2f m, %.1f s"
                 % (len(carry), length, carry[-1][0]))


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
