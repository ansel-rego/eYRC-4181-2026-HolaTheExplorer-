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
*  This script is to implement Task 2B of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:      [ Names of team members who worked on this file, separated by comma ]
# Filename:         path_planner.py
# Functions:        clear_circle_value, remove_drivable, _grow, _astar, _line_clear, _straighten, _legs,
#                   plan_route, PathPlanner.to_path_msg
# Global variables: PAINT_S_MIN, PAINT_V_MIN, PAINT_GROW_PX, PAINT_CELL_FRACTION,
#                   THIN_CELLS, MAX_FILL_CELLS, PADS, PAYLOAD_RADIUS, START_CLEAR, ROBOT_CLEAR, GOAL_CLEAR, DEBUG_IMAGE,
#                   EXTRA_CELLS, MAX_DETOUR, MAX_END_SHIFT, COMFORT, COMFORT_WEIGHT,
#                   STRAIGHT_CELLS, SLOW_SPEED, SLOW_BELOW, LOOK_AHEAD, STEP


############################ WHAT YOU HAVE TO DO ##############################
#
#  r2 must visit the three GOALS in order without touching an obstacle. This
#  node plans its path ONCE and publishes it on /r2/plan; path_follower.py
#  drives it.
#
#    camera frame --frame_to_occupancy()--> grid       (given)
#                 --remove_drivable()-----> grid       (YOURS)
#                 --planning_grid()-------> blocked    (given)
#                 --plan_route()----------> plan       (YOURS)
#                 --to_path_msg()---------> /r2/plan   (YOURS)
#
#  A plan is [(t, x, y), ...]: be at (x, y) t seconds after setting off.
#  To stop on a goal, repeat its point with a later t.
#
#  Frame (/r2/odom): origin at the arena's TOP-LEFT corner, x right, y DOWN.
#
#  Run:  ros2 launch hb_description task2b.launch.py
#        ros2 run task_2b path_planner
#        ros2 run task_2b path_follower
#
###############################################################################

import heapq
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

################# ADD EXTRA IMPORTS / GLOBALS HERE ############

# remove_drivable(): painted floor = anything clearly coloured (grey pillars and
# dark payloads have low saturation / brightness, so colour separates them)
PAINT_S_MIN, PAINT_V_MIN = 90, 90    # HSV thresholds for "painted"
PAINT_GROW_PX = 3                    # px, cover the anti-aliased edge of paint
PAINT_CELL_FRACTION = 0.20           # a cell is paint if this much of it is
THIN_CELLS = 5                       # cells, obstacles thinner than this are paint outlines
MAX_FILL_CELLS = 1500                # only fill holes in blobs smaller than this
PADS = [(0.534, 0.425), (1.904, 0.425),      # m, the four coloured pads (measured on the
        (0.316, 1.221), (2.126, 1.221)]      # camera frame); a payload stands on each
PAYLOAD_RADIUS = 0.12                # m, the payload sitting in the middle of each pad
START_CLEAR = 0.13                   # m, around each start circle (+ its dots)
ROBOT_CLEAR = 0.10                   # m, around a robot
GOAL_CLEAR = 0.114                   # m, around a goal = ROBOT_RADIUS + SAFETY_MARGIN
DEBUG_IMAGE = "/tmp/task2b_occupancy.png"   # grid after remove_drivable(), to look at

# plan_route(): try extra clearance first (in planning cells), fall back to none
EXTRA_CELLS = (3, 2, 1, 0)
MAX_DETOUR = 1.10                    # extra clearance may make the route this much longer
MAX_END_SHIFT = 0.15                 # m, how far a start/goal may be moved onto a free
                                     # cell when extra clearance is used
COMFORT = 0.20                       # m, the search prefers to stay this far from obstacles
COMFORT_WEIGHT = 2.0                 # how strongly (0 = shortest path only)
STRAIGHT_CELLS = 3                   # cells, a shortcut may not pass closer than this
                                     # (on top of the grown obstacles): keeps the bends
                                     # of the path away from obstacles
SLOW_SPEED = 0.12                    # m/s, speed right at the edge of the grown obstacles
SLOW_BELOW = 0.10                    # m, slow down when closer than this to them
LOOK_AHEAD = 0.15                    # m, start slowing this far before a tight spot
STEP = 0.04                          # m, the path is timed in pieces this long

###############################################################


# ----------------------------------------------------- arena (given, do not edit)
ROBOTS = ("r2",)
START_CIRCLES = [(0.8969, 2.2616), (1.2192, 2.2616), (1.5415, 2.2616)]
GOALS = [(1.2192, 0.4992),           # visited in this order
         (0.3592, 1.5592),
         (2.0792, 0.6992)]
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
PLAN_SPEED = 0.24                    # m/s, the speed the path is timed at
GOAL_STOP = 2.0                      # s, stop this long on each goal
START_DELAY = 2.0                    # s, from publishing to setting off

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


######################### OCCUPANCY GRID ######################

def read_frame(url=CAMERA_URL):
    """Given. One BGR frame from the overhead camera."""
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame from %s -- is task2b.launch.py running?" % url)
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
    """Like clear_circle(), but sets the cells within `radius` m of xy to `value`."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = value


def remove_drivable(frame, occupied, robots, goals):
    """Free what a robot may drive over; return the grid.

    frame_to_occupancy() marks everything that is not sand as an obstacle --
    including the painted floor, the start circles, the robot and the goal
    dots, which leaves NO path. Keep the pillars (and each pad's payload)."""
    ##############  ADD YOUR CODE HERE  ##############
    occupied = occupied.copy()

    # 1. painted floor: clearly coloured pixels are floor paint, not obstacles
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    paint = cv2.inRange(hsv, (0, PAINT_S_MIN, PAINT_V_MIN), (179, 255, 255))
    paint = cv2.dilate(paint, np.ones((PAINT_GROW_PX, PAINT_GROW_PX), np.uint8))
    frac = cv2.resize(paint.astype(np.float32) / 255.0, (GRID_CELLS, GRID_CELLS),
                      interpolation=cv2.INTER_AREA)
    is_paint = (frac > PAINT_CELL_FRACTION).astype(np.uint8) * 255
    occupied[is_paint > 0] = False

    # 2. what is left: drop thin outlines (the hexagon's rim, the pads' rings,
    #    the arena edge), then fill the hole left where a payload's coloured
    #    top was freed (small blobs only, so a ring of leftover border can
    #    never be filled into a solid wall)
    mask = cv2.morphologyEx(occupied.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            np.ones((THIN_CELLS, THIN_CELLS), np.uint8))
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < MAX_FILL_CELLS:
            cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    occupied = mask > 0

    # 3. a payload stands in the middle of each pad: the pad is floor, the
    #    payload is an obstacle. Its coloured top was freed above, so put it back
    for xy in PADS:
        clear_circle_value(occupied, xy, PAYLOAD_RADIUS, True)

    # 4. start circles (with the dots above them), robots, goal dots
    for xy in START_CIRCLES:
        clear_circle(occupied, xy, START_CLEAR)
    for xy in robots:
        clear_circle(occupied, xy, ROBOT_CLEAR)
    for xy in goals:
        clear_circle(occupied, xy, GOAL_CLEAR)

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


def _legs(blocked, start, max_shift):
    """One straightened point list per leg, each starting at the previous goal
    (exactly) and ending exactly on the next one. None if any leg fails."""
    legs, a = [], start
    for goal in GOALS:
        ca, cb = nearest_free(blocked, a), nearest_free(blocked, goal)
        if ca is None or cb is None:
            return None
        if math.dist(to_world(ca), a) > max_shift or math.dist(to_world(cb), goal) > max_shift:
            return None
        cells = _astar(blocked, ca, cb)
        if cells is None:
            return None
        legs.append([a] + _straighten(_grow(blocked, STRAIGHT_CELLS), [to_world(c) for c in cells]) + [goal])
        a = goal
    return legs


def plan_route(blocked, start):
    """start -> GOALS[0] -> GOALS[1] -> GOALS[2] on `blocked`.
    Returns [(t, x, y), ...] starting with (0.0, *start), or None."""
    ##############  ADD YOUR CODE HERE  ##############
    start = (float(start[0]), float(start[1]))

    # 1.+2. A* per leg, then straighten. Try with extra clearance too: the
    #       follower lags the path a little, so a wider berth is safer. Take the
    #       widest berth that does not make the route much longer (a wide berth
    #       can close a narrow gap and send the robot the long way round).
    options = []                                # (extra cells, legs, length)
    for extra in EXTRA_CELLS:
        legs = _legs(_grow(blocked, extra), start, math.inf if extra == 0 else MAX_END_SHIFT)
        if legs is not None:
            length = sum(math.dist(a, b) for leg in legs for a, b in zip(leg, leg[1:]))
            options.append((extra, legs, length))
    if not options:
        return None
    shortest = min(o[2] for o in options)
    extra, legs, length = next(o for o in options if o[2] <= MAX_DETOUR * shortest)
    print("plan_route: extra clearance %.2f m, route %.2f m" % (extra * PLAN_CELL, length), flush=True)

    # 2b. the first leg leaves the start circle with a diagonal. Make it sideways
    #     first, then straight on: the robot then has the gap lined up before it
    #     is anywhere near the obstacles, instead of having to move up and across
    #     at the same time
    first = legs[0]
    for i in range(len(first) - 1):
        (x0, y0), (x1, y1) = first[i], first[i + 1]
        if math.dist(first[i], first[i + 1]) < 0.1:
            continue                            # skip the hop onto the first cell
        if abs(x1 - x0) > 0.05 and abs(y1 - y0) > 0.05:
            tight = _grow(blocked, 1)
            if _line_clear(tight, (x0, y0), (x1, y0)) and _line_clear(tight, (x1, y0), (x1, y1)):
                legs[0] = first[:i + 1] + [(x1, y0)] + first[i + 1:]
        break

    # 3. time it: t = distance so far / speed, where the speed drops from
    #    PLAN_SPEED to SLOW_SPEED as the path gets close to the grown obstacles
    #    (the follower lags a little, and in a narrow gap that is what hits things),
    #    and sit GOAL_STOP on each goal
    away = cv2.distanceTransform(np.pad(~blocked, 1, constant_values=False).astype(np.uint8),
                                 cv2.DIST_L2, 5)[1:-1, 1:-1] * PLAN_CELL

    def speed_at(p):
        k = min(1.0, away[to_cell(p)] / SLOW_BELOW)
        return SLOW_SPEED + (PLAN_SPEED - SLOW_SPEED) * k

    plan, t, cur = [(0.0, *start)], 0.0, start
    for leg in legs:
        for p in leg[1:]:
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
        t += GOAL_STOP
        plan.append((t, cur[0], cur[1]))      # same point, later time = stop
    return plan
    ##################################################


######################### NODE ################################

class PathPlanner(Node):
    """Given, except to_path_msg(). Waits for /r2/odom, plans once, and
    publishes the path on /r2/plan (latched, so a late follower still gets it)."""

    def __init__(self):
        super().__init__("task_2b_path_planner")
        self.pose, self.pubs = {}, {}
        for r in ROBOTS:
            self.create_subscription(Odometry, f"/{r}/odom", self._odom_cb(r), 10)
            self.pubs[r] = self.create_publisher(Path, f"/{r}/plan", LATCHED)
        self.planned = False
        self.create_timer(0.2, self.tick)
        self.get_logger().info("task 2B planner: %s visits %s" % (ROBOTS[0], GOALS))

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
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg
        ##################################################

    def tick(self):
        if self.planned or len(self.pose) < len(ROBOTS):
            return                              # wait for the robot's odom
        self.planned = True
        robot = ROBOTS[0]

        frame = read_frame()
        occupied = remove_drivable(frame, frame_to_occupancy(frame),
                                   robots=list(self.pose.values()), goals=GOALS)
        blocked = planning_grid(occupied)
        self.get_logger().info("occupancy grid: %.1f%% obstacle; planning grid %d x %d"
                               % (100.0 * occupied.mean(), *blocked.shape))

        route = plan_route(blocked, self.pose[robot])
        if route is None:
            self.get_logger().error("no route. Start and goals, as the planner saw them:")
            for name, p in [("start", self.pose[robot])] + [("goal %d" % (k + 1), g) for k, g in enumerate(GOALS)]:
                self.get_logger().error("  %s %s %s" % (name, p, "BLOCKED" if blocked[to_cell(p)] else "free"))
            return

        start = self.get_clock().now() + Duration(seconds=START_DELAY)
        self.pubs[robot].publish(self.to_path_msg(route, start))
        length = sum(math.dist(a[1:], b[1:]) for a, b in zip(route, route[1:]))
        self.get_logger().info("%s: published /%s/plan, %d poses, %.2f m, finishes at t=%.1f s"
                               % (robot, robot, len(route), length, route[-1][0]))


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)                        # keeps the latched path available
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
