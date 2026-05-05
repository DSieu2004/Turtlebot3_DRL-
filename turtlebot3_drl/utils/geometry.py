import math
from typing import Tuple


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return yaw from a quaternion using the standard ROS ENU convention."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def heading_error_to_goal(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
) -> float:
    goal_heading = math.atan2(goal_y - robot_y, goal_x - robot_x)
    return normalize_angle(goal_heading - robot_yaw)


def goal_relative_geometry(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
) -> Tuple[float, float]:
    """Return paper features DG and MA for the current robot-goal geometry."""
    distance_to_goal = euclidean_distance(robot_x, robot_y, goal_x, goal_y)
    heading_error = heading_error_to_goal(robot_x, robot_y, robot_yaw, goal_x, goal_y)
    return distance_to_goal, heading_error
