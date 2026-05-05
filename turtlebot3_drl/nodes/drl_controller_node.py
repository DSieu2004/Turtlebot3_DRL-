import math
import time

import numpy as np

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from turtlebot3_drl.utils.geometry import goal_relative_geometry, quaternion_to_yaw
from turtlebot3_drl.utils.ros_gazebo import load_yaml, stamp_twist_zero
from turtlebot3_drl.utils.scan_features import (
    build_lidar_feature,
    clean_lidar_ranges,
    compute_nearest_obstacle,
)


class DRLControllerNode(Node):
    """Shadow-mode DRL inference node. Publishes /cmd_vel_drl by default."""

    def __init__(self):
        super().__init__("drl_controller_node")
        default_config = (
            f"{get_package_share_directory('turtlebot3_drl')}/config/gap_sac.yaml"
        )
        self.declare_parameter("config_file", default_config)
        self.declare_parameter("policy_checkpoint", "")
        self.config = load_yaml(self.get_parameter("config_file").value)
        ros_cfg = self.config.get("ros", {})
        controller_cfg = self.config.get("controller", {})
        state_cfg = self.config.get("state", {})
        action_cfg = self.config.get("action", {})

        self.declare_parameter("input_scan_topic", ros_cfg.get("input_scan_topic", "/scan"))
        self.declare_parameter("input_odom_topic", ros_cfg.get("input_odom_topic", "/odom"))
        self.declare_parameter("input_goal_topic", ros_cfg.get("input_goal_topic", "/goal_pose"))
        self.declare_parameter("input_plan_topic", ros_cfg.get("input_plan_topic", "/plan"))
        self.declare_parameter("output_cmd_vel_topic", ros_cfg.get("shadow_cmd_vel_topic", "/cmd_vel_drl"))
        self.declare_parameter("publish_directly_to_cmd_vel", False)
        self.declare_parameter("fixed_frame", ros_cfg.get("fixed_frame", "map"))
        self.declare_parameter("base_frame", ros_cfg.get("base_frame", "base_link"))
        self.declare_parameter("safety_stop_enabled", ros_cfg.get("safety_stop_enabled", True))
        self.declare_parameter("safety_stop_distance", ros_cfg.get("safety_stop_distance", 0.16))
        self.declare_parameter("inference_rate_hz", ros_cfg.get("inference_rate_hz", 10.0))
        self.declare_parameter("downsampled_lidar_size", state_cfg.get("downsampled_lidar_size", 36))
        self.declare_parameter("lidar_min_range", state_cfg.get("lidar_min_range", 0.12))
        self.declare_parameter("lidar_max_range", state_cfg.get("lidar_max_range", 3.5))
        self.declare_parameter("max_linear_velocity", action_cfg.get("max_linear_velocity", 0.22))
        self.declare_parameter("max_angular_velocity", action_cfg.get("max_angular_velocity", 2.84))
        self.declare_parameter("use_fixed_goal_when_missing", False)
        self.declare_parameter("fixed_goal_x", -1.0)
        self.declare_parameter("fixed_goal_y", -0.2)
        self.declare_parameter("fixed_goal_yaw", 0.0)
        self.declare_parameter("plan_target_mode", controller_cfg.get("plan_target_mode", "lookahead"))
        self.declare_parameter("lookahead_distance_m", controller_cfg.get("lookahead_distance_m", 0.6))
        self.declare_parameter("prefer_plan_goal", controller_cfg.get("prefer_plan_goal", True))
        self.declare_parameter("publish_debug", controller_cfg.get("publish_debug", True))
        self.declare_parameter("heading_guard_enabled", controller_cfg.get("heading_guard_enabled", True))
        self.declare_parameter("heading_guard_min_heading_rad", controller_cfg.get("heading_guard_min_heading_rad", 0.08))
        self.declare_parameter("heading_guard_gain", controller_cfg.get("heading_guard_gain", 1.2))
        self.declare_parameter("heading_guard_clearance_m", controller_cfg.get("heading_guard_clearance_m", 0.30))
        self.declare_parameter("goal_stop_distance_m", controller_cfg.get("goal_stop_distance_m", 0.35))
        self.declare_parameter("speed_governor_enabled", controller_cfg.get("speed_governor_enabled", True))
        self.declare_parameter("speed_slow_distance_m", controller_cfg.get("speed_slow_distance_m", 0.45))
        self.declare_parameter("speed_min_distance_m", controller_cfg.get("speed_min_distance_m", 0.22))
        self.declare_parameter("speed_min_scale", controller_cfg.get("speed_min_scale", 0.15))
        self.declare_parameter("recovery_enabled", controller_cfg.get("recovery_enabled", True))
        self.declare_parameter("recovery_linear_velocity", controller_cfg.get("recovery_linear_velocity", -0.035))
        self.declare_parameter("recovery_angular_velocity", controller_cfg.get("recovery_angular_velocity", 0.55))
        self.declare_parameter("recovery_front_angle_rad", controller_cfg.get("recovery_front_angle_rad", 0.35))
        self.declare_parameter("recovery_duration_sec", controller_cfg.get("recovery_duration_sec", 0.8))
        self.declare_parameter("narrow_passage_enabled", controller_cfg.get("narrow_passage_enabled", True))
        self.declare_parameter("narrow_passage_clearance_m", controller_cfg.get("narrow_passage_clearance_m", 0.42))
        self.declare_parameter(
            "narrow_passage_front_clearance_m",
            controller_cfg.get("narrow_passage_front_clearance_m", 0.24),
        )
        self.declare_parameter(
            "narrow_passage_linear_velocity",
            controller_cfg.get("narrow_passage_linear_velocity", 0.025),
        )
        self.declare_parameter(
            "narrow_passage_centering_gain",
            controller_cfg.get("narrow_passage_centering_gain", 1.4),
        )
        self.declare_parameter(
            "narrow_passage_max_angular_velocity",
            controller_cfg.get("narrow_passage_max_angular_velocity", 0.45),
        )
        self.declare_parameter("debug_selected_goal_topic", "/drl_selected_goal")
        self.declare_parameter("debug_status_topic", "/drl_controller_debug")

        output_topic = self.get_parameter("output_cmd_vel_topic").value
        if self.get_parameter("publish_directly_to_cmd_vel").value:
            output_topic = "/cmd_vel"
            self.get_logger().warn("Direct /cmd_vel publishing is enabled. Use only after safety validation.")

        self.latest_scan = None
        self.latest_odom = None
        self.latest_goal = None
        self.latest_plan = None
        self.selected_goal = None
        self.selected_goal_base_xy = None
        self.selected_goal_source = ""
        self._warned_missing_inputs = False
        self._warned_safety_stop = False
        self._warned_tf_unavailable = False
        self._recovery_until = 0.0
        self._recovery_turn_sign = 1.0
        self.controller_state = "idle"
        self.policy = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.selected_goal_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter("debug_selected_goal_topic").value,
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            self.get_parameter("debug_status_topic").value,
            10,
        )
        self.create_subscription(LaserScan, self.get_parameter("input_scan_topic").value, self._scan_callback, 10)
        self.create_subscription(Odometry, self.get_parameter("input_odom_topic").value, self._odom_callback, 10)
        self.create_subscription(PoseStamped, self.get_parameter("input_goal_topic").value, self._goal_callback, 10)
        self.create_subscription(Path, self.get_parameter("input_plan_topic").value, self._plan_callback, 10)

        self._load_policy_if_configured()
        rate = float(self.get_parameter("inference_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1e-3), self._on_timer)

    def _load_policy_if_configured(self):
        checkpoint = self.get_parameter("policy_checkpoint").value
        if not checkpoint:
            self.get_logger().warn("No policy_checkpoint set. Publishing zero velocity in shadow mode.")
            return
        try:
            from turtlebot3_drl.agents.gap_sac import GAPSACAgent
        except ImportError as exc:
            self.get_logger().error(str(exc))
            return
        state_dim = int(self.config.get("state", {}).get("downsampled_lidar_size", 36)) + 4
        self.policy = GAPSACAgent(state_dim, self.config)
        self.policy.load_checkpoint(checkpoint)
        self.get_logger().info(f"Loaded DRL policy checkpoint: {checkpoint}")

    def _scan_callback(self, msg):
        self.latest_scan = msg

    def _odom_callback(self, msg):
        self.latest_odom = msg

    def _goal_callback(self, msg):
        self.latest_goal = msg

    def _plan_callback(self, msg):
        self.latest_plan = msg

    def _fixed_goal_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = str(self.get_parameter("fixed_frame").value)
        pose.pose.position.x = float(self.get_parameter("fixed_goal_x").value)
        pose.pose.position.y = float(self.get_parameter("fixed_goal_y").value)
        pose.pose.orientation.w = 1.0
        return pose

    def _publish_zero(self):
        self.cmd_pub.publish(stamp_twist_zero(Twist()))

    def _pose_distance_to_robot(self, pose) -> float:
        if self.latest_odom is None:
            return math.inf
        robot = self.latest_odom.pose.pose.position
        return math.hypot(float(pose.position.x) - float(robot.x), float(pose.position.y) - float(robot.y))

    def _pose_to_base_xy(self, stamped_pose):
        source_frame = stamped_pose.header.frame_id or str(self.get_parameter("fixed_frame").value)
        base_frame = str(self.get_parameter("base_frame").value)
        if source_frame == base_frame:
            return float(stamped_pose.pose.position.x), float(stamped_pose.pose.position.y)
        try:
            transform = self.tf_buffer.lookup_transform(base_frame, source_frame, Time())
        except TransformException as exc:
            if not self._warned_tf_unavailable:
                self.get_logger().warn(
                    f"Waiting for TF transform {source_frame}->{base_frame}; falling back to odom geometry. {exc}"
                )
                self._warned_tf_unavailable = True
            return None

        yaw = quaternion_to_yaw(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        x = float(stamped_pose.pose.position.x)
        y = float(stamped_pose.pose.position.y)
        tx = float(transform.transform.translation.x)
        ty = float(transform.transform.translation.y)
        return math.cos(yaw) * x - math.sin(yaw) * y + tx, math.sin(yaw) * x + math.cos(yaw) * y + ty

    def _target_distance_to_robot(self, stamped_pose) -> float:
        base_xy = self._pose_to_base_xy(stamped_pose)
        if base_xy is not None:
            return math.hypot(base_xy[0], base_xy[1])
        return self._pose_distance_to_robot(stamped_pose.pose)

    def _select_plan_target(self):
        if self.latest_plan is None or not self.latest_plan.poses:
            return None
        mode = str(self.get_parameter("plan_target_mode").value).strip().lower()
        if mode == "final":
            return self.latest_plan.poses[-1]
        lookahead = float(self.get_parameter("lookahead_distance_m").value)
        if self.latest_odom is None:
            return self.latest_plan.poses[min(len(self.latest_plan.poses) - 1, 0)]
        for stamped_pose in self.latest_plan.poses:
            if self._target_distance_to_robot(stamped_pose) >= lookahead:
                return stamped_pose
        return self.latest_plan.poses[-1]

    def _select_active_goal(self):
        prefer_plan = bool(self.get_parameter("prefer_plan_goal").value)
        plan_target = self._select_plan_target()
        if prefer_plan and plan_target is not None:
            return plan_target, "plan_lookahead"
        if self.latest_goal is not None:
            return self.latest_goal, "goal_pose"
        if plan_target is not None:
            return plan_target, "plan_lookahead"
        return None, ""

    def _publish_debug(self, nearest_distance: float, safety_stop_active: bool) -> None:
        if not bool(self.get_parameter("publish_debug").value) or self.selected_goal is None:
            return
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = str(self.get_parameter("fixed_frame").value)
        pose_msg.pose = self.selected_goal
        self.selected_goal_pub.publish(pose_msg)

        if self.selected_goal_base_xy is not None:
            distance = math.hypot(*self.selected_goal_base_xy)
            heading = math.atan2(self.selected_goal_base_xy[1], self.selected_goal_base_xy[0])
        else:
            distance = math.inf
            heading = math.inf
        msg = String()
        msg.data = (
            f"target_source={self.selected_goal_source} "
            f"target_x={self.selected_goal.position.x:.3f} "
            f"target_y={self.selected_goal.position.y:.3f} "
            f"target_distance={distance:.3f} "
            f"target_heading={heading:.3f} "
            f"nearest_obstacle={nearest_distance:.3f} "
            f"controller_state={self.controller_state} "
            f"safety_stop={str(bool(safety_stop_active)).lower()}"
        )
        self.debug_pub.publish(msg)

    def _on_timer(self):
        goal, source = self._select_active_goal()
        if goal is None and bool(self.get_parameter("use_fixed_goal_when_missing").value):
            self.latest_goal = self._fixed_goal_pose()
            goal = self.latest_goal
            source = "fixed_fallback"
            self.get_logger().info(
                f"Using fixed fallback goal: x={self.latest_goal.pose.position.x:.3f}, "
                f"y={self.latest_goal.pose.position.y:.3f}"
            )
        self.selected_goal = goal.pose if goal is not None else None
        self.selected_goal_base_xy = self._pose_to_base_xy(goal) if goal is not None else None
        self.selected_goal_source = source
        if self.latest_scan is None or self.latest_odom is None or self.selected_goal is None:
            if not self._warned_missing_inputs:
                self.get_logger().warn("Waiting for /scan, /odom, and a goal before publishing DRL velocity.")
                self._warned_missing_inputs = True
            self._publish_zero()
            return
        if self._selected_goal_reached():
            self.controller_state = "goal_stop"
            self._publish_zero()
            return
        state, nearest_distance, nearest_angle = self._build_state(goal, self.selected_goal_base_xy)

        if self._recovery_active():
            self.controller_state = "recovery"
            self.cmd_pub.publish(self._build_recovery_twist(nearest_angle, self._recovery_turn_sign))
            self._publish_debug(nearest_distance, True)
            return

        if self.get_parameter("safety_stop_enabled").value:
            if nearest_distance < float(self.get_parameter("safety_stop_distance").value):
                if not self._warned_safety_stop:
                    self.get_logger().warn(
                        f"Safety stop active: nearest obstacle distance is {nearest_distance:.3f} m."
                    )
                    self._warned_safety_stop = True
                if bool(self.get_parameter("recovery_enabled").value):
                    self._start_recovery(nearest_angle)
                    self.controller_state = "recovery"
                    self.cmd_pub.publish(self._build_recovery_twist(nearest_angle, self._recovery_turn_sign))
                else:
                    self.controller_state = "safety_stop"
                    self._publish_zero()
                self._publish_debug(nearest_distance, True)
                return
        if self.policy is None:
            self._publish_debug(nearest_distance, False)
            self._publish_zero()
            return

        action = self.policy.select_action(state, deterministic=True)
        action = self._apply_heading_guard(action, nearest_distance)
        action = self._apply_speed_governor(action, nearest_distance)
        narrow_twist = self._build_narrow_passage_twist(nearest_distance)
        if narrow_twist is not None:
            self.controller_state = "narrow_passage"
            self.cmd_pub.publish(narrow_twist)
            self._publish_debug(nearest_distance, False)
            return
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.controller_state = "policy"
        self.cmd_pub.publish(twist)
        self._publish_debug(nearest_distance, False)

    def _selected_goal_reached(self) -> bool:
        if self.selected_goal_base_xy is None:
            return False
        distance = math.hypot(self.selected_goal_base_xy[0], self.selected_goal_base_xy[1])
        return distance <= float(self.get_parameter("goal_stop_distance_m").value)

    def _apply_heading_guard(self, action, nearest_distance: float):
        if not bool(self.get_parameter("heading_guard_enabled").value):
            return action
        if self.selected_goal_base_xy is None:
            return action
        if nearest_distance < float(self.get_parameter("heading_guard_clearance_m").value):
            return action

        target_heading = math.atan2(self.selected_goal_base_xy[1], self.selected_goal_base_xy[0])
        min_heading = float(self.get_parameter("heading_guard_min_heading_rad").value)
        if abs(target_heading) < min_heading:
            return action

        angular = float(action[1])
        if angular == 0.0 or math.copysign(1.0, angular) == math.copysign(1.0, target_heading):
            return action

        corrected = np.array(action, dtype=np.float32, copy=True)
        max_angular = float(self.get_parameter("max_angular_velocity").value)
        gain = float(self.get_parameter("heading_guard_gain").value)
        corrected[1] = float(np.clip(gain * target_heading, -max_angular, max_angular))
        # Slow forward motion while the guard corrects heading so the robot does not cut across the path.
        corrected[0] = min(float(corrected[0]), 0.5 * float(self.get_parameter("max_linear_velocity").value))
        return corrected

    def _apply_speed_governor(self, action, nearest_distance: float):
        if not bool(self.get_parameter("speed_governor_enabled").value):
            return action
        slow_distance = float(self.get_parameter("speed_slow_distance_m").value)
        min_distance = float(self.get_parameter("speed_min_distance_m").value)
        if nearest_distance >= slow_distance:
            return action

        min_scale = float(self.get_parameter("speed_min_scale").value)
        if slow_distance <= min_distance:
            scale = min_scale
        else:
            scale = min_scale + (1.0 - min_scale) * (
                (nearest_distance - min_distance) / (slow_distance - min_distance)
            )
        scale = float(np.clip(scale, min_scale, 1.0))
        governed = np.array(action, dtype=np.float32, copy=True)
        governed[0] = max(0.0, float(governed[0]) * scale)
        return governed

    def _recovery_active(self) -> bool:
        return time.monotonic() < self._recovery_until

    def _start_recovery(self, nearest_angle: float) -> None:
        self._recovery_until = time.monotonic() + max(float(self.get_parameter("recovery_duration_sec").value), 0.0)
        self._recovery_turn_sign = self._recovery_turn_sign_from_obstacle(nearest_angle)

    def _recovery_turn_sign_from_obstacle(self, nearest_angle: float) -> float:
        front_angle = abs(float(self.get_parameter("recovery_front_angle_rad").value))
        if abs(nearest_angle) <= front_angle and self.selected_goal_base_xy is not None:
            target_heading = math.atan2(self.selected_goal_base_xy[1], self.selected_goal_base_xy[0])
            return -1.0 if target_heading >= 0.0 else 1.0
        # Laser angles are positive to the left, so turn away from the nearest obstacle angle.
        return -1.0 if nearest_angle >= 0.0 else 1.0

    def _build_recovery_twist(self, nearest_angle: float, turn_sign: float | None = None) -> Twist:
        twist = Twist()
        twist.linear.x = float(self.get_parameter("recovery_linear_velocity").value)
        angular_speed = abs(float(self.get_parameter("recovery_angular_velocity").value))
        twist.angular.z = (turn_sign if turn_sign is not None else self._recovery_turn_sign_from_obstacle(nearest_angle)) * angular_speed
        return twist

    def _build_narrow_passage_twist(self, nearest_distance: float) -> Twist | None:
        if not bool(self.get_parameter("narrow_passage_enabled").value) or self.latest_scan is None:
            return None
        clearance = float(self.get_parameter("narrow_passage_clearance_m").value)
        front_clearance = float(self.get_parameter("narrow_passage_front_clearance_m").value)
        left_min = self._sector_min(0.35, 1.57)
        right_min = self._sector_min(-1.57, -0.35)
        front_min = self._sector_min(-0.35, 0.35)

        side_constrained = left_min < clearance and right_min < clearance
        front_constrained = front_min < front_clearance
        if not side_constrained and not front_constrained:
            return None

        twist = Twist()
        twist.linear.x = float(self.get_parameter("narrow_passage_linear_velocity").value)
        centering_gain = float(self.get_parameter("narrow_passage_centering_gain").value)
        max_angular = abs(float(self.get_parameter("narrow_passage_max_angular_velocity").value))
        target_heading = 0.0
        if self.selected_goal_base_xy is not None:
            target_heading = math.atan2(self.selected_goal_base_xy[1], self.selected_goal_base_xy[0])

        centering_error = float(left_min - right_min)
        angular = centering_gain * centering_error + 0.35 * target_heading
        if front_constrained:
            twist.linear.x = min(twist.linear.x, 0.5 * float(self.get_parameter("narrow_passage_linear_velocity").value))
            angular += self._recovery_turn_sign_from_obstacle(0.0) * 0.25
        twist.angular.z = float(np.clip(angular, -max_angular, max_angular))
        if nearest_distance < float(self.get_parameter("speed_min_distance_m").value):
            twist.linear.x = min(twist.linear.x, 0.01)
        return twist

    def _sector_min(self, min_angle: float, max_angle: float) -> float:
        if self.latest_scan is None:
            return math.inf
        min_range = float(self.get_parameter("lidar_min_range").value)
        max_range = float(self.get_parameter("lidar_max_range").value)
        cleaned = clean_lidar_ranges(self.latest_scan.ranges, min_range, max_range)
        angles = self._normalize_angles(self.latest_scan.angle_min + np.arange(len(cleaned)) * self.latest_scan.angle_increment)
        mask = (angles >= min_angle) & (angles <= max_angle) & (cleaned > (min_range + 1e-3))
        if not np.any(mask):
            return max_range
        return float(np.min(cleaned[mask]))

    def _build_state(self, goal_stamped, goal_base_xy):
        min_range = float(self.get_parameter("lidar_min_range").value)
        max_range = float(self.get_parameter("lidar_max_range").value)
        lidar = build_lidar_feature(
            self.latest_scan.ranges,
            min_range,
            max_range,
            int(self.get_parameter("downsampled_lidar_size").value),
            True,
        )
        cleaned = clean_lidar_ranges(self.latest_scan.ranges, min_range, max_range)
        valid_indices = np.where(cleaned > (min_range + 1e-3))[0]
        if valid_indices.size:
            nearest_valid_index = int(valid_indices[int(np.argmin(cleaned[valid_indices]))])
            DO = float(cleaned[nearest_valid_index])
            AO = self._normalize_angle(self.latest_scan.angle_min + nearest_valid_index * self.latest_scan.angle_increment)
        else:
            DO, AO = compute_nearest_obstacle(cleaned, self.latest_scan.angle_min, self.latest_scan.angle_increment)
        pose = self.latest_odom.pose.pose
        yaw = quaternion_to_yaw(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        if goal_base_xy is not None:
            DG = math.hypot(goal_base_xy[0], goal_base_xy[1])
            MA = math.atan2(goal_base_xy[1], goal_base_xy[0])
        else:
            goal_pose = goal_stamped.pose
            DG, MA = goal_relative_geometry(
                pose.position.x,
                pose.position.y,
                yaw,
                goal_pose.position.x,
                goal_pose.position.y,
            )
        return np.concatenate([lidar, np.asarray([DG, DO, AO, MA], dtype=np.float32)]), DO, AO

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _normalize_angles(angles: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(angles), np.cos(angles))


def main(args=None):
    rclpy.init(args=args)
    node = DRLControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
