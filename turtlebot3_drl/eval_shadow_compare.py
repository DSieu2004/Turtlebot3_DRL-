import argparse
import csv
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser(description="Record Nav2 /cmd_vel vs DRL shadow /cmd_vel_drl disagreement.")
    parser.add_argument("--output", required=True, help="CSV file to write.")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--sample-rate-hz", type=float, default=10.0)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--drl-cmd-vel-topic", default="/cmd_vel_drl")
    parser.add_argument("--drl-debug-topic", default="/drl_controller_debug")
    return parser.parse_args()


def sign(value: float, eps: float = 1e-3) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


class ShadowCompareRecorder(Node):
    def __init__(self, args):
        super().__init__("drl_shadow_compare")
        self.args = args
        self.latest_nav2_cmd = None
        self.latest_drl_cmd = None
        self.latest_drl_debug = {}
        self.rows = []
        self.create_subscription(Twist, args.cmd_vel_topic, self._nav2_callback, 10)
        self.create_subscription(Twist, args.drl_cmd_vel_topic, self._drl_callback, 10)
        self.create_subscription(String, args.drl_debug_topic, self._debug_callback, 10)

    def _nav2_callback(self, msg):
        self.latest_nav2_cmd = msg

    def _drl_callback(self, msg):
        self.latest_drl_cmd = msg

    def _debug_callback(self, msg):
        parsed = {}
        for item in msg.data.split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            parsed[key] = value
        self.latest_drl_debug = parsed

    def _debug_float(self, key: str) -> float:
        try:
            return float(self.latest_drl_debug.get(key, "nan"))
        except ValueError:
            return math.nan

    def sample(self):
        if self.latest_nav2_cmd is None or self.latest_drl_cmd is None:
            return
        nav2_linear = float(self.latest_nav2_cmd.linear.x)
        nav2_angular = float(self.latest_nav2_cmd.angular.z)
        drl_linear = float(self.latest_drl_cmd.linear.x)
        drl_angular = float(self.latest_drl_cmd.angular.z)
        target_heading = self._debug_float("target_heading")
        angular_sign_disagreement = (
            sign(nav2_angular) != 0 and sign(drl_angular) != 0 and sign(nav2_angular) != sign(drl_angular)
        )
        target_sign_disagreement = (
            math.isfinite(target_heading)
            and sign(target_heading) != 0
            and sign(drl_angular) != 0
            and sign(target_heading) != sign(drl_angular)
        )
        self.rows.append(
            {
                "stamp_sec": time.time(),
                "nav2_linear_x": nav2_linear,
                "nav2_angular_z": nav2_angular,
                "drl_linear_x": drl_linear,
                "drl_angular_z": drl_angular,
                "linear_abs_diff": abs(nav2_linear - drl_linear),
                "angular_abs_diff": abs(nav2_angular - drl_angular),
                "angular_sign_disagreement": angular_sign_disagreement,
                "target_source": self.latest_drl_debug.get("target_source", ""),
                "controller_state": self.latest_drl_debug.get("controller_state", ""),
                "target_distance": self._debug_float("target_distance"),
                "target_heading": target_heading,
                "nearest_obstacle": self._debug_float("nearest_obstacle"),
                "safety_stop": self.latest_drl_debug.get("safety_stop", ""),
                "drl_target_sign_disagreement": target_sign_disagreement,
                "nav2_zero": math.isclose(nav2_linear, 0.0, abs_tol=1e-3)
                and math.isclose(nav2_angular, 0.0, abs_tol=1e-3),
                "drl_zero": math.isclose(drl_linear, 0.0, abs_tol=1e-3)
                and math.isclose(drl_angular, 0.0, abs_tol=1e-3),
            }
        )

    def write_csv(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.args.output)), exist_ok=True)
        fields = [
            "stamp_sec",
            "nav2_linear_x",
            "nav2_angular_z",
            "drl_linear_x",
            "drl_angular_z",
            "linear_abs_diff",
            "angular_abs_diff",
            "angular_sign_disagreement",
            "target_source",
            "controller_state",
            "target_distance",
            "target_heading",
            "nearest_obstacle",
            "safety_stop",
            "drl_target_sign_disagreement",
            "nav2_zero",
            "drl_zero",
        ]
        with open(self.args.output, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def print_summary(self):
        if not self.rows:
            print("[shadow_compare] no paired /cmd_vel and /cmd_vel_drl samples recorded")
            return
        avg_linear_diff = sum(float(row["linear_abs_diff"]) for row in self.rows) / len(self.rows)
        avg_angular_diff = sum(float(row["angular_abs_diff"]) for row in self.rows) / len(self.rows)
        angular_sign_rate = sum(bool(row["angular_sign_disagreement"]) for row in self.rows) / len(self.rows)
        target_rows = [row for row in self.rows if math.isfinite(float(row["target_heading"]))]
        target_sign_rate = (
            sum(bool(row["drl_target_sign_disagreement"]) for row in target_rows) / len(target_rows)
            if target_rows
            else 0.0
        )
        drl_zero_rate = sum(bool(row["drl_zero"]) for row in self.rows) / len(self.rows)
        safety_stop_rate = sum(str(row["safety_stop"]).lower() == "true" for row in self.rows) / len(self.rows)
        states = {}
        for row in self.rows:
            state = str(row.get("controller_state", ""))
            states[state] = states.get(state, 0) + 1
        print(
            "[shadow_compare] "
            f"samples={len(self.rows)} "
            f"avg_linear_diff={avg_linear_diff:.3f} "
            f"avg_angular_diff={avg_angular_diff:.3f} "
            f"angular_sign_disagreement_rate={angular_sign_rate:.3f} "
            f"drl_target_sign_disagreement_rate={target_sign_rate:.3f} "
            f"drl_zero_rate={drl_zero_rate:.3f} "
            f"safety_stop_rate={safety_stop_rate:.3f} "
            f"controller_states={states}"
        )


def main():
    args = parse_args()
    rclpy.init()
    node = ShadowCompareRecorder(args)
    period = 1.0 / max(float(args.sample_rate_hz), 1e-3)
    end_time = time.time() + max(float(args.duration_sec), 0.0)
    try:
        while time.time() < end_time:
            rclpy.spin_once(node, timeout_sec=min(period, 0.1))
            node.sample()
            time.sleep(period)
        node.write_csv()
        node.print_summary()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
