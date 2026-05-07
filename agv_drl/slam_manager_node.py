"""
slam_manager_node.py — SLAM & Navigation lifecycle manager for the warehouse robot.

Provides topic-based control so the dashboard (via rosbridge) can
start/stop SLAM, save maps, list saved maps, and launch Nav2 with
a chosen map — all without SSH access to the Pi.

SLAM and Navigation are **mutually exclusive**: starting one will
automatically stop the other.

Subscribes:
    /slam/command  (std_msgs/String)
        "start_slam"           — launch async SLAM toolbox
        "stop_slam"            — kill SLAM process
        "save_map:<name>"      — save map to ~/maps/<name>.yaml + .pgm
        "list_map"             — scan ~/maps/ and publish list
        "load_map:<name>"      — stop SLAM, launch Nav2 with ~/maps/<name>.yaml
        "stop_nav"             — kill Nav2 process

Publishes:
    /slam/status   (std_msgs/String)   — 1 Hz heartbeat
        "IDLE"                 — nothing running
        "MAPPING"              — SLAM active
        "SAVING"               — map save in progress
        "SAVED:<name>"         — map saved successfully
        "NAV:<name>"           — Nav2 running with <name>
        "LOADING:<name>"       — Nav2 starting up
        "ERROR:<message>"      — something went wrong

    /slam/map_list (std_msgs/String)   — JSON array of map names
        '["warehouse_map","office_floor"]'
"""

import json
import os
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String


class SlamManager(Node):
    def __init__(self):
        super().__init__("slam_manager")

        self.status_pub = self.create_publisher(String, "/slam/status", 10)
        # Keep TRANSIENT_LOCAL so rosbridge/UI subscribers requesting latched data are compatible.
        # Publish is still request-driven (no startup/timer auto-publish).
        map_list_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_list_pub = self.create_publisher(
            String, "/slam/map_list", map_list_qos
        )
        self.create_subscription(String, "/slam/command", self.on_command, 10)

        # Publish status at 1 Hz so dashboard always has current state
        self.create_timer(1.0, self.publish_status)

        self.slam_process = None
        self.nav_process = None
        self.state = "IDLE"  # see docstring for values
        self.active_map = None  # map name when Nav2 is running

        # Ensure ~/maps directory exists
        self.maps_dir = os.path.expanduser("~/maps")
        os.makedirs(self.maps_dir, exist_ok=True)

        self.get_logger().info("SLAM Manager ready — listening on /slam/command")
        self.publish_status()

    # ── Status publisher ───────────────────────────────────────
    def publish_status(self):
        msg = String()
        msg.data = self.state
        self.status_pub.publish(msg)

    def set_state(self, new_state: str):
        self.state = new_state
        self.get_logger().info(f"State → {new_state}")
        self.publish_status()

    # ── Command handler ────────────────────────────────────────
    def on_command(self, msg: String):
        cmd = msg.data.strip()
        self.get_logger().info(f"Command received: {cmd}")

        if cmd == "start_slam":
            self.start_slam()
        elif cmd == "stop_slam":
            self.stop_slam()
        elif cmd.startswith("save_map:"):
            filename = cmd.split(":", 1)[1].strip() or "warehouse_map"
            self.save_map(filename)
        elif cmd in ("list_map", "list_maps"):
            self.list_maps()
        elif cmd.startswith("load_map:"):
            mapname = cmd.split(":", 1)[1].strip()
            if mapname:
                self.load_map(mapname)
            else:
                self.set_state("ERROR:no map name provided")
        elif cmd == "stop_nav":
            self.stop_nav()
        else:
            self.get_logger().warn(f"Unknown command: {cmd}")

    # ── Helper: kill a process group ───────────────────────────
    def _kill_process(self, proc, label="process"):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            self.get_logger().info(f"{label} terminated (SIGTERM)")
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                self.get_logger().info(f"{label} killed (SIGKILL)")
            except Exception:
                pass

    def _stream_subprocess_output(self, proc, prefix: str):
        """Stream a subprocess's stdout/stderr to the ROS log."""

        def _reader(stream, log_fn):
            try:
                for line in iter(stream.readline, ""):
                    text = line.rstrip()
                    if text:
                        log_fn(f"{prefix}: {text}")
            except Exception as exc:
                self.get_logger().debug(f"{prefix} stream reader stopped: {exc}")

        if proc.stdout is not None:
            threading.Thread(
                target=_reader,
                args=(proc.stdout, self.get_logger().info),
                daemon=True,
            ).start()
        if proc.stderr is not None:
            threading.Thread(
                target=_reader,
                args=(proc.stderr, self.get_logger().error),
                daemon=True,
            ).start()

    def _verify_loaded_nav_map(self, expected_yaml_path: str):
        """Check map_server yaml_filename and verify it matches the requested map."""
        expected = os.path.abspath(expected_yaml_path)
        last_output = ""
        mismatch_loaded = None
        mismatch_count = 0
        for _ in range(8):
            try:
                res = subprocess.run(
                    ["ros2", "param", "get", "/map_server", "yaml_filename"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
            except Exception as exc:
                last_output = str(exc)
                time.sleep(0.5)
                continue

            merged = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
            last_output = merged
            if res.returncode != 0:
                time.sleep(0.5)
                continue

            value = ""
            for line in merged.splitlines():
                if "String value is:" in line:
                    value = line.split("String value is:", 1)[1].strip()
                    break
            if not value:
                value = merged.strip().splitlines()[-1] if merged.strip() else ""

            value = value.strip().strip('"').strip("'")
            loaded = os.path.abspath(value) if value else ""
            if loaded == expected:
                return True, loaded

            if loaded:
                if loaded == mismatch_loaded:
                    mismatch_count += 1
                else:
                    mismatch_loaded = loaded
                    mismatch_count = 1

                # Declare mismatch only after seeing the same wrong value repeatedly.
                if mismatch_count >= 3:
                    return False, loaded

            time.sleep(0.5)

        return None, last_output

    # ── Start SLAM ─────────────────────────────────────────────
    def start_slam(self):
        if self.slam_process is not None and self.slam_process.poll() is None:
            self.get_logger().warn("SLAM already running")
            self.set_state("MAPPING")
            return

        # Mutual exclusivity: stop Nav2 first
        if self.nav_process is not None and self.nav_process.poll() is None:
            self.get_logger().info("Stopping Nav2 before starting SLAM…")
            self._kill_process(self.nav_process, "Nav2")
            self.nav_process = None
            self.active_map = None

        try:
            self.slam_process = subprocess.Popen(
                ["ros2", "launch", "turtlebot3_drl", "handheld_slam.launch.py"],
                preexec_fn=os.setsid,
            )
            self.set_state("MAPPING")
            self.get_logger().info(f"SLAM started (PID {self.slam_process.pid})")

            # Detect launch failures shortly after startup and surface a clear state.
            def _check_slam_boot():
                time.sleep(2.0)
                if self.slam_process is None:
                    return
                rc = self.slam_process.poll()
                if rc is not None:
                    self.set_state(f"ERROR:SLAM launch failed (code {rc})")
                    self.get_logger().error(
                        f"SLAM process exited early with code {rc}. Check launch logs."
                    )

            threading.Thread(target=_check_slam_boot, daemon=True).start()
        except Exception as e:
            self.set_state(f"ERROR:{e}")

    # ── Stop SLAM ──────────────────────────────────────────────
    def stop_slam(self):
        if self.slam_process is None or self.slam_process.poll() is not None:
            self.get_logger().info("SLAM not running — nothing to stop")
            self.slam_process = None
            self.set_state("IDLE")
            return

        self._kill_process(self.slam_process, "SLAM")
        self.slam_process = None
        self.set_state("IDLE")

    # ── Save Map ───────────────────────────────────────────────
    def save_map(self, filename: str):
        if self.slam_process is None or self.slam_process.poll() is not None:
            self.set_state("ERROR:SLAM process is not running")
            self.get_logger().error("save_map requested but SLAM process is not alive")
            return

        self.set_state("SAVING")

        def _do_save():
            filepath = os.path.join(self.maps_dir, filename)
            try:
                # Pre-check: confirm /map is available before invoking map_saver_cli.
                try:
                    map_wait = subprocess.run(
                        ["ros2", "topic", "echo", "/map", "--once"],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                except subprocess.TimeoutExpired:
                    self.get_logger().error("/map pre-check timed out after 20s")
                    self.get_logger().error(
                        "Hint: verify /scan and slam_toolbox are running in the same ROS domain"
                    )
                    self.set_state("ERROR:no /map from SLAM (timeout 20s)")
                    return

                if map_wait.returncode != 0:
                    err = (map_wait.stderr or map_wait.stdout or "no /map data").strip()
                    self.set_state(f"ERROR:no /map from SLAM — {err[:120]}")
                    self.get_logger().error(
                        f"save_map aborted: /map unavailable (returncode={map_wait.returncode})"
                    )
                    self.get_logger().error(
                        f"/map check stderr: {(map_wait.stderr or '').strip()}"
                    )
                    self.get_logger().error(
                        f"/map check stdout: {(map_wait.stdout or '').strip()}"
                    )
                    return

                # Give map stream a brief stabilization window.
                time.sleep(0.3)

                cmd = [
                    "ros2",
                    "run",
                    "nav2_map_server",
                    "map_saver_cli",
                    "-f",
                    filepath,
                    "--ros-args",
                    "-p",
                    "save_map_timeout:=20.0",
                ]
                self.get_logger().info(f"Launching map saver: {' '.join(cmd)}")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )
                self._stream_subprocess_output(proc, "map_saver")

                try:
                    return_code = proc.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    self.get_logger().error(
                        "map_saver timed out (40s); terminating process"
                    )
                    self._kill_process(proc, "map_saver")
                    self.set_state("ERROR:map save timed out (40s)")
                    return

                if return_code == 0:
                    self.set_state(f"SAVED:{filename}")
                    self.get_logger().info(f"Map saved → {filepath}.yaml")
                    # Publish updated list so dashboard sees the new map
                    self.list_maps()
                else:
                    self.get_logger().error(f"map_saver exited with code {return_code}")
                    self.set_state(
                        f"ERROR:save failed (code {return_code}) — check Pi logs"
                    )
            except Exception as e:
                self.get_logger().error(f"map_saver exception: {e}")
                self.set_state(f"ERROR:{e}")

        threading.Thread(target=_do_save, daemon=True).start()

    # ── List Maps ──────────────────────────────────────────────
    def list_maps(self):
        """Scan ~/maps/ for .yaml files and publish the list."""
        try:
            maps = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(self.maps_dir)
                if f.endswith(".yaml")
            )
        except Exception as e:
            self.get_logger().error(f"Failed to list maps: {e}")
            maps = []

        msg = String()
        msg.data = json.dumps(maps)
        self.map_list_pub.publish(msg)
        self.get_logger().info(f"Published map list: {maps}")

    # ── Load Map → Launch Nav2 ─────────────────────────────────
    def load_map(self, mapname: str):
        yaml_path = os.path.join(self.maps_dir, mapname + ".yaml")
        if not os.path.isfile(yaml_path):
            self.set_state(f"ERROR:map not found — {mapname}.yaml")
            return

        # Mutual exclusivity: stop SLAM first
        if self.slam_process is not None and self.slam_process.poll() is None:
            self.get_logger().info("Stopping SLAM before loading map…")
            self._kill_process(self.slam_process, "SLAM")
            self.slam_process = None

        # Stop existing Nav2 if running
        if self.nav_process is not None and self.nav_process.poll() is None:
            self.get_logger().info("Stopping previous Nav2…")
            self._kill_process(self.nav_process, "Nav2")
            self.nav_process = None

        self.set_state(f"LOADING:{mapname}")

        def _do_launch():
            try:
                launch_cmd = [
                    "ros2",
                    "launch",
                    "turtlebot3_drl",
                    "real_robot.launch.py",
                    f"map:={yaml_path}",
                ]
                self.get_logger().info(
                    f"Launching Nav2 + DRL with map: {yaml_path}"
                )
                self.nav_process = subprocess.Popen(
                    launch_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )
                self._stream_subprocess_output(self.nav_process, "nav2")
                self.active_map = mapname
                self.get_logger().info(
                    f"Nav2 started with map '{mapname}' (PID {self.nav_process.pid})"
                )
                # Wait a moment for Nav2 to initialize, then set NAV state
                time.sleep(2.0)
                self.nav_process.poll()
                if self.nav_process.returncode is not None:
                    self.set_state(
                        f"ERROR:Nav2 exited immediately (code {self.nav_process.returncode})"
                    )
                    self.nav_process = None
                    self.active_map = None
                else:
                    verified, detail = self._verify_loaded_nav_map(yaml_path)
                    if verified is False:
                        self.get_logger().error(
                            f"map_server loaded '{detail}' instead of requested '{yaml_path}'"
                        )
                        self.set_state("ERROR:wrong map loaded by map_server")
                        return
                    if verified is None:
                        self.get_logger().warn(
                            f"Could not verify loaded map from /map_server yet: {detail}"
                        )
                    self.set_state(f"NAV:{mapname}")
            except Exception as e:
                self.set_state(f"ERROR:{e}")
                self.nav_process = None
                self.active_map = None

        threading.Thread(target=_do_launch, daemon=True).start()

    # ── Stop Navigation ────────────────────────────────────────
    def stop_nav(self):
        if self.nav_process is None or self.nav_process.poll() is not None:
            self.get_logger().info("Nav2 not running — nothing to stop")
            self.nav_process = None
            self.active_map = None
            self.set_state("IDLE")
            return

        self._kill_process(self.nav_process, "Nav2")
        self.nav_process = None
        self.active_map = None
        self.set_state("IDLE")


def main(args=None):
    rclpy.init(args=args)
    node = SlamManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up any running processes
        if node.slam_process and node.slam_process.poll() is None:
            try:
                os.killpg(os.getpgid(node.slam_process.pid), signal.SIGTERM)
            except Exception:
                pass
        if node.nav_process and node.nav_process.poll() is None:
            try:
                os.killpg(os.getpgid(node.nav_process.pid), signal.SIGTERM)
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()