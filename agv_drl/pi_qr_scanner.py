"""
Raspberry Pi — QR Scanner + MJPEG Streamer
===========================================
Runs on the Pi.  Does three things at once:

  1. Captures webcam frames continuously
  2. Decodes QR codes → POSTs to the Warehouse server  POST /scan
  3. Serves an MJPEG stream at /video for the dashboard Camera tab

Install on the Pi:
  pip install flask opencv-python pyzbar requests

Run:
  python pi_qr_scanner.py

Configure the four constants below (or override with env vars).
"""

import os
import cv2
import time
import json
import gzip
import threading
import requests
from flask import Flask, Response, jsonify, request
from pyzbar.pyzbar import decode

# ─── Configuration ────────────────────────────────────────────────────────────
ROBOT_CODE = os.getenv("ROBOT_CODE", "AMR_01")
SERVER_URL = os.getenv(
    "SERVER_URL", "http://100.97.239.69:8000"
)  # ← your PC IP / Tailscale-reachable IP
STREAM_PORT = int(os.getenv("STREAM_PORT", "5000"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
COOLDOWN = float(os.getenv("COOLDOWN", "3.0"))  # seconds before same QR re-triggers
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Shared state — protected by lock
_lock = threading.Lock()
_frame = None  # latest JPEG-encoded bytes (annotated)
_latest_qr = ""
_latest_qr_time = 0.0

# Anti-spam memory
_last_qr = ""
_last_time = 0.0


def camera_loop():
    """Background thread: capture → QR detect → annotate → store frame."""
    global _frame, _last_qr, _last_time, _latest_qr, _latest_qr_time

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index", CAMERA_INDEX)
        return

    print(f"[CAM] Camera opened (index {CAMERA_INDEX})")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── QR detection ──────────────────────────────────────
        qr_codes = decode(frame)

        for qr in qr_codes:
            qr_data = qr.data.decode("utf-8", errors="replace")
            (x, y, w, h) = qr.rect

            # Draw green box + label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                frame,
                qr_data,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # ── Anti-spam check ───────────────────────────────
            now = time.time()
            if qr_data != _last_qr or (now - _last_time) > COOLDOWN:
                _last_qr = qr_data
                _last_time = now
                with _lock:
                    _latest_qr = qr_data
                    _latest_qr_time = now

                # Draw "SCANNED ✓" flash
                cv2.putText(
                    frame,
                    "SCANNED",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # POST to server (fire-and-forget in thread to avoid blocking)
                threading.Thread(
                    target=_post_scan,
                    args=(qr_data,),
                    daemon=True,
                ).start()

        # ── Encode frame to JPEG for MJPEG stream ────────────
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _frame = buf.tobytes()


def _post_scan(qr_data: str):
    """
    1. POST /pick/scan — verifies QR against the active pick task for an IN_PROGRESS order.
    2. If the shelf matches a task, wait 1.5 s (simulate pallet lift), then POST
       /pick/{pt_id}/complete to mark the pick done and broadcast the lift signal.
    3. If no active pick task at this location, fall back to /scan to keep the scan log.
    """
    base = SERVER_URL.rstrip("/")
    payload = {"robot_code": ROBOT_CODE, "qr_code": qr_data}
    try:
        # Try POST /pick/scan first (order picking workflow)
        print(f"[SCAN] POSTing to {base}/pick/scan with QR '{qr_data}'")
        r = requests.post(base + "/pick/scan", json=payload, timeout=5)
        if r.ok:
            result = r.json()
            if result.get("status") == "scanned":
                pt_id = result["pick_task_id"]
                print(
                    f"[PICK] ✓ Correct shelf '{qr_data}' — task {pt_id} confirmed, lifting pallet…"
                )
                time.sleep(1.5)  # brief pause to simulate pallet lift mechanism
                r2 = requests.post(base + f"/pick/{pt_id}/complete", timeout=5)
                if r2.ok:
                    print(f"[PICK] ✓ Task {pt_id} completed — pallet lifted")
                else:
                    print(f"[PICK] ✗ Complete failed {r2.status_code}: {r2.text}")
            else:
                if result.get("status") == "wrong_shelf":
                    expected = result.get("expected", "?")
                    print(
                        f"[PICK] ✗ WRONG SHELF! Scanned '{qr_data}' but expected '{expected}'"
                    )
                else:
                    # No active pick task at this location — just log the scan
                    print(
                        f"[SCAN] No active pick task at '{qr_data}', logging scan via /scan"
                    )
                    r3 = requests.post(base + "/scan", json=payload, timeout=5)
                    if r3.ok:
                        print(f"[SCAN] ✓ Logged to /scan")
        else:
            print(f"[SCAN] /pick/scan returned {r.status_code}: {r.text}")
    except requests.ConnectionError as e:
        print(f"[SCAN] ✗ Connection error to {base}: {e}")
        print(f"[SCAN]   Check: is server running at {SERVER_URL}?")
        print(f"[SCAN]   Check: is Pi network routing to PC?")
    except requests.Timeout:
        print(f"[SCAN] ✗ Timeout reaching {base} (took >5s)")
    except requests.RequestException as e:
        print(f"[SCAN] ✗ Request failed: {e}")


# ─── MJPEG streaming endpoint ────────────────────────────────────────────────


def _generate():
    """Yield MJPEG multipart frames."""
    while True:
        with _lock:
            jpg = _frame
        if jpg is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


@app.route("/video")
def video():
    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return {"robot": ROBOT_CODE, "camera": _frame is not None}


@app.route("/get_qr")
def get_qr():
    """Backward-compatible endpoint for clients that poll the latest QR result."""
    with _lock:
        qr_data = _latest_qr
        last_seen = _latest_qr_time

    if "application/json" in request.headers.get("Accept", ""):
        return jsonify(
            {
                "robot_code": ROBOT_CODE,
                "qr_code": qr_data,
                "last_seen": last_seen,
                "has_scan": bool(qr_data),
            }
        )

    return Response(qr_data, mimetype="text/plain")


@app.route("/list_maps")
def list_maps():
    """List saved map files in ~/maps/ — used as HTTP fallback when ROS topic fails."""
    import glob

    maps_dir = os.path.expanduser("~/maps")
    maps = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(maps_dir, "*.yaml"))
    )
    return jsonify(maps)


# ─── Live SLAM map via ROS subscription ──────────────────────────────────────

_map_lock = threading.Lock()
_map_cache = None  # dict with info + data (compressed)
_map_stamp = 0.0


def _ros_map_subscriber():
    """Background thread: subscribe to /map via rclpy and cache latest OccupancyGrid."""
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import OccupancyGrid
    except ImportError:
        print("[MAP-SUB] rclpy not available — /current_map endpoint disabled")
        return

    rclpy.init()
    node = rclpy.create_node("map_http_bridge")

    def on_map(msg: OccupancyGrid):
        global _map_cache, _map_stamp
        payload = {
            "info": {
                "resolution": msg.info.resolution,
                "width": msg.info.width,
                "height": msg.info.height,
                "origin": {
                    "position": {
                        "x": msg.info.origin.position.x,
                        "y": msg.info.origin.position.y,
                        "z": msg.info.origin.position.z,
                    },
                    "orientation": {
                        "x": msg.info.origin.orientation.x,
                        "y": msg.info.origin.orientation.y,
                        "z": msg.info.origin.orientation.z,
                        "w": msg.info.origin.orientation.w,
                    },
                },
            },
            "data": list(msg.data),
        }
        raw = json.dumps(payload).encode()
        compressed = gzip.compress(raw)
        with _map_lock:
            _map_cache = compressed
            _map_stamp = time.time()
        node.get_logger().info(
            f"[MAP-SUB] Cached map {msg.info.width}x{msg.info.height} "
            f"({len(compressed)} bytes compressed)"
        )

    node.create_subscription(OccupancyGrid, "/map", on_map, 10)
    print("[MAP-SUB] ROS /map subscriber started")
    rclpy.spin(node)


@app.route("/current_map")
def current_map():
    """Return the latest cached /map OccupancyGrid as gzipped JSON."""
    with _map_lock:
        data = _map_cache
        stamp = _map_stamp

    if data is None:
        return jsonify({"error": "no map available"}), 404

    return Response(
        data,
        mimetype="application/json",
        headers={
            "Content-Encoding": "gzip",
            "X-Map-Stamp": str(stamp),
        },
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  Warehouse Pi — QR Scanner + MJPEG Streamer")
    print(f"  Robot     : {ROBOT_CODE}")
    print(f"  Server    : {SERVER_URL}")
    print(f"  Stream    : http://0.0.0.0:{STREAM_PORT}/video")
    print(f"  Camera    : index {CAMERA_INDEX}")
    print(f"  Cooldown  : {COOLDOWN}s")
    print("=" * 56)

    # Start camera thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Start ROS /map subscriber for live SLAM map HTTP endpoint
    ros_thread = threading.Thread(target=_ros_map_subscriber, daemon=True)
    ros_thread.start()

    # Start Flask (MJPEG server)
    app.run(host="0.0.0.0", port=STREAM_PORT, threaded=True)