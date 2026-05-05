from typing import Optional


def try_call_empty_service(node, service_name: str, timeout_sec: float = 1.0) -> bool:
    """Call an Empty service if it exists; return False instead of raising."""
    try:
        from std_srvs.srv import Empty
    except ImportError:
        return False
    client = node.create_client(Empty, service_name)
    if not client.wait_for_service(timeout_sec=timeout_sec):
        return False
    future = client.call_async(Empty.Request())
    if future is None:
        return False
    import rclpy

    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    return future.done() and future.exception() is None


def load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def get_nested(config: dict, *keys: str, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def stamp_twist_zero(twist_msg):
    twist_msg.linear.x = 0.0
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = 0.0
    return twist_msg
