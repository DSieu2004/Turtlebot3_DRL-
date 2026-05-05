import math
from typing import Iterable, Tuple

import numpy as np


def clean_lidar_ranges(
    ranges: Iterable[float],
    min_range: float,
    max_range: float,
) -> np.ndarray:
    """Replace invalid scan values and clamp ranges for stable learning."""
    values = np.asarray(list(ranges), dtype=np.float32)
    values = np.nan_to_num(values, nan=max_range, posinf=max_range, neginf=min_range)
    return np.clip(values, min_range, max_range)


def normalize_ranges(ranges: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
    denom = max(max_range - min_range, 1e-6)
    return np.clip((ranges - min_range) / denom, 0.0, 1.0).astype(np.float32)


def downsample_ranges(ranges: np.ndarray, output_size: int) -> np.ndarray:
    """Downsample LiDAR with min pooling to preserve nearby obstacles."""
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if len(ranges) == output_size:
        return ranges.astype(np.float32)
    chunks = np.array_split(ranges, output_size)
    return np.asarray([float(np.min(chunk)) for chunk in chunks], dtype=np.float32)


def compute_nearest_obstacle(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
) -> Tuple[float, float]:
    """Return paper features DO and AO from a LaserScan."""
    if ranges.size == 0:
        return math.inf, 0.0
    index = int(np.argmin(ranges))
    distance = float(ranges[index])
    angle = angle_min + index * angle_increment
    return distance, angle


def compute_collision_condition(nearest_distance: float, collision_distance: float) -> bool:
    return nearest_distance <= collision_distance


def compute_safety_minimum_distance(ranges: Iterable[float], fallback: float = math.inf) -> float:
    values = np.asarray(list(ranges), dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback
    return float(np.min(finite_values))


def build_lidar_feature(
    ranges: Iterable[float],
    min_range: float,
    max_range: float,
    output_size: int,
    normalize: bool = True,
) -> np.ndarray:
    cleaned = clean_lidar_ranges(ranges, min_range, max_range)
    downsampled = downsample_ranges(cleaned, output_size)
    if normalize:
        return normalize_ranges(downsampled, min_range, max_range)
    return downsampled
