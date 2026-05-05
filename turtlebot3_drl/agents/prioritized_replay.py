from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np


class SumTree:
    """Binary sum tree for Prioritized Experience Replay."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
        self.data: List[Any] = [None] * self.capacity
        self.write_index = 0
        self.size = 0

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> int:
        data_index = self.write_index
        tree_index = data_index + self.capacity - 1
        self.data[data_index] = data
        self.update(tree_index, priority)
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_index

    def update(self, tree_index: int, priority: float) -> None:
        priority = max(float(priority), 0.0)
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, sample_value: float) -> Tuple[int, float, Any]:
        sample_value = min(max(float(sample_value), 0.0), max(self.total_priority - 1e-8, 0.0))
        index = 0
        while True:
            left = 2 * index + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if sample_value <= self.tree[left]:
                index = left
            else:
                sample_value -= self.tree[left]
                index = right
        data_index = index - self.capacity + 1
        return index, float(self.tree[index]), self.data[data_index]


@dataclass
class PrioritizedBatch:
    indices: np.ndarray
    weights: np.ndarray
    transitions: Sequence[Any]


class PrioritizedReplayBuffer:
    """PER buffer using the paper priority delta_i = (|td_error| + eps)^alpha."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = max(int(beta_frames), 1)
        self.epsilon = float(epsilon)
        self.frame = 1
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def _priority_from_td_error(self, td_error: float) -> float:
        return (abs(float(td_error)) + self.epsilon) ** self.alpha

    def beta_by_frame(self, frame: int | None = None) -> float:
        frame = self.frame if frame is None else frame
        fraction = min(float(frame) / self.beta_frames, 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def add(self, transition: Any, td_error: float | None = None) -> None:
        priority = self.max_priority if td_error is None else self._priority_from_td_error(td_error)
        self.tree.add(priority, transition)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> PrioritizedBatch:
        if len(self) == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        batch_size = min(int(batch_size), len(self))
        valid_leaf_start = self.tree.capacity - 1
        valid_leaf_end = valid_leaf_start + len(self)
        valid_priorities = self.tree.tree[valid_leaf_start:valid_leaf_end]
        total = float(np.sum(valid_priorities))
        if total <= 0.0:
            raise ValueError("total priority is zero")

        segment = total / batch_size
        beta = self.beta_by_frame()
        self.frame += 1

        indices = []
        priorities = []
        transitions = []
        for i in range(batch_size):
            transition = None
            priority = 0.0
            index = -1
            for _ in range(10):
                low = segment * i
                high = segment * (i + 1)
                value = np.random.uniform(low, high)
                index, priority, transition = self.tree.get(value)
                if transition is not None and priority > 0.0:
                    break
            if transition is None or priority <= 0.0:
                data_index = int(np.random.randint(0, len(self)))
                index = data_index + self.tree.capacity - 1
                priority = float(self.tree.tree[index])
                transition = self.tree.data[data_index]
            if transition is None or priority <= 0.0:
                raise RuntimeError("sampled an invalid replay transition")
            indices.append(index)
            priorities.append(priority)
            transitions.append(transition)

        probabilities = np.maximum(np.asarray(priorities, dtype=np.float32) / total, self.epsilon)
        weights = (len(self) * probabilities) ** (-beta)
        weights /= max(float(np.max(weights)), 1e-6)
        return PrioritizedBatch(
            indices=np.asarray(indices, dtype=np.int64),
            weights=weights.astype(np.float32),
            transitions=transitions,
        )

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        for index, td_error in zip(indices, td_errors):
            priority = self._priority_from_td_error(float(td_error))
            self.tree.update(int(index), priority)
            self.max_priority = max(self.max_priority, priority)
