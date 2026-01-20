import pickle
from typing import Optional, Tuple

import numpy as np


class TabularQAgent:
    requires_model = True

    def __init__(
        self,
        n_actions: int,
        hand_size: int = 6,
        min_value: int = -2,
        max_value: int = 12,
        alpha: float = 0.1,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.hand_size = hand_size
        self.values = list(range(min_value, max_value + 1))
        self.n_values = len(self.values)
        self.slot_stride = 2 + (self.n_values + 1)
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q = {}

    def encode_state(self, obs: np.ndarray) -> Tuple[int, ...]:
        state = []
        for slot in range(self.hand_size):
            base = slot * self.slot_stride
            active = obs[base] > 0.5
            visible = obs[base + 1] > 0.5
            if not active:
                state.append(0)
                continue
            if not visible:
                state.append(1)
                continue
            onehot = obs[base + 2 : base + 2 + self.n_values + 1]
            idx = int(np.argmax(onehot))
            if idx == 0:
                state.append(1)
            else:
                state.append(2 + (idx - 1))

        start = self.hand_size * self.slot_stride
        discard_onehot = obs[start : start + self.n_values]
        if discard_onehot.size == 0 or np.all(discard_onehot == 0):
            state.append(0)
        else:
            idx = int(np.argmax(discard_onehot))
            state.append(1 + idx)

        return tuple(state)

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.1,
        explore: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        state = self.encode_state(obs)
        if explore and self.rng.random() < epsilon:
            return self._random_action(action_mask)

        q_values = self._get_q(state)
        if action_mask is not None:
            valid = np.flatnonzero(action_mask)
            if valid.size > 0:
                best = valid[np.argmax(q_values[valid])]
                return int(best)
        return int(np.argmax(q_values))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        state = self.encode_state(obs)
        next_state = self.encode_state(next_obs)

        q_values = self._get_q(state)
        target = reward
        if not done:
            next_q = self._get_q(next_state)
            if next_action_mask is not None:
                valid = np.flatnonzero(next_action_mask)
                if valid.size > 0:
                    target = reward + self.gamma * float(np.max(next_q[valid]))
                else:
                    target = reward + self.gamma * float(np.max(next_q))
            else:
                target = reward + self.gamma * float(np.max(next_q))

        q_values[action] += self.alpha * (target - q_values[action])

    def _get_q(self, state: Tuple[int, ...]) -> np.ndarray:
        q_values = self.q.get(state)
        if q_values is None:
            q_values = np.zeros(self.n_actions, dtype=np.float32)
            self.q[state] = q_values
        return q_values

    def _random_action(self, action_mask: Optional[np.ndarray]) -> int:
        if action_mask is not None:
            valid = np.flatnonzero(action_mask)
            if valid.size > 0:
                return int(self.rng.choice(valid))
        return int(self.rng.integers(0, self.n_actions))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.q = pickle.load(f)
