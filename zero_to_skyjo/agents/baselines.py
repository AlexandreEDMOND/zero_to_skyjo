from typing import List, Optional

import numpy as np


class RandomAgent:
    requires_model = False

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        explore: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        if action_mask is not None:
            valid = np.flatnonzero(action_mask)
            if valid.size > 0:
                return int(self.rng.choice(valid))
        return int(self.rng.integers(0, self.n_actions))

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None


class GreedyMinAgent:
    requires_model = False

    def __init__(
        self,
        n_actions: int,
        hand_size: int = 6,
        min_value: int = -2,
        max_value: int = 12,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.hand_size = hand_size
        self.values = list(range(min_value, max_value + 1))
        self.n_values = len(self.values)
        self.slot_stride = 2 + (self.n_values + 1)
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        explore: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        discard_value = self._discard_value(obs)
        visible_slots, hidden_slots = self._slot_info(obs)

        action = None
        if discard_value is not None:
            best = self._best_visible_slot(visible_slots, discard_value)
            if best is not None:
                action = best
            elif hidden_slots:
                action = hidden_slots[0]
            elif visible_slots:
                action = max(visible_slots, key=lambda x: x[1])[0]

        if action is None:
            return self._fallback(action_mask)

        action = int(action)
        if action_mask is not None and not action_mask[action]:
            return self._fallback(action_mask)
        return action

    def _best_visible_slot(self, visible_slots: List[tuple], discard_value: int) -> Optional[int]:
        candidates = [(slot, value) for slot, value in visible_slots if value > discard_value]
        if not candidates:
            return None
        slot = max(candidates, key=lambda x: x[1])[0]
        return self._action_discard_replace(slot)

    def _slot_info(self, obs: np.ndarray) -> tuple:
        visible_slots = []
        hidden_slots = []
        for slot in range(self.hand_size):
            base = slot * self.slot_stride
            active = obs[base] > 0.5
            visible = obs[base + 1] > 0.5
            if not active:
                continue
            if visible:
                onehot = obs[base + 2 : base + 2 + self.n_values + 1]
                idx = int(np.argmax(onehot))
                if idx > 0:
                    value = self.values[idx - 1]
                    visible_slots.append((slot, value))
            else:
                hidden_slots.append(slot)
        return visible_slots, hidden_slots

    def _discard_value(self, obs: np.ndarray) -> Optional[int]:
        start = self.hand_size * self.slot_stride
        onehot = obs[start : start + self.n_values]
        if onehot.size == 0 or np.all(onehot == 0):
            return None
        idx = int(np.argmax(onehot))
        return self.values[idx]

    def _action_discard_replace(self, slot: int) -> int:
        return slot

    def _fallback(self, action_mask: Optional[np.ndarray]) -> int:
        if action_mask is not None:
            valid = np.flatnonzero(action_mask)
            if valid.size > 0:
                return int(self.rng.choice(valid))
        return int(self.rng.integers(0, self.n_actions))

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None


class GreedyThresholdAgent(GreedyMinAgent):
    requires_model = False

    def __init__(
        self,
        n_actions: int,
        hand_size: int = 6,
        min_value: int = -2,
        max_value: int = 12,
        threshold: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__(n_actions, hand_size, min_value, max_value, seed)
        self.threshold = threshold

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        explore: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        discard_value = self._discard_value(obs)
        visible_slots, hidden_slots = self._slot_info(obs)

        if discard_value is not None and discard_value > self.threshold:
            if hidden_slots:
                action = 2 * self.hand_size + hidden_slots[0]
            elif visible_slots:
                slot = max(visible_slots, key=lambda x: x[1])[0]
                action = self.hand_size + slot
            else:
                action = None
        else:
            action = None
            if discard_value is not None:
                best = self._best_visible_slot(visible_slots, discard_value)
                if best is not None:
                    action = best
                elif hidden_slots:
                    action = hidden_slots[0]
                elif visible_slots:
                    action = max(visible_slots, key=lambda x: x[1])[0]

        if action is None:
            return self._fallback(action_mask)

        action = int(action)
        if action_mask is not None and not action_mask[action]:
            return self._fallback(action_mask)
        return action
