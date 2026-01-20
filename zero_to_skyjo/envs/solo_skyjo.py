import random
from typing import List, Optional, Tuple, Dict

import numpy as np

from .spaces import Box, Discrete


class SoloSkyjoEnv:
    """Solo Skyjo with 2x3 grid, discard/deck choices, and column clears."""

    def __init__(
        self,
        seed: Optional[int] = None,
        max_actions: int = 40,
        action_penalty: int = 1,
        invalid_action_penalty: int = 2,
    ):
        self.rng = random.Random(seed)
        self.hand_size = 6
        self.columns = 3
        self.values = list(range(-2, 13))
        self.copies_per_value = 20
        self.max_actions = max_actions
        self.action_penalty = action_penalty
        self.invalid_action_penalty = invalid_action_penalty

        self.deck: List[int] = []
        self.discard: List[int] = []
        self.player_values: List[int] = []
        self.player_visible: List[bool] = []
        self.player_active: List[bool] = []
        self.action_count = 0
        self.done = False

        obs_dim = self.hand_size * (2 + (len(self.values) + 1)) + len(self.values)
        self.observation_space = Box(shape=(obs_dim,), low=0.0, high=1.0)
        self.action_space = Discrete(3 * self.hand_size)

    def seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)

        self.deck = []
        for v in self.values:
            self.deck += [v] * self.copies_per_value
        self.rng.shuffle(self.deck)

        self.player_values = [self.deck.pop() for _ in range(self.hand_size)]
        self.player_visible = [False] * self.hand_size
        self.player_active = [True] * self.hand_size

        self.discard = [self.deck.pop()]
        self.action_count = 0
        self.done = False

        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Call reset() before step(); episode is done.")

        info: Dict[str, object] = {}

        if action < 0 or action >= self.action_space.n:
            raise ValueError("Invalid action")

        action_type = action // self.hand_size
        slot = action % self.hand_size

        valid = self._is_valid_action(action_type, slot)
        invalid_penalty = 0.0
        if not valid:
            info["invalid_action"] = True
            invalid_penalty = float(self.invalid_action_penalty)
        else:
            if action_type == 0:
                self._take_discard_replace(slot)
            elif action_type == 1:
                self._draw_keep_replace(slot)
            else:
                self._draw_discard_reveal(slot)

            self._clear_matching_columns()

        self.action_count += 1

        terminated = self._all_visible() or self._active_count() == 0
        truncated = False
        if not terminated and self.action_count >= self.max_actions:
            truncated = True

        if terminated or truncated:
            self._reveal_all_active()
            self.done = True
            final_score = self._score()
            reward = -float(final_score + self.action_penalty * self.action_count + invalid_penalty)
            info.update(
                {
                    "final_score": final_score,
                    "action_count": self.action_count,
                    "remaining_cards": self._active_count(),
                }
            )
            return self._observe(), reward, terminated, truncated, info

        reward = -invalid_penalty if invalid_penalty > 0 else 0.0
        return self._observe(), reward, False, False, info

    def _is_valid_action(self, action_type: int, slot: int) -> bool:
        if not self.player_active[slot]:
            return False
        if action_type == 2 and self.player_visible[slot]:
            return False
        if action_type in (1, 2) and not self.deck:
            return False
        return True

    def _take_discard_replace(self, slot: int) -> None:
        taken = self.discard.pop()
        removed = self.player_values[slot]
        self.player_values[slot] = taken
        self.player_visible[slot] = True
        self.discard.append(removed)

    def _draw_keep_replace(self, slot: int) -> None:
        drawn = self.deck.pop()
        removed = self.player_values[slot]
        self.player_values[slot] = drawn
        self.player_visible[slot] = True
        self.discard.append(removed)

    def _draw_discard_reveal(self, slot: int) -> None:
        drawn = self.deck.pop()
        self.discard.append(drawn)
        self.player_visible[slot] = True

    def _clear_matching_columns(self) -> None:
        for col in range(self.columns):
            top = col
            bottom = col + self.columns
            if not (self.player_active[top] and self.player_active[bottom]):
                continue
            if not (self.player_visible[top] and self.player_visible[bottom]):
                continue
            if self.player_values[top] != self.player_values[bottom]:
                continue
            self.player_active[top] = False
            self.player_active[bottom] = False
            self.player_visible[top] = False
            self.player_visible[bottom] = False
            self.player_values[top] = 0
            self.player_values[bottom] = 0

    def _all_visible(self) -> bool:
        for active, visible in zip(self.player_active, self.player_visible):
            if active and not visible:
                return False
        return True

    def _active_count(self) -> int:
        return sum(1 for active in self.player_active if active)

    def _reveal_all_active(self) -> None:
        for i, active in enumerate(self.player_active):
            if active:
                self.player_visible[i] = True

    def _score(self) -> int:
        total = 0
        for value, active in zip(self.player_values, self.player_active):
            if active:
                total += value
        return total

    def _value_index(self, value: int) -> int:
        return value - self.values[0]

    def _observe(self) -> np.ndarray:
        hand_feats = []
        for active, visible, value in zip(
            self.player_active, self.player_visible, self.player_values
        ):
            hand_feats.append(1.0 if active else 0.0)
            hand_feats.append(1.0 if (active and visible) else 0.0)
            onehot = [0.0] * (len(self.values) + 1)
            if active:
                if visible:
                    onehot[self._value_index(value) + 1] = 1.0
                else:
                    onehot[0] = 1.0
            hand_feats.extend(onehot)

        discard_onehot = [0.0] * len(self.values)
        if self.discard:
            discard_onehot[self._value_index(self.discard[-1])] = 1.0

        obs = np.array(hand_feats + discard_onehot, dtype=np.float32)
        return obs

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        deck_available = len(self.deck) > 0
        discard_available = len(self.discard) > 0

        for slot in range(self.hand_size):
            if not self.player_active[slot]:
                continue

            if discard_available:
                mask[slot] = True

            if deck_available:
                mask[self.hand_size + slot] = True
                if not self.player_visible[slot]:
                    mask[2 * self.hand_size + slot] = True

        return mask

    def render(self) -> None:
        print("== SoloSkyjo ==")
        for row in range(2):
            row_vals = []
            for col in range(self.columns):
                idx = row * self.columns + col
                if not self.player_active[idx]:
                    row_vals.append("[  ]")
                    continue
                vis = self.player_visible[idx]
                val = self.player_values[idx]
                row_vals.append(f"[{'V' if vis else '?'}:{val if vis else 'X'}]")
            print(" ".join(row_vals))
        discard_val = self.discard[-1] if self.discard else None
        print(f"Discard: {discard_val} | Deck size: {len(self.deck)}")
        print(f"Actions: {self.action_count}/{self.max_actions}")
