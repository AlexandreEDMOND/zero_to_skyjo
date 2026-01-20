import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

from .spaces import Box, Discrete


@dataclass
class Offer:
    visible_value: int
    hidden_value: int


class MiniSkyjoEnv:
    """Mini Skyjo environment with a fixed action and observation format."""

    def __init__(self, reshuffle_returned: bool = True, seed: Optional[int] = None):
        self.reshuffle_returned = reshuffle_returned
        self.rng = random.Random(seed)
        self.n_values = 6
        self.copies_per_value = 6
        self.hand_size = 6

        self.deck: List[int] = []
        self.player_values: List[int] = []
        self.player_visible: List[bool] = []
        self.discard: List[int] = []
        self.offer: Optional[Offer] = None
        self.exchanges_count = 0
        self.done = False

        obs_dim = self.hand_size * (self.n_values + 2) + 2 * self.n_values + 2
        self.observation_space = Box(shape=(obs_dim,), low=0.0, high=1.0)
        self.action_space = Discrete(2 * self.hand_size)

    def seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)

        # Build deck
        self.deck = []
        for v in range(1, self.n_values + 1):
            self.deck += [v] * self.copies_per_value
        self.rng.shuffle(self.deck)

        # Draw player initial hand (face-down)
        self.player_values = [self.deck.pop() for _ in range(self.hand_size)]
        self.player_visible = [False] * self.hand_size

        self.discard = []
        self.exchanges_count = 0
        self.done = False

        # Prepare first offer if possible
        if len(self.deck) >= 2:
            vis = self.deck.pop()
            hid = self.deck.pop()
            self.offer = Offer(visible_value=vis, hidden_value=hid)
        else:
            self.offer = None
            self._reveal_all()
            self.done = True

        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Call reset() before step(); episode is done.")

        if action < 0 or action >= self.action_space.n:
            raise ValueError("Invalid action")

        offer_choice = action // self.hand_size  # 0: visible, 1: hidden
        slot = action % self.hand_size

        if self.offer is None:
            raise RuntimeError("No offer available during step()")

        taken_value = self.offer.visible_value if offer_choice == 0 else self.offer.hidden_value
        removed_value = self.player_values[slot]

        self.player_values[slot] = taken_value
        self.player_visible[slot] = True
        self.discard.append(removed_value)

        returned_value = self.offer.hidden_value if offer_choice == 0 else self.offer.visible_value
        self.deck.append(returned_value)
        if self.reshuffle_returned:
            self.rng.shuffle(self.deck)

        self.exchanges_count += 1

        # Termination: all visible or deck < 2 before next offer
        if all(self.player_visible) or len(self.deck) < 2:
            self._reveal_all()
            self.done = True
            final_score = sum(self.player_values) + self.exchanges_count
            reward = -float(final_score)
            info = {"final_score": final_score, "exchanges": self.exchanges_count}
            obs = self._observe_terminal()
            return obs, reward, True, False, info

        vis = self.deck.pop()
        hid = self.deck.pop()
        self.offer = Offer(visible_value=vis, hidden_value=hid)

        reward = 0.0
        info = {}
        return self._observe(), reward, False, False, info

    def _reveal_all(self) -> None:
        for i in range(self.hand_size):
            self.player_visible[i] = True

    def _deck_histogram(self) -> List[int]:
        hist = [0] * self.n_values
        for v in self.deck:
            hist[v - 1] += 1
        return hist

    def _observe(self) -> np.ndarray:
        hand_feats = []
        for vis, val in zip(self.player_visible, self.player_values):
            hand_feats.append(1.0 if vis else 0.0)
            if vis:
                onehot = [0.0] * (self.n_values + 1)
                onehot[val] = 1.0
            else:
                onehot = [0.0] * (self.n_values + 1)
                onehot[0] = 1.0
            hand_feats.extend(onehot)

        if self.offer is not None:
            offer_vis_onehot = [0.0] * self.n_values
            offer_vis_onehot[self.offer.visible_value - 1] = 1.0
            offer_hidden_flag = 1.0
        else:
            offer_vis_onehot = [0.0] * self.n_values
            offer_hidden_flag = 0.0

        deck_hist = self._deck_histogram()
        deck_size = len(self.deck)
        norm = 36.0
        deck_hist_norm = [h / norm for h in deck_hist]
        deck_size_norm = [deck_size / norm]

        obs = np.array(
            hand_feats + offer_vis_onehot + [offer_hidden_flag] + deck_hist_norm + deck_size_norm,
            dtype=np.float32,
        )
        return obs

    def _observe_terminal(self) -> np.ndarray:
        hand_feats = []
        for vis, val in zip(self.player_visible, self.player_values):
            hand_feats.append(1.0 if vis else 0.0)
            onehot = [0.0] * (self.n_values + 1)
            onehot[val] = 1.0
            hand_feats.extend(onehot)

        offer_vis_onehot = [0.0] * self.n_values
        offer_hidden_flag = 0.0
        deck_hist = self._deck_histogram()
        deck_size = len(self.deck)
        norm = 36.0
        deck_hist_norm = [h / norm for h in deck_hist]
        deck_size_norm = [deck_size / norm]

        obs = np.array(
            hand_feats + offer_vis_onehot + [offer_hidden_flag] + deck_hist_norm + deck_size_norm,
            dtype=np.float32,
        )
        return obs

    def render(self) -> None:
        print("== MiniSkyjo ==")
        print("Hand:")
        for i, (v, vis) in enumerate(zip(self.player_values, self.player_visible)):
            label = f"[{i}] {'V' if vis else '?'}:{v if vis else 'X'}"
            print(label, end="  ")
        print("\nDiscard top:", self.discard[-5:])
        if self.offer:
            print(
                f"Offer visible={self.offer.visible_value}, hidden=? | Deck size={len(self.deck)}"
            )
        else:
            print("No offer")
        print(f"Exchanges: {self.exchanges_count}")
