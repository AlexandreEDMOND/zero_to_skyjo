from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class ReplayBuffer:
    capacity: int
    obs: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    next_obs: List[np.ndarray]
    dones: List[bool]
    idx: int = 0
    full: bool = False

    @classmethod
    def create(cls, capacity: int) -> "ReplayBuffer":
        return cls(capacity, [], [], [], [], [], 0, False)

    def push(self, obs, action, reward, next_obs, done) -> None:
        if len(self.obs) < self.capacity:
            self.obs.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_obs.append(next_obs)
            self.dones.append(done)
        else:
            self.obs[self.idx] = obs
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward
            self.next_obs[self.idx] = next_obs
            self.dones[self.idx] = done
            self.full = True
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        size = self.capacity if self.full else len(self.obs)
        idxs = np.random.randint(0, size, size=batch_size)
        o = torch.tensor(np.stack([self.obs[i] for i in idxs]), dtype=torch.float32, device=device)
        a = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long, device=device)
        r = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32, device=device)
        o2 = torch.tensor(
            np.stack([self.next_obs[i] for i in idxs]), dtype=torch.float32, device=device
        )
        d = torch.tensor([self.dones[i] for i in idxs], dtype=torch.float32, device=device)
        return o, a, r, o2, d

    def __len__(self) -> int:
        return self.capacity if self.full else len(self.obs)
