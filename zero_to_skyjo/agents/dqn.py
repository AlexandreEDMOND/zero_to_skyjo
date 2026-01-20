import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


class DQNAgent:
    requires_model = True

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 512,
        lr: float = 3e-4,
        gamma: float = 0.995,
        target_update: int = 500,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        device: str = "cpu",
        action_noise_std: float = 0.01,
        action_noise_epsilon: float = 0.2,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update = target_update
        self.grad_clip = grad_clip
        self.device = device
        self.action_noise_std = action_noise_std
        self.action_noise_epsilon = action_noise_epsilon

        self.q = DuelingQNet(obs_dim, n_actions, hidden=hidden).to(device)
        self.q_target = DuelingQNet(obs_dim, n_actions, hidden=hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)
        self.update_steps = 0

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float,
        explore: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        valid_actions = None
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            valid_actions = np.flatnonzero(mask)
            if valid_actions.size == 0:
                action_mask = None
                valid_actions = None

        if explore and random.random() < epsilon:
            if valid_actions is not None:
                return int(np.random.choice(valid_actions))
            return random.randrange(self.n_actions)

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q(x).squeeze(0)
            if explore and epsilon < self.action_noise_epsilon and self.action_noise_std > 0:
                q_values = q_values + self.action_noise_std * torch.randn_like(q_values)
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~mask_t, float("-inf"))
            return int(torch.argmax(q_values).item())

    def update(self, batch) -> float:
        o, a, r, o2, d = batch

        with torch.no_grad():
            q_next_online = self.q(o2)
            next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
            q_next_target = self.q_target(o2)
            max_next = q_next_target.gather(1, next_actions).squeeze(1)
            target = r + self.gamma * (1.0 - d) * max_next

        q_values = self.q(o)
        q_a = q_values.gather(1, a.view(-1, 1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(self.q.state_dict())
