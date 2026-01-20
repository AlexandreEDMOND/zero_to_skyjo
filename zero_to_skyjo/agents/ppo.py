from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature(x)
        return self.policy(h), self.value(h)


class PPOAgent:
    requires_model = True

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.model = ActorCritic(obs_dim, n_actions, hidden=hidden).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _apply_action_mask(
        self, logits: torch.Tensor, action_mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        if action_mask is None:
            return logits
        mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
        if mask.sum() == 0:
            return logits
        return logits.masked_fill(~mask, -1e9)

    def act(
        self,
        obs: np.ndarray,
        explore: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs_t)
        logits = self._apply_action_mask(logits, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.squeeze(0).item())

    def select_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        explore: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        action, _, _ = self.act(obs, explore=explore, action_mask=action_mask)
        return action

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * next_nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns

    def update(
        self,
        batch: Dict[str, np.ndarray],
        epochs: int = 4,
        minibatch_size: int = 256,
    ) -> Dict[str, float]:
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        losses = []
        for _ in range(epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                new_log_probs, entropy, values = self.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = -torch.min(ratio * mb_advantages, clipped * mb_advantages).mean()
                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses.append(loss.item())

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
        }

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
