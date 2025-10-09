# mini_skyjo_rl.py
#
# A single-file prototype to simulate your API, train a tiny DQN on CPU,
# and demo a terminal playthrough. It includes:
# - MiniSkyjoEnv: environment matching your rules
# - DQN agent (PyTorch, CPU)
# - Training loop with learning curve (saved to learning_curve.png and metrics.csv)
# - Demo mode to print a sample episode with the trained model
#
# Usage:
#   python mini_skyjo_rl.py --train --episodes 5000
#   python mini_skyjo_rl.py --demo --model_path model.pt
#
# You can also just run:
#   python mini_skyjo_rl.py --train --demo
#
import argparse
import random
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====== ENVIRONMENT ======
# Rules:
# - Deck: values 1..6, 6 copies each => 36 cards total.
# - Player has 6 cards face-down at start (values known to env, hidden to agent unless revealed).
# - Each turn, offer two cards: one visible (value known) and one hidden (value unknown to agent).
# - Agent must pick one of the two offers and a slot among 6 to replace.
#   The chosen card becomes VISIBLE in that slot.
#   The removed card goes to discard (and its value becomes visible).
#   The unchosen offered card is returned to the deck and we reshuffle (simple i.i.d. assumption).
# - End when all 6 player cards are visible OR deck size < 2 (before next offer).
# - Final score = sum(all 6 values) + exchanges_count. Reward = -final_score at terminal, 0 otherwise.
#
# Observation (MDP informed to bootstrap learning):
#  - player hand: for each of 6 slots: [is_visible, one-hot(value 1..6 or 0 if hidden)]
#  - offer: one-hot(offer_visible_value 1..6) + offer_hidden_flag (always 1 during a live offer)
#  - deck histogram (counts of remaining values 1..6, normalized by 36)
#  - deck_size normalized by 36
#
# Action space: 12 discrete actions = 2 (which offer) * 6 (which slot)
#   index = offer_choice * 6 + slot, offer_choice in {0 (visible), 1 (hidden)}, slot in {0..5}
#


@dataclass
class Offer:
    visible_value: int  # 1..6
    hidden_value: int   # 1..6, unknown to the agent


class MiniSkyjoEnv:
    def __init__(self, reshuffle_returned=True, seed: Optional[int] = None):
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

    def seed(self, seed: int):
        self.rng.seed(seed)

    def reset(self) -> np.ndarray:
        # Build deck
        self.deck = []
        for v in range(1, self.n_values + 1):
            self.deck += [v] * self.copies_per_value
        self.rng.shuffle(self.deck)

        # Draw player initial 6 cards (face-down)
        self.player_values = [self.deck.pop() for _ in range(self.hand_size)]
        self.player_visible = [False] * self.hand_size

        self.discard = []
        self.exchanges_count = 0
        self.done = False

        # Prepare first offer if possible; else, end immediately
        if len(self.deck) >= 2:
            vis = self.deck.pop()
            hid = self.deck.pop()
            self.offer = Offer(visible_value=vis, hidden_value=hid)
        else:
            # End (rare edge case)
            self.offer = None
            self._reveal_all()
            self.done = True

        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Call reset() before step(); episode is done.")

        # Decode action
        if action < 0 or action >= 12:
            raise ValueError("Invalid action")
        offer_choice = action // 6  # 0: visible, 1: hidden
        slot = action % 6
        if self.offer is None:
            raise RuntimeError("No offer available during step()")

        # Take offered card and replace chosen slot
        taken_value = self.offer.visible_value if offer_choice == 0 else self.offer.hidden_value
        removed_value = self.player_values[slot]

        # The chosen card becomes visible in that slot
        self.player_values[slot] = taken_value
        self.player_visible[slot] = True

        # The removed card goes to discard (now visible)
        self.discard.append(removed_value)

        # The unchosen offered card returns to deck
        returned_value = self.offer.hidden_value if offer_choice == 0 else self.offer.visible_value
        self.deck.append(returned_value)
        if self.reshuffle_returned:
            self.rng.shuffle(self.deck)

        self.exchanges_count += 1

        # Check termination: all visible or deck < 2 before next offer
        if all(self.player_visible) or len(self.deck) < 2:
            self._reveal_all()  # reveal any remaining hidden cards per your rule
            self.done = True
            final_score = sum(self.player_values) + self.exchanges_count
            reward = -float(final_score)
            info = {"final_score": final_score, "exchanges": self.exchanges_count}
            obs = self._observe_terminal()
            return obs, reward, True, info

        # Otherwise, generate next offer
        vis = self.deck.pop()
        hid = self.deck.pop()
        self.offer = Offer(visible_value=vis, hidden_value=hid)

        # Non-terminal step
        reward = 0.0
        info = {}
        return self._observe(), reward, False, info

    def _reveal_all(self):
        for i in range(self.hand_size):
            self.player_visible[i] = True

    def _deck_histogram(self) -> List[int]:
        hist = [0] * self.n_values
        for v in self.deck:
            hist[v - 1] += 1
        return hist

    def _observe(self) -> np.ndarray:
        # Build observation vector
        # Hand features: 6 * (is_visible + one-hot(0..6))
        hand_feats = []
        for vis, val in zip(self.player_visible, self.player_values):
            hand_feats.append(1.0 if vis else 0.0)
            if vis:
                onehot = [0.0] * (self.n_values + 1)  # index 0 unused for visible
                onehot[val] = 1.0
            else:
                onehot = [0.0] * (self.n_values + 1)
                onehot[0] = 1.0  # 0 indicates unknown value
            hand_feats.extend(onehot)

        # Offer features
        if self.offer is not None:
            offer_vis_onehot = [0.0] * self.n_values
            offer_vis_onehot[self.offer.visible_value - 1] = 1.0
            offer_hidden_flag = 1.0
        else:
            offer_vis_onehot = [0.0] * self.n_values
            offer_hidden_flag = 0.0

        # Deck histogram (normalized)
        deck_hist = self._deck_histogram()
        deck_size = len(self.deck)
        norm = 36.0
        deck_hist_norm = [h / norm for h in deck_hist]
        deck_size_norm = [deck_size / norm]

        obs = np.array(hand_feats + offer_vis_onehot + [offer_hidden_flag] + deck_hist_norm + deck_size_norm, dtype=np.float32)
        return obs

    def _observe_terminal(self) -> np.ndarray:
        # Same shape as _observe, but offer becomes zeroed and hidden flag 0
        hand_feats = []
        for vis, val in zip(self.player_visible, self.player_values):
            hand_feats.append(1.0 if vis else 0.0)
            onehot = [0.0] * (self.n_values + 1)
            onehot[val] = 1.0  # now every card is visible
            hand_feats.extend(onehot)

        offer_vis_onehot = [0.0] * self.n_values
        offer_hidden_flag = 0.0
        deck_hist = self._deck_histogram()
        deck_size = len(self.deck)
        norm = 36.0
        deck_hist_norm = [h / norm for h in deck_hist]
        deck_size_norm = [deck_size / norm]

        obs = np.array(hand_feats + offer_vis_onehot + [offer_hidden_flag] + deck_hist_norm + deck_size_norm, dtype=np.float32)
        return obs

    def render(self):
        print("== MiniSkyjo ==")
        print("Hand:")
        for i, (v, vis) in enumerate(zip(self.player_values, self.player_visible)):
            s = f"[{i}] {'V' if vis else '?'}:{v if vis else 'X'}"
            print(s, end='  ')
        print("\nDiscard top:", self.discard[-5:])
        if self.offer:
            print(f"Offer visible={self.offer.visible_value}, hidden=?    | Deck size={len(self.deck)}")
        else:
            print("No offer")
        print(f"Exchanges: {self.exchanges_count}")


# ====== DQN AGENT ======

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # V(s)
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # A(s, a)
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)                 # (B, 1)
        a = self.adv(h)                   # (B, nA)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q



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
    def create(cls, capacity: int):
        return cls(capacity, [], [], [], [], [], 0, False)

    def push(self, o, a, r, o2, d):
        if len(self.obs) < self.capacity:
            self.obs.append(o)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_obs.append(o2)
            self.dones.append(d)
        else:
            self.obs[self.idx] = o
            self.actions[self.idx] = a
            self.rewards[self.idx] = r
            self.next_obs[self.idx] = o2
            self.dones[self.idx] = d
            self.full = True
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        size = self.capacity if self.full else len(self.obs)
        idxs = np.random.randint(0, size, size=batch_size)
        o = torch.tensor(np.stack([self.obs[i] for i in idxs]), dtype=torch.float32)
        a = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long)
        r = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32)
        o2 = torch.tensor(np.stack([self.next_obs[i] for i in idxs]), dtype=torch.float32)
        d = torch.tensor([self.dones[i] for i in idxs], dtype=torch.float32)
        return o, a, r, o2, d

    def __len__(self):
        return self.capacity if self.full else len(self.obs)


def select_action(qnet: DuelingQNet, obs: np.ndarray, n_actions: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q = qnet(x).squeeze(0)
        if epsilon < 0.2:
            q = q + 0.01 * torch.randn_like(q)  # petit bruit
        return int(torch.argmax(q).item())

def train_dqn(episodes=20000, buffer_capacity=100000, batch_size=256, gamma=0.995,
              lr=3e-4, target_update=500, start_learning=1000, epsilon_start=1.0,
              epsilon_end=0.15, epsilon_decay_steps=300000, eval_every=1000,
              model_path="model.pt", metrics_path="metrics.csv"):
    env = MiniSkyjoEnv(seed=123)
    obs = env.reset()
    obs_dim = obs.shape[0]
    n_actions = 12

    q = DuelingQNet(obs_dim, n_actions)
    q_target = DuelingQNet(obs_dim, n_actions)
    q_target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=lr, weight_decay=1e-5)  # léger L2


    rb = ReplayBuffer.create(buffer_capacity)

    global_step = 0
    epsilon = epsilon_start
    best_avg_score = 1000

    # Metrics
    ep_rewards = []
    ep_scores = []
    ep_lengths = []
    moving = []
    moving_steps = []

    def epsilon_by_step(step):
        t = min(1.0, step / float(epsilon_decay_steps))
        return epsilon_start + t * (epsilon_end - epsilon_start)

    for ep in tqdm(range(1, episodes + 1), desc="Training"):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        final_score = None

        while not done:
            epsilon = epsilon_by_step(global_step)
            action = select_action(q, obs, n_actions, epsilon)
            next_obs, reward, done, info = env.step(action)
            rb.push(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward
            steps += 1
            global_step += 1

            # Learn
            if len(rb) >= start_learning:
                o, a, r, o2, d = rb.sample(batch_size)
                with torch.no_grad():
                    # Double DQN: action selection by online net, evaluation by target net
                    q_next_online = q(o2)
                    next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)  # (B,1)

                    q_next_target = q_target(o2)
                    max_next = q_next_target.gather(1, next_actions).squeeze(1)      # (B,)

                    target = r + gamma * (1.0 - d) * max_next

                q_values = q(o)
                q_a = q_values.gather(1, a.view(-1, 1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_a, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)  # clip pour stabilité
                optimizer.step()

                if global_step % target_update == 0:
                    q_target.load_state_dict(q.state_dict())


        if "final_score" in info:
            final_score = info["final_score"]
        ep_rewards.append(total_reward)
        ep_scores.append(final_score if final_score is not None else math.nan)
        ep_lengths.append(steps)

        # Moving average of scores for quick visual
        window = 50
        if len(ep_scores) >= window:
            moving.append(sum(ep_scores[-window:]) / window)
            moving_steps.append(sum(ep_lengths[-window:]) / window)
        else:
            moving.append(sum(ep_scores) / max(1, len(ep_scores)))
            moving_steps.append(sum(ep_lengths) / max(1, len(ep_lengths)))

        if ep % eval_every == 0:
            print(f"[Episode {ep:5d}] avg_score(last50)={moving[-1]:.2f}  epsilon={epsilon:.3f}  avg_steps(last50)={moving_steps[-1]:.2f} ")
            if moving[-1] < best_avg_score:
                best_avg_score =  moving[-1]
                torch.save(q.state_dict(), model_path)
                print(f"Saved model to {model_path}")

    # Save model
    torch.save(q.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save metrics CSV
    import csv
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "final_score", "total_reward", "ep_length", "moving_avg_score_50"])
        for i, (s, r, l, m) in enumerate(zip(ep_scores, ep_rewards, ep_lengths, moving), start=1):
            w.writerow([i, s, r, l, m])
    print(f"Saved metrics to {metrics_path}")

    # Plot learning curve (moving average of final score)
    plt.figure()
    plt.plot(moving, label="Moving Avg Final Score (50 ep)")
    plt.xlabel("Episode")
    plt.ylabel("Score (lower is better)")
    plt.title("Learning Curve - MiniSkyjo DQN (CPU)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    print("Saved plot to learning_curve.png")

    # Plot average steps per episode
    plt.figure()
    plt.plot(moving_steps, label="Moving Avg Episode Length (50 ep)")
    plt.xlabel("Episode")
    plt.ylabel("Length (steps)")
    plt.title("Episode Lengths - MiniSkyjo DQN (CPU)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("episode_lengths.png")
    print("Saved plot to episode_lengths.png")


def demo(model_path="model.pt", seed=999):
    env = MiniSkyjoEnv()
    obs = env.reset()
    obs_dim = obs.shape[0]
    n_actions = 12
    q = DuelingQNet(obs_dim, n_actions)
    if os.path.exists(model_path):
        q.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model from {model_path}")
    else:
        print("Model not found, using untrained weights.")

    done = False
    step_idx = 0
    while not done:
        print("\n--- STEP", step_idx, "---")
        env.render()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = q(x).squeeze(0).numpy()
        action = int(np.argmax(q_values))
        offer_choice = action // 6  # 0 visible, 1 hidden
        slot = action % 6
        print(f"Agent action: take {'VISIBLE' if offer_choice == 0 else 'HIDDEN'} -> replace slot {slot}")

        obs, reward, done, info = env.step(action)
        step_idx += 1

    print("\n=== EPISODE END ===")
    env.render()
    if "final_score" in info:
        print(f"Final score: {info['final_score']}  (exchanges={info['exchanges']})")


def evaluate(model_path="model.pt", episodes=1000):
    env = MiniSkyjoEnv()
    obs = env.reset()
    obs_dim = obs.shape[0]
    n_actions = 12

    # Charger le modèle
    q = DuelingQNet(obs_dim, n_actions)   # ou QNet si tu n’as pas encore switché
    q.load_state_dict(torch.load(model_path, map_location="cpu"))
    q.eval()

    scores = []
    steps_list = []

    for ep in tqdm(range(episodes)):
        obs = env.reset()
        done = False
        steps = 0
        final_score = None

        while not done:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_values = q(x).squeeze(0)
                action = int(torch.argmax(q_values).item())  # greedy policy

            obs, reward, done, info = env.step(action)
            steps += 1

            if done and "final_score" in info:
                final_score = info["final_score"]

        scores.append(final_score)
        steps_list.append(steps)

    avg_score = sum(scores) / len(scores)
    avg_steps = sum(steps_list) / len(steps_list)

    print(f"Evaluation over {episodes} games:")
    print(f"  Average score: {avg_score:.2f}")
    print(f"  Average steps: {avg_steps:.2f}")

    return {"avg_score": avg_score, "avg_steps": avg_steps, "scores": scores, "steps": steps_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train DQN on CPU")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes")
    parser.add_argument("--demo", action="store_true", help="Run a terminal demo with the trained model")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to save/load the model")
    parser.add_argument("--metrics_path", type=str, default="metrics.csv", help="CSV metrics path")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained model")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Number of eval episodes")
    args = parser.parse_args()

    if args.train:
        train_dqn(episodes=args.episodes, model_path=args.model_path, metrics_path=args.metrics_path)
    if args.demo:
        demo(model_path=args.model_path)
    if args.eval:
        evaluate(model_path=args.model_path, episodes=args.eval_episodes)


if __name__ == "__main__":
    main()