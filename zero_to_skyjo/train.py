import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import tomllib
from tqdm import tqdm

from zero_to_skyjo.registry import make_agent, make_env
from zero_to_skyjo.utils import ReplayBuffer, set_seed


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def make_run_dir(runs_dir: Path, exp_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config_snapshot(config_path: Path, run_dir: Path) -> None:
    content = config_path.read_text(encoding="utf-8")
    (run_dir / "config.toml").write_text(content, encoding="utf-8")


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    t = min(1.0, step / float(decay_steps))
    return start + t * (end - start)


def _save_metrics(
    run_dir: Path,
    window: int,
    ep_scores: list,
    ep_rewards: list,
    ep_lengths: list,
    moving_scores: list,
    moving_lengths: list,
) -> None:
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "final_score",
                "total_reward",
                "ep_length",
                f"moving_avg_score_{window}",
            ]
        )
        for i, (s, r, l, m) in enumerate(
            zip(ep_scores, ep_rewards, ep_lengths, moving_scores), start=1
        ):
            writer.writerow([i, s, r, l, m])
    print(f"Saved metrics to {metrics_path}")

    plt.figure()
    plt.plot(moving_scores, label=f"Moving Avg Final Score ({window} ep)")
    plt.xlabel("Episode")
    plt.ylabel("Score (lower is better)")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    curve_path = run_dir / "learning_curve.png"
    plt.savefig(curve_path)
    print(f"Saved plot to {curve_path}")

    plt.figure()
    plt.plot(moving_lengths, label=f"Moving Avg Episode Length ({window} ep)")
    plt.xlabel("Episode")
    plt.ylabel("Length (steps)")
    plt.title("Episode Lengths")
    plt.legend()
    plt.tight_layout()
    length_path = run_dir / "episode_lengths.png"
    plt.savefig(length_path)
    print(f"Saved plot to {length_path}")


def _train_dqn(
    env,
    agent,
    train_cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    buffer_capacity = int(train_cfg.get("buffer_capacity", 100000))
    batch_size = int(train_cfg.get("batch_size", 256))
    start_learning = int(train_cfg.get("start_learning", 1000))
    episodes = int(train_cfg.get("episodes", 20000))
    epsilon_start = float(train_cfg.get("epsilon_start", 1.0))
    epsilon_end = float(train_cfg.get("epsilon_end", 0.15))
    epsilon_decay_steps = int(train_cfg.get("epsilon_decay_steps", 300000))
    eval_every = int(train_cfg.get("eval_every", 1000))
    window = int(train_cfg.get("plot_window", 50))

    rb = ReplayBuffer.create(buffer_capacity)

    global_step = 0
    best_avg_score = float("inf")

    ep_rewards = []
    ep_scores = []
    ep_lengths = []
    moving_scores = []
    moving_lengths = []

    best_model_path = run_dir / "model_best.pt"
    final_model_path = run_dir / "model_final.pt"

    for ep in tqdm(range(1, episodes + 1), desc="Training"):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info = {}

        while not done:
            epsilon = epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)
            action_mask = env.action_mask() if hasattr(env, "action_mask") else None
            action = agent.select_action(obs, epsilon, explore=True, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rb.push(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward
            steps += 1
            global_step += 1

            if len(rb) >= start_learning:
                batch = rb.sample(batch_size, device=agent.device)
                agent.update(batch)

        final_score = info.get("final_score")
        ep_rewards.append(total_reward)
        ep_scores.append(final_score if final_score is not None else np.nan)
        ep_lengths.append(steps)

        if len(ep_scores) >= window:
            moving_scores.append(float(np.mean(ep_scores[-window:])))
            moving_lengths.append(float(np.mean(ep_lengths[-window:])))
        else:
            moving_scores.append(float(np.mean(ep_scores)))
            moving_lengths.append(float(np.mean(ep_lengths)))

        if eval_every > 0 and ep % eval_every == 0:
            avg_score = moving_scores[-1]
            print(
                f"[Episode {ep:5d}] avg_score(last{window})={avg_score:.2f} "
                f"epsilon={epsilon:.3f} avg_steps(last{window})={moving_lengths[-1]:.2f}"
            )
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                agent.save(str(best_model_path))
                print(f"Saved best model to {best_model_path}")

    agent.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")

    _save_metrics(run_dir, window, ep_scores, ep_rewards, ep_lengths, moving_scores, moving_lengths)


def _train_ppo(
    env,
    agent,
    train_cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    episodes = int(train_cfg.get("episodes", 20000))
    eval_every = int(train_cfg.get("eval_every", 1000))
    window = int(train_cfg.get("plot_window", 50))
    ppo_epochs = int(train_cfg.get("ppo_epochs", 4))
    minibatch_size = int(train_cfg.get("ppo_minibatch_size", 256))

    best_avg_score = float("inf")

    ep_rewards = []
    ep_scores = []
    ep_lengths = []
    moving_scores = []
    moving_lengths = []

    best_model_path = run_dir / "model_best.pt"
    final_model_path = run_dir / "model_final.pt"

    for ep in tqdm(range(1, episodes + 1), desc="Training"):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info = {}

        obs_buf = []
        actions_buf = []
        log_probs_buf = []
        values_buf = []
        rewards_buf = []
        dones_buf = []

        while not done:
            action_mask = env.action_mask() if hasattr(env, "action_mask") else None
            action, log_prob, value = agent.act(obs, explore=True, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs)
            actions_buf.append(action)
            log_probs_buf.append(log_prob)
            values_buf.append(value)
            rewards_buf.append(reward)
            dones_buf.append(float(done))

            obs = next_obs
            total_reward += reward
            steps += 1

        advantages, returns = agent.compute_gae(
            np.array(rewards_buf, dtype=np.float32),
            np.array(values_buf, dtype=np.float32),
            np.array(dones_buf, dtype=np.float32),
            last_value=0.0,
        )

        batch = {
            "obs": np.array(obs_buf, dtype=np.float32),
            "actions": np.array(actions_buf, dtype=np.int64),
            "log_probs": np.array(log_probs_buf, dtype=np.float32),
            "returns": returns.astype(np.float32),
            "advantages": advantages.astype(np.float32),
        }
        agent.update(batch, epochs=ppo_epochs, minibatch_size=minibatch_size)

        final_score = info.get("final_score")
        ep_rewards.append(total_reward)
        ep_scores.append(final_score if final_score is not None else np.nan)
        ep_lengths.append(steps)

        if len(ep_scores) >= window:
            moving_scores.append(float(np.mean(ep_scores[-window:])))
            moving_lengths.append(float(np.mean(ep_lengths[-window:])))
        else:
            moving_scores.append(float(np.mean(ep_scores)))
            moving_lengths.append(float(np.mean(ep_lengths)))

        if eval_every > 0 and ep % eval_every == 0:
            avg_score = moving_scores[-1]
            print(
                f"[Episode {ep:5d}] avg_score(last{window})={avg_score:.2f} "
                f"avg_steps(last{window})={moving_lengths[-1]:.2f}"
            )
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                agent.save(str(best_model_path))
                print(f"Saved best model to {best_model_path}")

    agent.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")

    _save_metrics(run_dir, window, ep_scores, ep_rewards, ep_lengths, moving_scores, moving_lengths)


def _train_tabular_q(
    env,
    agent,
    train_cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    episodes = int(train_cfg.get("episodes", 20000))
    epsilon_start = float(train_cfg.get("epsilon_start", 1.0))
    epsilon_end = float(train_cfg.get("epsilon_end", 0.1))
    epsilon_decay_steps = int(train_cfg.get("epsilon_decay_steps", 50000))
    eval_every = int(train_cfg.get("eval_every", 1000))
    window = int(train_cfg.get("plot_window", 50))

    global_step = 0
    best_avg_score = float("inf")

    ep_rewards = []
    ep_scores = []
    ep_lengths = []
    moving_scores = []
    moving_lengths = []

    best_model_path = run_dir / "model_best.pkl"
    final_model_path = run_dir / "model_final.pkl"

    for ep in tqdm(range(1, episodes + 1), desc="Training"):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info = {}

        while not done:
            epsilon = epsilon_by_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)
            action_mask = env.action_mask() if hasattr(env, "action_mask") else None
            action = agent.select_action(obs, epsilon, explore=True, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_action_mask = env.action_mask() if hasattr(env, "action_mask") else None
            agent.update(obs, action, reward, next_obs, done, next_action_mask)

            obs = next_obs
            total_reward += reward
            steps += 1
            global_step += 1

        final_score = info.get("final_score")
        ep_rewards.append(total_reward)
        ep_scores.append(final_score if final_score is not None else np.nan)
        ep_lengths.append(steps)

        if len(ep_scores) >= window:
            moving_scores.append(float(np.mean(ep_scores[-window:])))
            moving_lengths.append(float(np.mean(ep_lengths[-window:])))
        else:
            moving_scores.append(float(np.mean(ep_scores)))
            moving_lengths.append(float(np.mean(ep_lengths)))

        if eval_every > 0 and ep % eval_every == 0:
            avg_score = moving_scores[-1]
            print(
                f"[Episode {ep:5d}] avg_score(last{window})={avg_score:.2f} "
                f"avg_steps(last{window})={moving_lengths[-1]:.2f}"
            )
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                agent.save(str(best_model_path))
                print(f"Saved best model to {best_model_path}")

    agent.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")

    _save_metrics(run_dir, window, ep_scores, ep_rewards, ep_lengths, moving_scores, moving_lengths)


def train_from_config(cfg: Dict[str, Any], run_dir: Path) -> None:
    env_cfg = cfg.get("env", {})
    agent_cfg = cfg.get("agent", {})
    train_cfg = cfg.get("train", {})

    seed = cfg.get("experiment", {}).get("seed")
    if seed is not None:
        set_seed(seed)

    env = make_env(env_cfg)
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    n_actions = env.action_space.n

    agent = make_agent(agent_cfg, obs_dim, n_actions)
    agent_id = agent_cfg.get("id", "dqn_dueling")

    if agent_id == "ppo":
        _train_ppo(env, agent, train_cfg, run_dir)
    elif agent_id == "dqn_dueling":
        _train_dqn(env, agent, train_cfg, run_dir)
    elif agent_id == "tabular_q":
        _train_tabular_q(env, agent, train_cfg, run_dir)
    else:
        raise ValueError(f"Training not supported for agent id: {agent_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument("--runs_dir", default="runs", help="Base directory for outputs")
    parser.add_argument("--name", default=None, help="Override experiment name")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    exp_name = args.name or cfg.get("experiment", {}).get("name") or config_path.stem
    run_dir = make_run_dir(Path(args.runs_dir), exp_name)
    save_config_snapshot(config_path, run_dir)

    train_from_config(cfg, run_dir)


if __name__ == "__main__":
    main()
