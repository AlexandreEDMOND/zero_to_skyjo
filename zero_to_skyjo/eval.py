import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import tomllib
from tqdm import tqdm

from zero_to_skyjo.registry import make_agent, make_env
from zero_to_skyjo.utils import set_seed


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def evaluate_from_config(
    cfg: Dict[str, Any], model_path: Optional[Path], episodes: int
) -> Dict[str, float]:
    env_cfg = cfg.get("env", {})
    agent_cfg = cfg.get("agent", {})

    seed = cfg.get("experiment", {}).get("seed")
    if seed is not None:
        set_seed(seed)

    env = make_env(env_cfg)
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    n_actions = env.action_space.n

    agent = make_agent(agent_cfg, obs_dim, n_actions)
    requires_model = getattr(agent, "requires_model", False)
    if requires_model and model_path is None:
        raise ValueError("Model path required for this agent.")
    if model_path is not None and hasattr(agent, "load"):
        agent.load(str(model_path))

    scores = []
    steps_list = []

    for _ in tqdm(range(episodes), desc="Eval"):
        obs, _ = env.reset()
        done = False
        steps = 0
        info = {}

        while not done:
            action_mask = env.action_mask() if hasattr(env, "action_mask") else None
            action = agent.select_action(obs, epsilon=0.0, explore=False, action_mask=action_mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        scores.append(info.get("final_score", np.nan))
        steps_list.append(steps)

    avg_score = float(np.mean(scores))
    avg_steps = float(np.mean(steps_list))

    print(f"Evaluation over {episodes} episodes:")
    print(f"  Average score: {avg_score:.2f}")
    print(f"  Average steps: {avg_steps:.2f}")

    return {"avg_score": avg_score, "avg_steps": avg_steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument("--model", default=None, help="Path to a model .pt (optional for baselines)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of eval episodes")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model_path = Path(args.model) if args.model else None
    evaluate_from_config(cfg, model_path, args.episodes)


if __name__ == "__main__":
    main()
