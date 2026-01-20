import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

import tomllib

from zero_to_skyjo.registry import make_agent, make_env
from zero_to_skyjo.utils import set_seed


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def run_demo(cfg: Dict[str, Any], model_path: Optional[Path], seed: int, delay: float) -> None:
    env_cfg = cfg.get("env", {})
    agent_cfg = cfg.get("agent", {})

    if seed is not None:
        set_seed(seed)

    env = make_env(env_cfg)
    obs, _ = env.reset(seed=seed)
    obs_dim = obs.shape[0]
    n_actions = env.action_space.n

    agent = make_agent(agent_cfg, obs_dim, n_actions)
    requires_model = getattr(agent, "requires_model", False)
    if requires_model and model_path is None:
        raise ValueError("Model path required for this agent.")
    if model_path is not None and hasattr(agent, "load"):
        agent.load(str(model_path))

    done = False
    step_idx = 0
    info = {}

    while not done:
        print("\n--- STEP", step_idx, "---")
        env.render()
        action_mask = env.action_mask() if hasattr(env, "action_mask") else None
        action = agent.select_action(obs, epsilon=0.0, explore=False, action_mask=action_mask)
        offer_choice = action // env.hand_size
        slot = action % env.hand_size
        print(
            f"Agent action: take {'VISIBLE' if offer_choice == 0 else 'HIDDEN'} -> replace slot {slot}"
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_idx += 1

        if delay > 0:
            time.sleep(delay)

    print("\n=== EPISODE END ===")
    env.render()
    if "final_score" in info:
        print(f"Final score: {info['final_score']}  (exchanges={info['exchanges']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument("--model", default=None, help="Path to a model .pt (optional for baselines)")
    parser.add_argument("--seed", type=int, default=999, help="Demo seed")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between steps")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model_path = Path(args.model) if args.model else None
    run_demo(cfg, model_path, args.seed, args.delay)


if __name__ == "__main__":
    main()
