import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from zero_to_skyjo.demo import run_demo
from zero_to_skyjo.eval import evaluate_from_config
from zero_to_skyjo.train import load_config, make_run_dir, save_config_snapshot, train_from_config


MODES = {"train", "eval", "demo"}


def resolve_mode(cfg: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    return cfg.get("experiment", {}).get("mode", "train")


def resolve_model_path(cfg: Dict[str, Any], override: Optional[str]) -> Optional[Path]:
    if override:
        return Path(override)

    exp_cfg = cfg.get("experiment", {})
    if "model_path" in exp_cfg:
        return Path(exp_cfg["model_path"])

    for section in ("eval", "demo"):
        section_cfg = cfg.get(section, {})
        if "model_path" in section_cfg:
            return Path(section_cfg["model_path"])

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument("--mode", choices=sorted(MODES), default=None, help="Override mode")
    parser.add_argument("--model", default=None, help="Path to a model .pt")
    parser.add_argument("--runs_dir", default="runs", help="Base directory for outputs")
    parser.add_argument("--name", default=None, help="Override experiment name")
    parser.add_argument("--episodes", type=int, default=None, help="Override train episodes")
    parser.add_argument("--eval_episodes", type=int, default=None, help="Override eval episodes")
    parser.add_argument("--delay", type=float, default=None, help="Demo delay in seconds")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    mode = resolve_mode(cfg, args.mode)

    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "train":
        if args.episodes is not None:
            cfg.setdefault("train", {})["episodes"] = args.episodes

        exp_name = args.name or cfg.get("experiment", {}).get("name") or Path(args.config).stem
        run_dir = make_run_dir(Path(args.runs_dir), exp_name)
        save_config_snapshot(Path(args.config), run_dir)
        train_from_config(cfg, run_dir)
        return

    model_path = resolve_model_path(cfg, args.model)

    if mode == "eval":
        eval_eps = cfg.get("eval", {}).get("episodes", 1000)
        if args.eval_episodes is not None:
            eval_eps = args.eval_episodes
        evaluate_from_config(cfg, model_path, int(eval_eps))
        return

    if mode == "demo":
        demo_cfg = cfg.get("demo", {})
        seed = int(demo_cfg.get("seed", 999))
        delay = demo_cfg.get("delay", 0.0)
        if args.delay is not None:
            delay = args.delay
        run_demo(cfg, model_path, seed=seed, delay=float(delay))
        return


if __name__ == "__main__":
    main()
