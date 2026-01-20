from typing import Any, Dict

from zero_to_skyjo.agents.baselines import GreedyMinAgent, GreedyThresholdAgent, RandomAgent
from zero_to_skyjo.agents.dqn import DQNAgent
from zero_to_skyjo.agents.ppo import PPOAgent
from zero_to_skyjo.agents.tabular_q import TabularQAgent
from zero_to_skyjo.envs.mini_skyjo import MiniSkyjoEnv
from zero_to_skyjo.envs.solo_skyjo import SoloSkyjoEnv


ENV_REGISTRY = {
    "mini_skyjo": MiniSkyjoEnv,
    "solo_skyjo": SoloSkyjoEnv,
}

AGENT_REGISTRY = {
    "dqn_dueling": DQNAgent,
    "ppo": PPOAgent,
    "tabular_q": TabularQAgent,
    "random": RandomAgent,
    "greedy_min": GreedyMinAgent,
    "greedy_threshold": GreedyThresholdAgent,
}


def make_env(cfg: Dict[str, Any]):
    env_id = cfg.get("id")
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"Unknown env id: {env_id}")

    if env_id == "mini_skyjo":
        return MiniSkyjoEnv(
            reshuffle_returned=cfg.get("reshuffle_returned", True),
            seed=cfg.get("seed"),
        )
    if env_id == "solo_skyjo":
        return SoloSkyjoEnv(
            seed=cfg.get("seed"),
            max_actions=cfg.get("max_actions", 40),
            action_penalty=cfg.get("action_penalty", 1),
            invalid_action_penalty=cfg.get("invalid_action_penalty", 2),
        )

    raise ValueError(f"Env id not handled: {env_id}")


def make_agent(cfg: Dict[str, Any], obs_dim: int, n_actions: int):
    agent_id = cfg.get("id")
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent id: {agent_id}")

    if agent_id == "dqn_dueling":
        return DQNAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden=cfg.get("hidden", 512),
            lr=cfg.get("lr", 3e-4),
            gamma=cfg.get("gamma", 0.995),
            target_update=cfg.get("target_update", 500),
            weight_decay=cfg.get("weight_decay", 1e-5),
            grad_clip=cfg.get("grad_clip", 1.0),
            device=cfg.get("device", "cpu"),
            action_noise_std=cfg.get("action_noise_std", 0.01),
            action_noise_epsilon=cfg.get("action_noise_epsilon", 0.2),
        )
    if agent_id == "ppo":
        return PPOAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden=cfg.get("hidden", 256),
            lr=cfg.get("lr", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            clip_eps=cfg.get("clip_eps", 0.2),
            ent_coef=cfg.get("ent_coef", 0.01),
            vf_coef=cfg.get("vf_coef", 0.5),
            max_grad_norm=cfg.get("max_grad_norm", 0.5),
            device=cfg.get("device", "cpu"),
        )
    if agent_id == "tabular_q":
        return TabularQAgent(
            n_actions=n_actions,
            hand_size=cfg.get("hand_size", 6),
            min_value=cfg.get("min_value", -2),
            max_value=cfg.get("max_value", 12),
            alpha=cfg.get("alpha", 0.1),
            gamma=cfg.get("gamma", 0.99),
            seed=cfg.get("seed"),
        )
    if agent_id == "random":
        return RandomAgent(n_actions=n_actions, seed=cfg.get("seed"))
    if agent_id == "greedy_min":
        return GreedyMinAgent(
            n_actions=n_actions,
            hand_size=cfg.get("hand_size", 6),
            min_value=cfg.get("min_value", -2),
            max_value=cfg.get("max_value", 12),
            seed=cfg.get("seed"),
        )
    if agent_id == "greedy_threshold":
        return GreedyThresholdAgent(
            n_actions=n_actions,
            hand_size=cfg.get("hand_size", 6),
            min_value=cfg.get("min_value", -2),
            max_value=cfg.get("max_value", 12),
            threshold=cfg.get("threshold", 5),
            seed=cfg.get("seed"),
        )

    raise ValueError(f"Agent id not handled: {agent_id}")
