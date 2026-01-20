import numpy as np

from zero_to_skyjo.envs.solo_skyjo import SoloSkyjoEnv


def _random_valid_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise RuntimeError("No valid actions available")
    return int(rng.choice(valid))


def test_random_runs(n_runs: int = 100, seed: int = 123) -> None:
    rng = np.random.default_rng(seed)
    scores = []
    steps = []
    truncated = 0

    for _ in range(n_runs):
        env = SoloSkyjoEnv(seed=int(rng.integers(0, 1_000_000_000)))
        obs, _ = env.reset()
        done = False

        while not done:
            action = _random_valid_action(env.action_mask(), rng)
            obs, reward, terminated, tr, info = env.step(action)
            done = terminated or tr
            if done:
                scores.append(info.get("final_score", np.nan))
                steps.append(info.get("action_count", np.nan))
                if tr:
                    truncated += 1

    print("Random runs")
    print(f"  runs: {n_runs}")
    print(f"  avg_score: {float(np.mean(scores)):.2f}")
    print(f"  avg_steps: {float(np.mean(steps)):.2f}")
    print(f"  truncated: {truncated}/{n_runs}")


def test_fixed_slot_max_actions(seed: int = 456) -> None:
    env = SoloSkyjoEnv(seed=seed)
    obs, _ = env.reset()

    slot = 0
    action = env.hand_size + slot  # draw keep -> replace

    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Fixed slot max actions")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")
    print(f"  action_count: {info.get('action_count')}")


def test_reveal_all_in_six(seed: int = 789) -> None:
    env = SoloSkyjoEnv(seed=seed)
    obs, _ = env.reset()

    done = False
    for slot in range(env.hand_size):
        if not env.player_active[slot]:
            continue
        if env.player_visible[slot]:
            continue
        action = 2 * env.hand_size + slot  # draw discard -> reveal
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            break

    print("Reveal all in <= 6")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")
    print(f"  action_count: {info.get('action_count')}")
    print(f"  remaining_cards: {info.get('remaining_cards')}")


def test_force_clear_all_columns(seed: int = 321) -> None:
    env = SoloSkyjoEnv(seed=seed)
    obs, _ = env.reset()

    env.player_values = [1, 2, 3, 1, 2, 3]
    env.player_visible = [True] * env.hand_size
    env.player_active = [True] * env.hand_size
    env._clear_matching_columns()

    active_count = env._active_count()
    print("Force clear all columns")
    print(f"  active_count_after_clear: {active_count}")

    obs, reward, terminated, truncated, info = env.step(0)
    print(f"  terminated_after_step: {terminated}")
    print(f"  final_score: {info.get('final_score')}")


def main() -> None:
    test_random_runs()
    print("-")
    test_fixed_slot_max_actions()
    print("-")
    test_reveal_all_in_six()
    print("-")
    test_force_clear_all_columns()


if __name__ == "__main__":
    main()
