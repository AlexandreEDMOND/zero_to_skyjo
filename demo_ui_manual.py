import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import tomllib

try:
    import pygame
except ImportError as exc:
    raise SystemExit("pygame is required. Install it with: uv pip install pygame") from exc

from zero_to_skyjo.registry import make_env
from zero_to_skyjo.utils import set_seed
from zero_to_skyjo.render import pygame_demo as render


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def action_label(env, action: int) -> str:
    action_type = action // env.hand_size
    slot = action % env.hand_size
    if hasattr(env, "offer"):
        return f"{'VISIBLE' if action_type == 0 else 'HIDDEN'} -> slot {slot}"
    if action_type == 0:
        return f"DISCARD -> slot {slot}"
    if action_type == 1:
        return f"DRAW keep -> slot {slot}"
    return f"DRAW discard -> reveal slot {slot}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reset")
    parser.add_argument("--scale", type=float, default=None, help="Window scale")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    demo_cfg = cfg.get("demo_ui", {})

    seed = int(demo_cfg.get("seed", 999)) if args.seed is None else args.seed
    scale = float(demo_cfg.get("window_scale", 1.0)) if args.scale is None else args.scale

    if seed is not None:
        set_seed(seed)

    env = make_env(cfg.get("env", {}))
    obs, _ = env.reset(seed=seed)

    layout = render.Layout(scale=scale)

    pygame.init()
    screen = pygame.display.set_mode((layout.width, layout.height))
    pygame.display.set_caption(f"{env.__class__.__name__} - Manual Demo")
    font = pygame.font.Font(None, int(28 * scale))
    font_small = pygame.font.Font(None, int(20 * scale))
    clock = pygame.time.Clock()

    running = True
    done = False
    info: Dict[str, object] = {}
    status = "WAIT"
    input_str = ""
    final_score: Optional[int] = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset(seed=seed)
                    done = False
                    info = {}
                    status = "RESET"
                    input_str = ""
                    final_score = None
                elif event.key == pygame.K_BACKSPACE:
                    input_str = input_str[:-1]
                elif event.key == pygame.K_RETURN:
                    if not done and input_str:
                        action = int(input_str)
                        input_str = ""
                        mask = env.action_mask() if hasattr(env, "action_mask") else None
                        valid = True
                        if action < 0 or action >= env.action_space.n:
                            valid = False
                        if mask is not None and valid and not mask[action]:
                            valid = False
                        if not valid:
                            status = "INVALID ACTION"
                        else:
                            obs, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                            status = action_label(env, action)
                            if done:
                                final_score = info.get("final_score")
                    elif done:
                        status = "DONE"
                else:
                    if event.unicode.isdigit() and len(input_str) < 2:
                        input_str += event.unicode

        render.draw_scene(
            screen,
            (font, font_small),
            layout,
            env,
            None,
            int(getattr(env, "action_count", 0)),
            status,
            final_score,
        )

        info_rect = layout.info_origin()
        base_x = info_rect.x
        base_y = info_rect.y + 260

        render.draw_text(screen, font_small, "Manual controls", base_x, base_y)
        render.draw_text(
            screen,
            font_small,
            "Type action number (0-17), Enter to play",
            base_x,
            base_y + 22,
        )
        render.draw_text(screen, font_small, "Backspace: edit | R: reset | Esc: quit", base_x, base_y + 44)
        render.draw_text(screen, font_small, "0-5: discard -> replace slot", base_x, base_y + 70)
        render.draw_text(screen, font_small, "6-11: draw keep -> replace slot", base_x, base_y + 92)
        render.draw_text(screen, font_small, "12-17: draw discard -> reveal slot", base_x, base_y + 114)
        render.draw_text(screen, font_small, f"Input: {input_str}", base_x, base_y + 140)
        if done:
            render.draw_text(screen, font_small, "Episode done - press R", base_x, base_y + 162)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
