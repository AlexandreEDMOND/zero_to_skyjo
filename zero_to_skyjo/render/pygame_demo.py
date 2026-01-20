import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import tomllib

from zero_to_skyjo.registry import make_agent, make_env
from zero_to_skyjo.utils import set_seed


try:
    import pygame
except ImportError as exc:
    raise SystemExit("pygame is required. Install it with: uv pip install pygame") from exc


@dataclass
class Layout:
    scale: float = 1.0
    card_w: int = 90
    card_h: int = 120
    margin: int = 20
    gap: int = 15
    width: int = 900
    height: int = 600

    def __post_init__(self) -> None:
        self.card_w = int(self.card_w * self.scale)
        self.card_h = int(self.card_h * self.scale)
        self.margin = int(self.margin * self.scale)
        self.gap = int(self.gap * self.scale)
        self.width = int(self.width * self.scale)
        self.height = int(self.height * self.scale)

    def hand_positions(self) -> Dict[int, pygame.Rect]:
        start_x = self.margin
        start_y = self.margin + self.card_h + self.gap * 3
        positions = {}
        idx = 0
        for row in range(2):
            for col in range(3):
                x = start_x + col * (self.card_w + self.gap)
                y = start_y + row * (self.card_h + self.gap)
                positions[idx] = pygame.Rect(x, y, self.card_w, self.card_h)
                idx += 1
        return positions

    def offer_positions(self) -> Dict[str, pygame.Rect]:
        start_x = self.margin
        start_y = self.margin
        visible = pygame.Rect(start_x, start_y, self.card_w, self.card_h)
        hidden = pygame.Rect(start_x + self.card_w + self.gap, start_y, self.card_w, self.card_h)
        return {"visible": visible, "hidden": hidden}

    def info_origin(self) -> pygame.Rect:
        grid_w = 3 * self.card_w + 2 * self.gap
        x = self.margin + grid_w + self.gap * 2
        y = self.margin
        w = max(self.card_w * 2, self.width - x - self.margin)
        h = self.height - self.margin * 2
        return pygame.Rect(x, y, w, h)


VALUE_COLORS = {
    1: (120, 200, 120),
    2: (150, 210, 120),
    3: (200, 220, 120),
    4: (230, 190, 120),
    5: (230, 150, 120),
    6: (220, 100, 100),
}

BACKGROUND = (245, 242, 235)
TEXT = (30, 30, 30)
HIDDEN = (60, 60, 60)
BORDER = (20, 20, 20)
HIGHLIGHT = (240, 200, 70)
INACTIVE = (190, 190, 190)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def draw_text(surface, font, text: str, x: int, y: int, color=TEXT) -> None:
    label = font.render(text, True, color)
    surface.blit(label, (x, y))


def draw_card(surface, font, rect: pygame.Rect, value: Optional[int], visible: bool, highlight: bool) -> None:
    if visible and value is not None:
        color = VALUE_COLORS.get(value, (200, 200, 200))
        text_color = TEXT
        label = str(value)
    else:
        color = HIDDEN
        text_color = (230, 230, 230)
        label = "?"

    pygame.draw.rect(surface, color, rect, border_radius=8)
    pygame.draw.rect(surface, BORDER, rect, width=2, border_radius=8)

    if highlight:
        highlight_rect = rect.inflate(8, 8)
        pygame.draw.rect(surface, HIGHLIGHT, highlight_rect, width=3, border_radius=10)

    text = font.render(label, True, text_color)
    text_rect = text.get_rect(center=rect.center)
    surface.blit(text, text_rect)


def draw_inactive_card(surface, font, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, INACTIVE, rect, border_radius=8)
    pygame.draw.rect(surface, BORDER, rect, width=2, border_radius=8)
    text = font.render("-", True, TEXT)
    text_rect = text.get_rect(center=rect.center)
    surface.blit(text, text_rect)


def draw_scene(
    surface,
    fonts,
    layout: Layout,
    env,
    action: Optional[int],
    step_idx: int,
    status: str,
    final_score: Optional[int],
) -> None:
    surface.fill(BACKGROUND)
    font, font_small = fonts

    offer_rects = layout.offer_positions()
    hand_rects = layout.hand_positions()
    info_rect = layout.info_origin()

    offer_choice = None
    slot_choice = None
    if action is not None:
        offer_choice = action // env.hand_size
        slot_choice = action % env.hand_size

    if hasattr(env, "offer"):
        draw_text(
            surface, font_small, "Offer visible", offer_rects["visible"].x, offer_rects["visible"].y - 20
        )
        draw_text(
            surface, font_small, "Offer hidden", offer_rects["hidden"].x, offer_rects["hidden"].y - 20
        )

        if env.offer is not None:
            draw_card(
                surface,
                font,
                offer_rects["visible"],
                env.offer.visible_value,
                True,
                highlight=offer_choice == 0,
            )
            draw_card(
                surface,
                font,
                offer_rects["hidden"],
                env.offer.hidden_value,
                False,
                highlight=offer_choice == 1,
            )
        else:
            draw_card(surface, font, offer_rects["visible"], None, False, False)
            draw_card(surface, font, offer_rects["hidden"], None, False, False)
    else:
        draw_text(surface, font_small, "Discard", offer_rects["visible"].x, offer_rects["visible"].y - 20)
        draw_text(surface, font_small, "Deck", offer_rects["hidden"].x, offer_rects["hidden"].y - 20)
        discard_value = env.discard[-1] if env.discard else None
        draw_card(
            surface,
            font,
            offer_rects["visible"],
            discard_value,
            discard_value is not None,
            highlight=offer_choice == 0,
        )
        draw_card(
            surface,
            font,
            offer_rects["hidden"],
            None,
            False,
            highlight=offer_choice in (1, 2),
        )

    draw_text(surface, font_small, "Hand", layout.margin, hand_rects[0].y - 20)
    for idx, rect in hand_rects.items():
        value = env.player_values[idx]
        visible = env.player_visible[idx]
        if hasattr(env, "player_active") and not env.player_active[idx]:
            draw_inactive_card(surface, font, rect)
        else:
            draw_card(surface, font, rect, value, visible, highlight=slot_choice == idx)

    info_x = info_rect.x
    info_y = info_rect.y
    draw_text(surface, font, "Info", info_x, info_y)
    draw_text(surface, font_small, f"Step: {step_idx}", info_x, info_y + 35)
    draw_text(surface, font_small, f"Deck size: {len(env.deck)}", info_x, info_y + 60)
    if hasattr(env, "exchanges_count"):
        draw_text(surface, font_small, f"Exchanges: {env.exchanges_count}", info_x, info_y + 85)
    else:
        draw_text(
            surface,
            font_small,
            f"Actions: {env.action_count}/{env.max_actions}",
            info_x,
            info_y + 85,
        )

    deck_rect = pygame.Rect(info_x, info_y + 120, layout.card_w, layout.card_h)
    discard_rect = pygame.Rect(info_x + layout.card_w + layout.gap, info_y + 120, layout.card_w, layout.card_h)

    draw_text(surface, font_small, "Deck", deck_rect.x, deck_rect.y - 20)
    draw_card(surface, font, deck_rect, None, False, False)
    draw_text(surface, font_small, str(len(env.deck)), deck_rect.x + 8, deck_rect.y + 8)

    draw_text(surface, font_small, "Discard", discard_rect.x, discard_rect.y - 20)
    if env.discard:
        top = env.discard[-1]
        draw_card(surface, font, discard_rect, top, True, False)
    else:
        draw_card(surface, font, discard_rect, None, False, False)

    draw_text(surface, font_small, f"Action: {status}", info_x, info_y + 120 + layout.card_h + 20)

    if final_score is not None:
        draw_text(surface, font, f"Final score: {final_score}", info_x, info_y + 120 + layout.card_h + 55)


def wait_with_events(clock, delay_s: float) -> bool:
    if delay_s <= 0:
        return True

    end_time = time.perf_counter() + delay_s
    while time.perf_counter() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        clock.tick(60)
    return True


def run_pygame_demo(
    cfg: Dict[str, Any],
    model_path: Optional[Path],
    seed: int,
    delay: float,
    fps: int,
    scale: float,
) -> None:
    if seed is not None:
        set_seed(seed)

    env = make_env(cfg.get("env", {}))
    obs, _ = env.reset(seed=seed)
    obs_dim = obs.shape[0]
    n_actions = env.action_space.n

    agent = make_agent(cfg.get("agent", {}), obs_dim, n_actions)
    requires_model = getattr(agent, "requires_model", False)
    if requires_model and model_path is None:
        raise ValueError("Model path required for this agent.")
    if model_path is not None and hasattr(agent, "load"):
        agent.load(str(model_path))

    layout = Layout(scale=scale)

    pygame.init()
    screen = pygame.display.set_mode((layout.width, layout.height))
    pygame.display.set_caption(f"{env.__class__.__name__} - Agent Demo")
    font = pygame.font.Font(None, int(28 * scale))
    font_small = pygame.font.Font(None, int(20 * scale))
    clock = pygame.time.Clock()

    running = True
    done = False
    step_idx = 0
    info = {}

    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        action_mask = env.action_mask() if hasattr(env, "action_mask") else None
        action = agent.select_action(obs, epsilon=0.0, explore=False, action_mask=action_mask)
        action_type = action // env.hand_size
        slot = action % env.hand_size
        if hasattr(env, "offer"):
            status = f"{'VISIBLE' if action_type == 0 else 'HIDDEN'} -> slot {slot}"
        else:
            if action_type == 0:
                status = f"DISCARD -> slot {slot}"
            elif action_type == 1:
                status = f"DRAW keep -> slot {slot}"
            else:
                status = f"DRAW discard -> reveal slot {slot}"

        draw_scene(screen, (font, font_small), layout, env, action, step_idx, status, None)
        pygame.display.flip()

        if not wait_with_events(clock, delay):
            running = False
            break

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_idx += 1
        clock.tick(max(1, fps))

    if running:
        final_score = info.get("final_score") if done else None
        draw_scene(screen, (font, font_small), layout, env, None, step_idx, "END", final_score)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                    waiting = False
            clock.tick(30)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument("--model", default=None, help="Path to a model .pt (optional for baselines)")
    parser.add_argument("--seed", type=int, default=None, help="Demo seed")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between steps")
    parser.add_argument("--fps", type=int, default=None, help="Target fps")
    parser.add_argument("--scale", type=float, default=None, help="Window scale")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    demo_cfg = cfg.get("demo_ui", {})

    seed = int(demo_cfg.get("seed", 999)) if args.seed is None else args.seed
    delay = float(demo_cfg.get("delay", 0.2)) if args.delay is None else args.delay
    fps = int(demo_cfg.get("fps", 60)) if args.fps is None else args.fps
    scale = float(demo_cfg.get("window_scale", 1.0)) if args.scale is None else args.scale

    model_path = Path(args.model) if args.model else None
    run_pygame_demo(cfg, model_path, seed=seed, delay=delay, fps=fps, scale=scale)


if __name__ == "__main__":
    main()
