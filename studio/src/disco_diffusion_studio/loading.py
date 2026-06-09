"""The startup loading screen, shown while the models load on a background thread.

Loading the weights takes ~a minute, so ``main`` opens the window at its final size, kicks the
load off on a daemon thread, and pumps :func:`loading_screen` until it finishes (or the user
closes the window). The App then adopts that same window, so there's no resize once loading ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame
from disco_diffusion import DiscoSession

from .constants import APP_TITLE
from .theme import WINDOW_BG


@dataclass
class LoadingState:
    """Shared between the loading thread and the loading screen."""

    status: str = "starting"  # the component currently loading
    session: DiscoSession | None = None
    error: Exception | None = None
    done: bool = False


def loading_screen(screen: pygame.Surface, state: LoadingState) -> bool:
    """Render a loading screen until the model load finishes; ``False`` if the user closes it."""
    title_font = pygame.font.SysFont(None, 40)
    status_font = pygame.font.SysFont(None, 24)
    hint_font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        w, h = screen.get_size()
        cx, cy = w // 2, h // 2
        screen.fill(WINDOW_BG)
        title = title_font.render(APP_TITLE, True, (231, 233, 240))
        screen.blit(title, title.get_rect(center=(cx, cy - 38)))
        if state.error is not None:
            msg = status_font.render(f"Failed to load: {state.error}", True, (239, 107, 129))
            screen.blit(msg, msg.get_rect(center=(cx, cy + 8)))
            hint = hint_font.render("close the window to exit", True, (122, 130, 144))
            screen.blit(hint, hint.get_rect(center=(cx, cy + 42)))
        else:
            dots = "." * (1 + (pygame.time.get_ticks() // 400) % 3)
            msg = status_font.render(f"loading {state.status}{dots}", True, (150, 196, 255))
            screen.blit(msg, msg.get_rect(center=(cx, cy + 8)))
            # An indeterminate progress sweep so it's clearly alive during the long load.
            bar = pygame.Rect(0, 0, min(420, w - 80), 4)
            bar.center = (cx, cy + 46)
            pygame.draw.rect(screen, (38, 44, 56), bar, border_radius=2)
            frac = (pygame.time.get_ticks() % 1400) / 1400.0
            seg_w = bar.width // 4
            sx = max(bar.left, bar.left + int((bar.width + seg_w) * frac) - seg_w)
            seg_w = min(seg_w, bar.right - sx)
            if seg_w > 0:
                rect = (sx, bar.top, seg_w, bar.height)
                pygame.draw.rect(screen, (109, 124, 255), rect, border_radius=2)
        pygame.display.flip()
        clock.tick(30)
        if state.done and state.error is None:
            return True
