"""The startup loading screen (rendering + its done/quit exit conditions)."""

from __future__ import annotations

import pygame

from disco_diffusion_studio.ui.loading import LoadingState, loading_screen


def test_loading_screen_returns_true_when_done(app):
    state = LoadingState(status="CLIP RN50", done=True)
    assert loading_screen(app.screen, state) is True


def test_loading_screen_returns_false_on_quit(app):
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    state = LoadingState(status="diffusion model")
    assert loading_screen(app.screen, state) is False
