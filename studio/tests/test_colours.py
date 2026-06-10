"""Colour config: palette loading, recents persistence, and hex<->rgb conversion."""

from __future__ import annotations

from disco_diffusion_studio.paint import colours as P


def test_load_colours_has_palette_and_empty_recents():
    colours = P.load_colours()
    assert len(colours.palette) >= 1
    assert colours.recent == []


def test_recents_round_trip():
    colours = P.load_colours()
    updated = P.ColourConfig(palette=colours.palette, recent=[(10, 20, 30), (255, 0, 128)])
    P.save_colours(updated)
    assert P.load_colours().recent == [(10, 20, 30), (255, 0, 128)]


def test_hex_rgb_round_trip():
    assert P._hex_to_rgb(P._rgb_to_hex((1, 2, 3))) == (1, 2, 3)
    assert P._hex_to_rgb("#ffffff") == (255, 255, 255)
    assert P._rgb_to_hex((0, 0, 0)) == "#000000"


def test_load_colours_falls_back_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "CONFIG_PATH", tmp_path / "missing.toml")
    colours = P.load_colours()
    assert len(colours.palette) == len(P.DEFAULT_PALETTE)
