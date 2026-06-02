"""Tests for RunConfig validation and the cut-schedule parser."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from disco_diffusion.config import RunConfig, parse_schedule


def test_defaults_reproduce_lighthouse() -> None:
    config = RunConfig()
    assert config.steps == 250
    assert (config.width, config.height) == (1280, 768)
    assert config.clip_models == ["ViT-B/32", "ViT-B/16", "RN50"]
    assert "lighthouse" in config.prompts[0]


def test_side_snapping_to_multiple_of_64() -> None:
    config = RunConfig(width=1000, height=700)
    assert config.side_x == 960  # 1000 // 64 * 64
    assert config.side_y == 640  # 700 // 64 * 64


def test_parse_schedule_repeated_and_single() -> None:
    assert parse_schedule("[12]*400+[4]*600") == [12.0] * 400 + [4.0] * 600
    assert parse_schedule("[1]*1000") == [1.0] * 1000
    assert len(parse_schedule("[0.2]*400+[0]*600")) == 1000


def test_parse_schedule_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_schedule("12 * 400")


def test_all_default_schedules_are_1000_long() -> None:
    config = RunConfig()
    for schedule in (
        config.cut_overview_schedule(),
        config.cut_innercut_schedule(),
        config.cut_ic_pow_schedule(),
        config.cut_icgray_p_schedule(),
    ):
        assert len(schedule) == 1000


def test_unknown_clip_model_rejected() -> None:
    with pytest.raises(ValidationError):
        RunConfig(clip_models=["NotARealModel"])


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        RunConfig(nonexistent_field=1)
