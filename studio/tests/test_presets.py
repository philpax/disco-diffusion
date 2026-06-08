"""Presets: bundled loading, the Default-matches-RunConfig invariant, and save round-trips."""

from __future__ import annotations

from disco_diffusion import RunConfig

from disco_diffusion_studio import presets as P


def test_bundled_presets_load():
    presets = P.load_presets()
    assert {"Default", "2022 sauce"} <= set(presets)
    assert list(presets)[0] == "Default"  # Default always sorts first


def test_default_preset_matches_runconfig_defaults():
    # The studio opens on "Default", so the preset must equal a fresh RunConfig's live values.
    default = P.load_presets()["Default"]
    cfg = RunConfig()
    assert default.config.clip_guidance_scale == cfg.clip_guidance_scale
    assert default.config.cut_overview == cfg.cut_overview
    assert set(default.clip_models) == set(cfg.clip_models)
    assert default.use_secondary_model == cfg.use_secondary_model


def test_save_preset_round_trips():
    src = P.load_presets()["2022 sauce"]
    name, path = P.save_preset("my test recipe", src)
    assert path.exists()
    reloaded = P.load_presets()
    assert name in reloaded
    assert reloaded[name].config.model_dump() == src.config.model_dump()
    assert set(reloaded[name].clip_models) == set(src.clip_models)


def test_save_preset_sanitizes_filename():
    name, path = P.save_preset("../sneaky/../name!!.toml", P.load_presets()["Default"])
    assert path.parent == P.PRESETS_DIR  # stays inside the presets dir
    assert path.suffix == ".toml"


def test_preset_config_fields_are_runconfig_fields():
    # Mirrors the import-time guard in presets.py: PresetConfig must not drift from RunConfig.
    assert set(P.PresetConfig.model_fields) <= set(RunConfig.model_fields)
