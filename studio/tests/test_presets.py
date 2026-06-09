"""Presets: bundled loading, the Default-matches-RunConfig invariant, and save round-trips."""

from __future__ import annotations

import zipfile

import pytest
from disco_diffusion import RunConfig
from PIL import Image
from pydantic import ValidationError

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


def _session(**overrides):
    default = P.load_presets()["Default"]
    base = dict(
        width=1280, height=768, steps=120, seed=42, denoise=60,
        prompts=[("a vast landscape", 1.0, False), ("yellow", 0.5, True)],
        config=default.config, clip_models=default.clip_models,
        use_secondary_model=default.use_secondary_model,
    )
    return P.Session(**{**base, **overrides})


def test_session_round_trips(tmp_path):
    sess = _session()
    P.save_session(str(tmp_path / "work.zip"), sess)
    loaded, image, history = P.load_session(str(tmp_path / "work.zip"))
    assert loaded == sess  # incl. the inline-table prompts
    assert image is None and history == []  # no result / history bundled


def test_session_bundles_and_restores_result_image(tmp_path):
    img = Image.new("RGB", (16, 12), (10, 200, 50))
    out = P.save_session(str(tmp_path / "s.zip"), _session(), img)
    _loaded, restored, _history = P.load_session(str(out))
    assert restored is not None and restored.size == (16, 12)


def test_session_bundles_and_restores_history(tmp_path):
    item = P.HistoryItem(
        label="paint soft", step=3, index=8, total=20,
        prompts=[("a", 1.0, False)], config=P.GuidanceSnapshot(clip_guidance_scale=7000),
    )
    history = [(item, Image.new("RGB", (16, 12), (200, 100, 50)))]
    out = P.save_session(str(tmp_path / "s.zip"), _session(), None, history)
    _loaded, _image, restored = P.load_session(str(out))
    assert len(restored) == 1
    meta, preview = restored[0]
    assert meta == item  # metadata round-trips (validated JSON)
    assert preview.size == (16, 12)


def test_save_session_defaults_zip_suffix(tmp_path):
    out = P.save_session(str(tmp_path / "noext"), _session(prompts=[]))
    assert out.suffix == ".zip"


def test_load_session_validates_malformed_toml(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        # missing the required [config] and [models] tables
        zf.writestr("session.toml", "[output]\nwidth = 64\nheight = 64\nsteps = 10\nseed = 1\n")
    with pytest.raises(ValidationError):
        P.load_session(str(bad))
