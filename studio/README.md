# Disco Diffusion Studio

An interactive desktop UI for [Disco Diffusion](../README.md), built on its external-control
API (`disco_diffusion.DiscoSession`). It takes manual control of the sampling loop so you can
**steer the image as it forms** — mixing, crossfading, and swapping prompts live, between steps.

Full-quality steps are slow on purpose: each one is a window to retune the prompt mix and watch
the image respond.

![the studio UI](docs/screenshot.png)

## What you can do

- **Play / Pause / Stop** the diffusion loop; a step counter shows progress.
- **Prompts**: add/remove rows, each with a live **weight slider (0–2)**. Text applies on
  Enter *or* when you click away; an amber `edited · Enter` badge shows when a box hasn't been
  applied yet. Each row shows the **normalised %** it actually contributes to guidance.
- **Steps**: set the total step count (while paused/stopped).
- **Size**: width/height (snapped to multiples of 64) and a landscape/portrait flip. The image
  is letterboxed into the window, so changing it never resizes or re-tiles the window.
- **Save** the current frame — opens a file dialog (the frame is frozen when you click, so it
  won't change while you pick a location); a `.png` extension is added if you omit one.

Generation runs on a background thread, so the UI stays responsive while the GPU works.

## Running

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) member of the
Disco Diffusion repo, so run it **from the repo root** — it shares the library's environment and
model weights (and pulls PyTorch from the CUDA 12.8 wheel index, same as the library):

```sh
# from the repo root:
uv run disco-studio
```

Options: `disco-studio --help` (`--steps`, `--width`, `--height`, `--compile`, `--cpu`,
`--models-dir`, `--out`). Paths default to the repo-root `models/` and `images_out/`.

## Layout

```
src/disco_diffusion_studio/
  app.py      # the App: window, widgets, event loop, run lifecycle
  worker.py   # GenerationWorker: the background thread driving a Sampler
  layout.py   # sizing tokens + a tiny flow layout (Row / Stack)
  theme.py    # palette + pygame_gui theme
```

## Development

From the repo root (the workspace runs ruff/format/mypy over this member):

```sh
uv run --directory studio ruff check .
uv run --directory studio ruff format --check .
uv run --directory studio mypy
```
