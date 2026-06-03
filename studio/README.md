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
- **Size**: width/height (snapped to multiples of 64) and a landscape/portrait flip. The
  canvas (at the chosen size) is shown before you generate, so you can see the aspect — and
  paint on it — before pressing Play.
- **Canvas navigation**: the window is a viewport onto the canvas. **Hold the right mouse
  button** to navigate — drag to pan, scroll to zoom toward the cursor; release to go back to
  drawing. **F** fits the canvas to the window, **0** is 100%. A help line in the canvas
  corner shows the bindings for the current mode.
- **Paint** directly onto the canvas to steer the diffusion (drawing mode). Left-drag with a
  brush (Soft / Hard / Spray); **scroll** changes brush size and **Shift+scroll** changes
  opacity; a colour palette and a brush-preview ring follow the cursor. Strokes are noised to
  the current step and injected into the live latent, so the painted region pulls the image
  toward your colours/shapes and then evolves with the prompts. Strokes show as an overlay
  until a step bakes them in; **Clear** discards unbaked strokes.
  - Toggle **Noise** to deposit *fresh tinted noise* instead of plain colour: the region is
    replaced with new noise at the current level, biased to your colour, so the model invents
    **new structure** there. This is what you want early on — a plain colour stroke is washed
    out by the renoise at high noise levels, whereas tinted noise survives and gets resolved
    into shapes of that colour.
- **Save** the current frame — opens a file dialog (the frame is frozen when you click, so it
  won't change while you pick a location); a `.png` extension is added if you omit one.

Generation runs on a background thread, so the UI stays responsive while the GPU works.

![painting onto the canvas to steer the diffusion](docs/painting.png)

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
  paint.py    # brushes, colour palette, and the paintable RGBA layer
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
