# Disco Diffusion (typed `uv` port)

A durable, strongly-typed port of the semi-famous 2022 [Disco Diffusion]
CLIP-guided diffusion art generator. The original was a Google Colab notebook that
bootstrapped itself with runtime `pip install` / `git clone` and pinned a long-dead
ML stack. This version is:

- a proper Python **library + CLI**, managed with [`uv`](https://docs.astral.sh/uv/);
- **strongly typed** (`mypy --strict`), linted and formatted (`ruff`);
- running on a **modern PyTorch stack** with CUDA 12.8 wheels, so it uses recent
  NVIDIA GPUs (developed on an RTX 5090; works on Ampere such as the 3090 too);
- **self-contained**: every fragile 2022 research repo it depended on is vendored
  in-tree under `src/disco_diffusion/vendor/`, so it keeps working even if those
  upstream repositories disappear. Only model *weights* are downloaded at runtime.

The 3D, video, turbo, VR and 2D-animation modes from the original have been removed;
this port focuses on generating still images.

## Requirements

- `uv`
- An NVIDIA GPU with recent drivers (CUDA 12.8-capable). CPU works but is extremely
  slow. On NixOS, the provided `shell.nix` wires up CUDA/Triton.

## Setup

```sh
uv sync
```

This installs PyTorch from the CUDA 12.8 wheel index along with the rest of the
dependencies. Verify the GPU is visible:

```sh
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

## Usage

Generate the canonical lighthouse image (faithful defaults — 1280×768, 250 steps,
ViT-B/32 + ViT-B/16 + RN50, secondary model). This downloads ~2.5 GB of model weights
into `models/` on first run:

```sh
uv run disco-diffusion generate
```

A quick, low-fidelity smoke test:

```sh
uv run disco-diffusion generate --steps 25 --width 512 --height 512 \
    --prompt "a beautiful painting of a lighthouse, trending on artstation"
```

Useful options (`disco-diffusion generate --help` for the full list):

| Option | Default | Meaning |
| --- | --- | --- |
| `--prompt, -p` | lighthouse | Text prompt (repeatable; supports `text:weight`) |
| `--steps` | 250 | Diffusion steps |
| `--width` / `--height` | 1280 / 768 | Output size (snapped down to a multiple of 64) |
| `--seed` | random | Reproducibility seed |
| `--n-batches` | 1 | Number of images to generate |
| `--clip-guidance-scale` | 5000 | Strength of CLIP guidance |
| `--init-image` | none | Init image path/URL (set `--skip-steps` to ~50% of steps) |
| `--diffusion-model` | `512x512_diffusion_uncond_finetune_008100` | Primary checkpoint |
| `--sampling-mode` | `ddim` | `ddim` or `plms` |
| `--clip-model` | (the three above) | CLIP model (repeatable) |
| `--cpu` | off | Force CPU |

Images and a JSON settings dump are written to `images_out/<batch_name>/`.

### Library API

```python
from disco_diffusion.config import RunConfig
from disco_diffusion.generate import generate

paths = generate(RunConfig(prompts=["a serene mountain lake at dawn"], steps=100))
print(paths)
```

## Development

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Credits & license

Disco Diffusion is the work of many people — see [`CREDITS.md`](CREDITS.md) for the
full provenance and the licensing of each vendored component. The project is
MIT-licensed (© 2021 Katherine Crowson); see [`LICENSE`](LICENSE). This port retains
that license and all original attribution.

[Disco Diffusion]: https://github.com/alembics/disco-diffusion
