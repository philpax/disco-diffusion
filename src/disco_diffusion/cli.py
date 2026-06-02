"""Command-line interface for Disco Diffusion.

Example::

    disco-diffusion generate --prompt "a lighthouse" --steps 100 --width 768 --height 768

With no arguments it reproduces the canonical lighthouse image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from .config import DiffusionModel, RunConfig, SamplingMode
from .generate import generate as run_generate

app = typer.Typer(add_completion=False, help="Disco Diffusion: CLIP-guided diffusion art.")


@app.callback()
def _main() -> None:
    """Disco Diffusion: CLIP-guided diffusion art."""


@app.command()
def generate(
    prompt: Annotated[
        list[str] | None,
        typer.Option("--prompt", "-p", help="Text prompt (repeatable). Supports ':weight' suffix."),
    ] = None,
    steps: Annotated[int, typer.Option(help="Number of diffusion steps.")] = 250,
    width: Annotated[int, typer.Option(help="Output width (snapped to /64).")] = 1280,
    height: Annotated[int, typer.Option(help="Output height (snapped to /64).")] = 768,
    seed: Annotated[int | None, typer.Option(help="Random seed (default: random).")] = None,
    n_batches: Annotated[int, typer.Option(help="Number of images to generate.")] = 1,
    clip_guidance_scale: Annotated[int, typer.Option(help="CLIP guidance strength.")] = 5000,
    init_image: Annotated[str | None, typer.Option(help="Optional init image path/URL.")] = None,
    skip_steps: Annotated[int, typer.Option(help="Steps to skip (use ~50% with init).")] = 10,
    diffusion_model: Annotated[
        DiffusionModel, typer.Option(help="Primary diffusion checkpoint.")
    ] = DiffusionModel.finetune_512,
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="Diffusion sampler.")
    ] = SamplingMode.ddim,
    secondary_model: Annotated[
        bool, typer.Option(help="Use the secondary model for guidance.")
    ] = True,
    clip_model: Annotated[
        list[str] | None,
        typer.Option("--clip-model", help="CLIP model to use (repeatable)."),
    ] = None,
    output_dir: Annotated[Path, typer.Option(help="Output directory.")] = Path("images_out"),
    models_dir: Annotated[Path, typer.Option(help="Model cache directory.")] = Path("models"),
    batch_name: Annotated[str, typer.Option(help="Batch name (output subfolder).")] = "TimeToDisco",
    cpu: Annotated[bool, typer.Option(help="Force CPU (very slow).")] = False,
    compile: Annotated[
        bool,
        typer.Option(
            help="torch.compile the UNet/CLIP (~2x faster once warm; first run compiles)."
        ),
    ] = True,
    fast: Annotated[
        bool,
        typer.Option("--fast", help="Enable all fast (slightly lossy) levers (~58s)."),
    ] = False,
    fast_fp16_secondary: Annotated[
        bool,
        typer.Option(help="fp16 secondary model (~3s faster; mild ~3dB departure)."),
    ] = False,
) -> None:
    """Generate an image from a text prompt."""
    overrides: dict[str, Any] = {
        "steps": steps,
        "width": width,
        "height": height,
        "seed": seed,
        "n_batches": n_batches,
        "clip_guidance_scale": clip_guidance_scale,
        "init_image": init_image,
        "skip_steps": skip_steps,
        "diffusion_model": diffusion_model,
        "diffusion_sampling_mode": sampling_mode,
        "use_secondary_model": secondary_model,
        "output_dir": output_dir,
        "models_dir": models_dir,
        "batch_name": batch_name,
        "cpu": cpu,
        "compile": compile,
        # --fast turns on all the individual fast levers.
        "fast_fp16_secondary": fast or fast_fp16_secondary,
    }
    if prompt:
        overrides["prompts"] = prompt
    if clip_model:
        overrides["clip_models"] = clip_model

    config = RunConfig(**overrides)
    typer.echo(f"Config: {config.model_dump_json(indent=2)}")
    paths = run_generate(config)
    typer.echo("\nGenerated:")
    for path in paths:
        typer.echo(f"  {path}")


if __name__ == "__main__":
    app()
