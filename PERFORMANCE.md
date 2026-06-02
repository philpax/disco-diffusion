# Performance

The default 1280×768 / 250-step run on an RTX 5090 went from **~124 s to ~59 s — a 2.1×
speedup — with no loss of fidelity** (the output stays within the run-to-run noise floor;
see [Faithfulness](#faithfulness)). This document records each optimization and its measured
impact.

All numbers below are the **warm sampling-loop time** (the 240 denoising steps that actually
run after `--skip-steps`), measured at a fixed seed (1234) with the default configuration on
one RTX 5090, torch 2.11.0+cu128. Per-step times are the mean over steady-state steps
(warmup / first-run compile excluded). Every milestone was measured against the *same*
`uv.lock`, so the deltas reflect code changes, not dependency changes.

## The faithful optimization path

Each row is a commit; each is faithful (within the noise floor). Cumulative speedup is vs.
the post-port eager baseline.

| # | Optimization | ms/step | sampling | incremental | cumulative |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Post-port baseline (eager) | 516 | 124 s | — | 1.00× |
| B | `torch.compile` + batched guidance + TF32 | 328 | 79 s | 1.57× | 1.57× |
| C | Cached resize matrix | 255 | 61 s | 1.29× | 2.02× |
| D | UNet `max-autotune` | 245 | 59 s | 1.04× | **2.11×** |

A single denoising step is ~45% primary UNet and ~55% CLIP guidance:

```
per warm step (HEAD, ~245 ms):
  UNet forward (compiled) .......... ~109 ms   convolution-bound fp16 GEMMs
  guidance (cond_fn) ............... ~135 ms
    ├─ CLIP + secondary backward ... ~70 ms    inherent (3 CLIP models + secondary)
    ├─ cutout generation ........... ~38 ms    resample + augmentations
    ├─ CLIP forward (×3) ........... ~26 ms
    └─ secondary forward ........... ~17 ms
```

### B. `torch.compile` + batched guidance + TF32 — 1.57×

The biggest single win. Three changes that landed together:

- **`torch.compile` on the UNet and CLIP image encoders.** The UNet forward drops from
  258 ms (eager) to ~128 ms — Inductor fuses the GroupNorm/SiLU/residual elementwise ops
  into the convolution epilogues, which eager runs as dozens of separate memory-bound
  kernels. Compiled kernels are cached on disk (`models/.inductor_cache`), so the one-time
  warmup (~1 min) is paid only on the first run with a given configuration.
- **Batched CLIP guidance.** The `cutn_batches` cutout sets were encoded and
  back-propagated one batch at a time; collapsing them into a single batched CLIP encode +
  one `autograd.grad` per model is numerically identical (the per-batch gradient mean
  reduces to one mean over all cutouts) but far fewer launches.
- **TF32 matmuls** (`set_float32_matmul_precision("high")`) and **disabled gradient
  checkpointing** (the guidance gradient flows through the *secondary* model, not the
  primary UNet, so checkpointing the UNet only added recompute and broke compile graphs).

### C. Cached resize matrix — 1.29×

After compile, the cutout machinery became the bottleneck: `resize_right` (the lanczos/cubic
resampler) is called ~144×/step (≈12 inner crops × 4 `cutn_batches` × 3 CLIP models) and is
Python-heavy, recomputing its resampling weights on every call.

The insight: **`resize_right` with a fixed interpolation method is a constant linear operator
on pixel values** — for a given (in_size, out_size) it equals a fixed matrix `M`. So extract
`M` once (by resizing an identity basis), cache it, and apply it as a matmul. This reproduces
`resize_right` to float rounding (~3e-7) — bit-identical for our purposes — while being ~10×
faster per resize. See `cutouts.py` (`_resize_matrix` / `_resample_to`).

### D. UNet `max-autotune` — 1.04×

The compiled UNet is convolution-bound (~50% of its runtime is fp16 tensor-core GEMMs).
Compiling with `mode="max-autotune-no-cudagraphs"` lets Inductor benchmark Triton/CUTLASS
conv+matmul templates and pick the fastest, taking the UNet forward from ~128 ms to ~109 ms
(~14%). It's lossless (still fp16 tensor cores) and the slower first-run autotune is cached
on disk. CUDA graphs are skipped because the forward is GPU-bound, not launch-bound (they
gave no measurable speedup). End-to-end this is a smaller ~4% because guidance is the other
half of the step.

## Faithfulness

Disco Diffusion is **not** bit-reproducible, even with a fixed seed and the original code —
the CLIP-guidance backward uses non-deterministic GPU reductions, and the chaos compounds
over ~240 steps, so two identical-seed runs differ by ~17–21 dB PSNR. "Faithful" therefore
means *within that noise floor*.

Measured at seed 1234: the fully-optimized HEAD image differs from the eager baseline image
by **16.5 dB** — essentially equal to the ~17 dB two-runs-of-the-same-code floor. In other
words, an optimized image differs from an eager one by no more than two eager runs differ
from each other. Composition, style and quality are preserved.

## Why ~59 s is the faithful floor

The per-step budget is now genuinely hard to reduce without changing the result. The
following were all implemented and measured, and **rejected** — each either gave no speedup
or moved the output below the noise floor:

| Idea | Result | Why |
| --- | --- | --- |
| `channels_last` UNet | slower (120→135 ms compiled) | this NCHW model fuses better under Inductor; the 1D-conv attention reshapes force conversions |
| SDPA / FlashAttention | 0.5–0.7× (slower) | attention is only ~1% of the compiled UNet; the batched einsum wins at these shapes |
| CUDA graphs (`reduce-overhead`) | ~0 | the UNet forward is GPU-bound, not launch-bound |
| `torch.compile` the secondary model | ~0 | too small to benefit |
| `max-autotune` on CLIP encoders | ~0 | the 224² ViT/RN50 are too small (unlike the big UNet) |
| fp8 convs / CLIP | n/a / too lossy | fp8 convs unavailable; fp8 CLIP departs too far |
| Batched inner-crop resampling | bit-faithful, but 0 end-to-end | cutouts run in fp32; densifying the operators adds FLOPs that offset the saved autograd nodes |
| Stream overlap (UNet ∥ guidance) | 0 speedup + 11 dB | real `cond_fn` serializes via `.item()` syncs; concurrency also reorders the nondeterministic reductions below the floor |
| Shared cutouts across CLIP models | 1.4× guidance, but 9.8 dB | reusing identical cutouts across all 3 models collapses gradient diversity (≈ halving `cutn-batches`) |

The remaining cost is real, faithful compute: convolution FLOPs in the UNet and the CLIP
backward in the guidance. Neither has a faithful lever left.

## Opt-in speed levers (lossy)

Getting *under* the faithful floor requires a measured fidelity tradeoff. All are **off by
default** — the default stays faithful.

| Flag | Effect | Cost |
| --- | --- | --- |
| `--fast-fp16-secondary` | secondary guidance model in fp16 | mild ~3 dB departure (borderline-faithful) |
| `--fast` | enables all `--fast-*` levers above | |
| `--cutn-batches 2` | fewer CLIP guidance samples | ~11 dB — a different (still good) sample |
| `--guidance-every N` | recompute CLIP guidance every N steps | see below |

`--guidance-every N` reuses the CLIP guidance gradient between recomputes (it drifts slowly
step-to-step). Unlike the precision levers, it doesn't *degrade* the image — it produces a
**different but equally coherent sample**, like a seed change:

| N | speedup | sampling | PSNR vs faithful | note |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.00× | ~59 s | — | faithful (default) |
| 2 | 1.46× | ~44 s | 9.4 dB | different composition, same quality |
| 3 | 1.60× | ~40 s | 8.2 dB | slight style drift |
| 4 | 1.73× | ~37 s | 7.4 dB | more saturated / posterized |

It is deliberately **not** bundled into `--fast` (it's a more visible departure than the
~3 dB fp16-secondary), and the faithful default (`N=1`) is a verified no-op.

## Reproducing these measurements

Each milestone was measured by checking out the commit and timing every `ddim_sample` call
(GPU-synchronized), taking the mean of the steady-state steps (first 10 skipped to exclude
the first-run compile), at the default config with `seed=1234`. The `uv.lock` is identical
across all milestones, so the environment (torch 2.11.0+cu128) is held constant and the
deltas are purely from code.
