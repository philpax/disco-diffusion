# Performance

On an RTX 5090, the default 1280×768 / 250-step run takes about 59 seconds once warm, down from ~124 seconds right after the port: a 2.1× speedup with no visible change to the output, which stays within the run-to-run noise floor (see [Faithfulness](#faithfulness)). On an RTX 3090 the same run is about 4m27s once warm, roughly 4.5× slower.

Times are for the warm sampling loop, meaning the 240 denoising steps that run after `--skip-steps`, measured at seed 1234 with the default configuration on one RTX 5090 (torch 2.11.0+cu128). Per-step figures are the mean over steady-state steps, excluding warmup and the first-run compile. All milestones share the same `uv.lock`, so the differences come from code, not dependencies.

## The optimization path

Each row is a commit, all faithful (within the noise floor); cumulative is relative to the eager baseline (A).

| # | Optimization | ms/step | sampling | incremental | cumulative |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Post-port baseline (eager) | 516 | 124 s | — | 1.00× |
| B | `torch.compile` + batched guidance + TF32 | 328 | 79 s | 1.57× | 1.57× |
| C | Cached resize matrix | 255 | 61 s | 1.29× | 2.02× |
| D | UNet `max-autotune` | 245 | 59 s | 1.04× | 2.11× |

A single denoising step is roughly 45% primary UNet and 55% CLIP guidance:

```
per warm step (HEAD, ~245 ms):
  UNet forward (compiled) .......... ~109 ms   convolution-bound fp16 GEMMs
  guidance (cond_fn) ............... ~135 ms
    ├─ CLIP + secondary backward ... ~70 ms    inherent (3 CLIP models + secondary)
    ├─ cutout generation ........... ~38 ms    resample + augmentations
    ├─ CLIP forward (×3) ........... ~26 ms
    └─ secondary forward ........... ~17 ms
```

### torch.compile, batched guidance, TF32

`torch.compile` on the UNet and CLIP image encoders drops the UNet forward from 258 ms eager to about 128 ms, as Inductor fuses GroupNorm, SiLU and residual adds into the convolution epilogues that eager mode runs as dozens of separate memory-bound kernels; the compiled kernels are cached on disk under `models/.inductor_cache`, so the ~1 minute warmup is paid once per configuration. 

Batched CLIP guidance replaces the per-batch encode and backward over the `cutn_batches` cutout sets with one batched encode and one `autograd.grad` per model, which is numerically identical (the per-batch gradient mean is a mean over all cutouts) but launches far fewer kernels.

TF32 matmuls (`set_float32_matmul_precision("high")`) and disabling gradient checkpointing finish it off; the guidance gradient runs through the secondary model rather than the primary UNet, so checkpointing the UNet only added recompute and broke the compile graph.

### Cached resize matrix

With the UNet compiled, the cutout code became the bottleneck. `resize_right`, the lanczos/cubic resampler, is called around 144 times per step (roughly 12 inner crops × 4 `cutn_batches` × 3 CLIP models) and rebuilds its resampling weights on every call. But with a fixed interpolation method it is a constant linear operator on the pixels: for a given input and output size it is the same matrix every time. Building that matrix once by resizing an identity basis, caching it, and applying it as a matmul reproduces `resize_right` to about 3e-7 (bit-identical for our purposes) and runs roughly 10× faster per resize. See `_resize_matrix` and `_resample_to` in `cutouts.py`.

### UNet max-autotune

The compiled UNet spends about half its time in fp16 tensor-core convolutions. Compiling with `mode="max-autotune-no-cudagraphs"` benchmarks Triton and CUTLASS templates for those convs and matmuls and keeps the fastest, taking the forward from ~128 ms to ~109 ms. It is lossless (still fp16 tensor cores), the longer first-run autotune is cached, and CUDA graphs are left off because the forward is GPU-bound rather than launch-bound. End-to-end this is about 4%, since the UNet is just under half the step.

## Faithfulness

Disco Diffusion is not bit-reproducible, even with a fixed seed and the original code: the CLIP-guidance backward uses non-deterministic GPU reductions, and that noise compounds over ~240 steps, so two runs from the same seed land 17–21 dB apart in PSNR. "Faithful" means staying within that floor. At seed 1234 the optimized HEAD image is 16.5 dB from the eager baseline, about the same distance as two runs of identical code, so composition, style and quality are preserved.

## Why ~59 s is the floor

The remaining cost is convolutions in the UNet and the CLIP backward in the guidance, neither of which has a faithful lever left. The following were implemented and measured, then dropped, each because it either did not help or pushed the output below the noise floor:

| Idea | Result | Why |
| --- | --- | --- |
| `channels_last` UNet | slower (120→135 ms compiled) | this NCHW model fuses better under Inductor; the 1D-conv attention reshapes force conversions |
| SDPA / FlashAttention | 0.5–0.7× (slower) | attention is only ~1% of the compiled UNet, and the batched einsum wins at these shapes |
| CUDA graphs (`reduce-overhead`) | no change | the UNet forward is GPU-bound, not launch-bound |
| `torch.compile` the secondary model | no change | too small to benefit |
| `max-autotune` on the CLIP encoders | no change | the 224² ViT/RN50 are too small, unlike the big UNet |
| fp8 convs / CLIP | unavailable / too lossy | fp8 convs aren't available, and fp8 CLIP departs too far |
| Batched inner-crop resampling | bit-faithful, but no net gain | cutouts run in fp32, so densifying the operators adds FLOPs that offset the saved autograd nodes |
| Stream overlap (UNet ∥ guidance) | no gain, and 11 dB | the real `cond_fn` serializes on `.item()` syncs, and concurrency reorders the nondeterministic reductions below the floor |
| Shared cutouts across CLIP models | 1.4× on guidance, but 9.8 dB | reusing identical cutouts across all three models collapses gradient diversity (about the same as halving `cutn-batches`) |

## Opt-in speed levers (lossy)

Going below the floor requires giving up some fidelity, so these are off by default.

| Flag | Effect | Cost |
| --- | --- | --- |
| `--fast-fp16-secondary` | secondary guidance model in fp16 | mild ~3 dB departure (borderline-faithful) |
| `--fast` | enables all the `--fast-*` levers above | |
| `--cutn-batches 2` | fewer CLIP guidance samples | ~11 dB, a different but still good sample |
| `--guidance-every N` | recompute CLIP guidance every N steps | see below |

`--guidance-every N` reuses the CLIP guidance gradient between recomputes, which drifts slowly from step to step. Rather than degrading the image, it changes it: the result is a different but equally coherent sample, closer to a different seed.

| N | speedup | sampling | PSNR vs faithful | note |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.00× | ~59 s | — | faithful (default) |
| 2 | 1.46× | ~44 s | 9.4 dB | different composition, same quality |
| 3 | 1.60× | ~40 s | 8.2 dB | slight style drift |
| 4 | 1.73× | ~37 s | 7.4 dB | more saturated / posterized |

It is not bundled into `--fast` (a more visible change than the fp16 secondary), and `N=1` is a verified no-op.

## Reproducing the numbers

Each milestone was measured by checking out the commit and timing every `ddim_sample` call (GPU-synchronized), averaging the steady-state steps and dropping the first ten to skip the first-run compile, at the default config with seed 1234.
