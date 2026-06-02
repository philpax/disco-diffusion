# Credits & Attribution

Disco Diffusion is a community creation built on the work of many people. This
typed/`uv` port (v6) is purely a modernization and repackaging — **all of the
underlying ideas, models, and code belong to the original authors below.** Please
keep this attribution intact.

## Notebook provenance

Original notebook by **Katherine Crowson** (https://github.com/crowsonkb,
https://twitter.com/RiversHaveWings). It uses either OpenAI's 256x256 unconditional
ImageNet or Katherine Crowson's fine-tuned 512x512 diffusion model
(https://github.com/openai/guided-diffusion), together with CLIP
(https://github.com/openai/CLIP) to connect text prompts with images.

Modified by **Daniel Russell** (https://github.com/russelldc) to include optimal
params for quick generations in 15-100 timesteps rather than 1000, plus more robust
augmentations.

Further improvements from **Dango233** and **nshepperd** helped improve diffusion
quality, especially for short runs.

**Vark** added code to load in multiple CLIP models at once.

Zoom, pan, rotation, and keyframe features were taken from **Chigozie Nri**'s VQGAN
Zoom Notebook (https://github.com/chigozienri).

Advanced DangoCutn Cutout method by **Dango233**.

**Somnai** (https://twitter.com/Somnai_dreams) added 2D Diffusion animation
techniques, QoL improvements and various implementations of tech and techniques.

3D animation implementation added by **Adam Letts** (https://twitter.com/gandamu_ml)
in collaboration with Somnai. (3D mode is removed in this port.)

Turbo feature by **Chris Allen** (https://twitter.com/zippy731).

Improvements to ability to run on local systems, Windows support, and dependency
installation by **HostsServer** (https://twitter.com/HostsServer).

VR Mode by **Tom Mason** (https://twitter.com/nin_artificial).

Horizontal and Vertical symmetry functionality by **nshepperd**. Symmetry
transformation_steps by **huemin** (https://twitter.com/huemin_art). Symmetry
integration into Disco Diffusion by **Dmitrii Tochilkin** (https://twitter.com/cut_pow).

Warp and custom model support by **Alex Spirin** (https://twitter.com/devdef).

Pixel Art Diffusion, Watercolor Diffusion, and Pulp SciFi Diffusion models from
**KaliYuga** (https://twitter.com/KaliYuga_ai).

Integration of OpenCLIP models and initiation of integration of KaliYuga models by
**Palmweaver / Chris Scalf** (https://twitter.com/ChrisScalf11).

Integrated portrait_generator_v001 from **Felipe3DArtist**
(https://twitter.com/Felipe3DArtist).

MiDaS version fix by **Steffen Moelter**.

The secondary diffusion model originates in **Katherine Crowson**'s
`v-diffusion-pytorch` (https://github.com/crowsonkb/v-diffusion-pytorch).

## Vendored components

The following research code is vendored under `src/disco_diffusion/vendor/`, kept
close to upstream, each retaining its original `LICENSE`:

| Component | Upstream | Author(s) | License |
| --- | --- | --- | --- |
| `guided_diffusion` | https://github.com/kostarion/guided-diffusion (fork of https://github.com/openai/guided-diffusion) | OpenAI; Disco Diffusion contributors | MIT |
| `clip` | https://github.com/openai/CLIP | OpenAI | MIT |
| `resize_right` | https://github.com/assafshocher/ResizeRight | Assaf Shocher | MIT |
| `lpips` | https://github.com/richzhang/PerceptualSimilarity | Richard Zhang et al. | BSD-2-Clause |

Model weights are downloaded at runtime from their respective hosts and are subject
to their own licenses/terms.

## This port

The original project is MIT-licensed (© 2021 Katherine Crowson); see `LICENSE`.
This port retains that license.
