"""Model checkpoint download + verification.

Only the *weights* are fetched at runtime (and cached under ``models_dir``); all
*code* lives in this repo. Ported from the original Disco Diffusion notebook's
``diff_model_map`` / ``download_model``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# name -> (sha256, [mirror URIs])
MODEL_MAP: dict[str, tuple[str, list[str]]] = {
    "256x256_diffusion_uncond": (
        "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1",
        [
            "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
            "https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt",
        ],
    ),
    "512x512_diffusion_uncond_finetune_008100": (
        "9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648",
        [
            "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt",
            "https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt",
        ],
    ),
    "portrait_generator_v001": (
        "b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9",
        [
            "https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt",
        ],
    ),
    "secondary": (
        "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a",
        [
            "https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth",
            "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth",
        ],
    ),
}


def model_filename(name: str) -> str:
    """The on-disk filename for a model (basename of its first URI)."""
    uris = MODEL_MAP[name][1]
    return Path(urlparse(uris[0]).path).name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(uri: str, dest: Path) -> None:
    with requests.get(uri, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)


def ensure_model(name: str, models_dir: Path, verify: bool = False) -> Path:
    """Ensure model ``name`` is present in ``models_dir``; download if missing.

    Returns the local path. If ``verify`` is set, re-checks the sha256 of an
    existing file and re-downloads on mismatch.
    """
    sha, uris = MODEL_MAP[name]
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / model_filename(name)

    if dest.exists():
        if not verify or _sha256(dest) == sha:
            return dest
        print(f"{name}: sha256 mismatch, re-downloading.")

    last_error: Exception | None = None
    for uri in uris:
        try:
            print(f"Downloading {name} from {uri}")
            _download(uri, dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            print(f"  failed: {exc}")
    raise RuntimeError(f"Could not download {name} from any mirror") from last_error
