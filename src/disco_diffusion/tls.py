"""Make urllib-based downloads use certifi's CA bundle.

Several dependencies (``torch.hub`` for the LPIPS VGG backbone, OpenAI CLIP, ...)
download weights with :mod:`urllib`, which uses the system trust store. In some
environments (e.g. behind a private-CA proxy) that store rejects the chain even
though ``requests``/``certifi`` accept it. Pointing urllib at certifi makes all of
these behave consistently.
"""

from __future__ import annotations

import ssl

_configured = False


def ensure_certifi_ssl() -> None:
    """Install a certifi-backed default HTTPS context for urllib (idempotent)."""
    global _configured
    if _configured:
        return
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        ssl._create_default_https_context = lambda *a, **k: context
    except Exception:  # pragma: no cover - best effort; fall back to system default
        pass
    _configured = True
