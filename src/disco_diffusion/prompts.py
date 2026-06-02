"""Prompt parsing helpers.

Ported from the original Disco Diffusion notebook (Katherine Crowson et al.).
"""

from __future__ import annotations


def parse_prompt(prompt: str) -> tuple[str, float]:
    """Split a prompt into its text and trailing ``:weight`` (default ``1.0``).

    Handles ``http(s)://`` paths whose scheme also contains a colon.
    """
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])
