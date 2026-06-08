"""Native OS file dialogs via ``crossfiledialog``.

``crossfiledialog`` shells out to the platform's native picker — zenity/kdialog on Linux, the
Win32 common dialog on Windows, ``osascript`` on macOS. Its import raises if no Linux backend is
found (e.g. zenity isn't installed), so we import lazily and re-raise a clear :class:`Unavailable`
that callers surface to the user. Calls block until the user picks a path or cancels.
"""

from __future__ import annotations


class Unavailable(RuntimeError):
    """No native file-dialog backend is available (on Linux, install zenity or kdialog)."""


def _cfd():
    try:
        import crossfiledialog
    except Exception as exc:  # noqa: BLE001 - NoImplementationFoundException, ImportError, …
        raise Unavailable(str(exc) or "no native dialog backend") from exc
    return crossfiledialog


def is_available() -> bool:
    """Whether a native dialog backend is present (cheap to call)."""
    try:
        _cfd()
    except Unavailable:
        return False
    return True


def save_file(title: str = "Save image", start_dir: str | None = None) -> str | None:
    """Open a native Save dialog; return the chosen path, or ``None`` if cancelled."""
    return _cfd().save_file(title=title, start_dir=start_dir) or None


def open_file(title: str = "Open image", start_dir: str | None = None) -> str | None:
    """Open a native Open dialog; return the chosen path, or ``None`` if cancelled."""
    return _cfd().open_file(title=title, start_dir=start_dir) or None
