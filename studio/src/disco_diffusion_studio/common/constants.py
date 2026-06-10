"""App-wide tuning constants, shared by the App and the UI modules (``_ui_*.py``).

Kept out of ``app`` so the UI builders / event router / renderer can import them without a cycle
back through ``app``. (Layout sizes live in ``layout``; colours in ``theme``.)
"""

from __future__ import annotations

APP_TITLE = "Disco Diffusion Studio"  # window caption + loading-screen heading

BRUSH_SIZE_MIN, BRUSH_SIZE_MAX = 4.0, 160.0  # brush radius bounds (slider + scroll/[])
BRUSH_STRENGTH_MIN, BRUSH_STRENGTH_MAX = 0.05, 1.0  # brush opacity bounds (slider + shift-scroll)

STEP_MIN, STEP_MAX = 1, 1000  # supported total-steps range

# Help HUD text per interaction mode (hold right mouse = navigate, release = draw).
DRAW_HELP = (
    "left-drag: paint · scroll/[]: size · shift+scroll: opacity · 1-9: colour"
    " · hold right: pan/zoom · F: fit · space: play · ctrl+S/Z: save/undo"
)
NAV_HELP = "right-drag: pan · scroll: zoom · release right: draw · F: fit · space: play/pause"

CANVAS_EMPTY_BG = (18, 20, 26)  # placeholder canvas fill shown before the first frame
CANVAS_BORDER = (70, 78, 92)

# Debounce for the model auto-reload: changing the CLIP set / secondary toggle queues a reload
# that fires this long after the *last* change, so rapid toggling doesn't reload repeatedly.
RELOAD_DEBOUNCE_MS = 1200
# Debounce for the guidance-change history checkpoint: a guidance slider streams events while
# dragged, so we checkpoint once the value has been quiescent for this long (one settled change
# = one revertible entry), rather than once per pixel of the drag.
GUIDANCE_CHECKPOINT_MS = 500
