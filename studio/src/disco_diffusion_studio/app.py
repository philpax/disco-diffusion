"""Disco Diffusion Studio — interactive control of the sampling loop.

Take manual control of the diffusion loop and steer it live by mixing encoded prompts at
every step. The image is shown at the top; the panel below holds the controls:

  * play / pause + stop, a step counter, and a total-steps box (editable while paused/stopped)
  * a list of prompts, each with a live weight slider (0-2) and a remove button; text applies
    on Enter or when focus leaves the box, and each row shows the normalised mix it contributes
  * width / height (snapped to multiples of 64) and a landscape/portrait flip
  * save

Steps are deliberately slow at full quality: each one is an opportunity to retune the prompt
mix and watch the image respond. Generation runs on a background thread so the UI stays
responsive; all torch/CUDA work happens on that one thread (see worker.py).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pygame
import pygame_gui
from disco_diffusion import DiscoSession, EncodedPrompt, RunConfig
from pygame_gui.core import ObjectID
from pygame_gui.windows import UIFileDialog

from .layout import (
    CTRL_H,
    DEFAULT_H,
    DEFAULT_W,
    LABEL_H,
    MARGIN,
    MIN_IMAGE_H,
    MIN_WINDOW_W,
    PAD,
    PANEL_H,
    ROW_PITCH,
    Row,
    Stack,
    snap_side,
)
from .theme import (
    DIVIDER,
    IMAGE_BG,
    MUTED_COLOR,
    PANEL_BG,
    PENDING_COLOR,
    READOUT_COLOR,
    THEME,
    WINDOW_BG,
)
from .worker import GenerationWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("disco_diffusion_studio")


# --- app ---------------------------------------------------------------------


@dataclass
class PromptRow:
    text: str = ""
    weight: float = 1.0


@dataclass
class App:
    session: DiscoSession
    out_dir: Path

    width: int = DEFAULT_W
    height: int = DEFAULT_H
    steps: int = 100
    prompts: list[PromptRow] = field(default_factory=list)

    worker: GenerationWorker | None = None
    paused: bool = False
    _frame_surface: pygame.Surface | None = None
    _frame_key: tuple[int, int] | None = None  # (id(array), index) to detect new frames
    _pending_size: tuple[int, int] | None = None  # coalesced window resize, applied per frame
    _file_dialog: UIFileDialog | None = None  # open Save dialog, if any
    _save_surface: pygame.Surface | None = None  # frame frozen when Save was clicked

    def __post_init__(self) -> None:
        self.width = snap_side(self.width)
        self.height = snap_side(self.height)
        if not self.prompts:
            self.prompts = [PromptRow("a vast alien landscape, oil painting", 1.0)]
        self._encode_cache: dict[str, EncodedPrompt] = {}
        self._cache_lock = threading.Lock()
        # The window size is independent of the generation size: it's seeded from the
        # initial dimensions, then owned by the user (resizable). The image is letterboxed
        # into the image region, so flipping orientation never changes the window.
        self.win_w = max(self.width, MIN_WINDOW_W)
        self.win_h = max(self.height, MIN_IMAGE_H) + PANEL_H
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Disco Diffusion - interactive")
        # Enforce the minimum size natively (SDL) instead of by re-calling set_mode on
        # resize. Re-calling set_mode issues a SetWindowSize that a tiling WM treats as the
        # app demanding geometry, snapping a floated window back under tiling. With a native
        # minimum we just adopt whatever size the window becomes and never fight the WM.
        try:
            window = pygame.Window.from_display_module()
            window.minimum_size = (MIN_WINDOW_W, MIN_IMAGE_H + PANEL_H)
        except (AttributeError, pygame.error):  # older pygame / headless: best-effort only
            pass
        self.manager = pygame_gui.UIManager((self.win_w, self.win_h))
        self.manager.get_theme().load_theme(io.StringIO(json.dumps(THEME)))
        # element-reference tables, rebuilt by _build_ui
        self._row_elements: list[pygame_gui.core.UIElement] = []
        self._remove_buttons: dict[pygame_gui.elements.UIButton, int] = {}
        self._prompt_entries: dict[pygame_gui.elements.UITextEntryLine, int] = {}
        self._weight_sliders: dict[pygame_gui.elements.UIHorizontalSlider, int] = {}
        # The prompt entry that last held keyboard focus, so we can auto-apply its text
        # when focus moves away (no need to press Enter).
        self._focused_entry: pygame_gui.elements.UITextEntryLine | None = None
        self._build_ui()

    # -- geometry --
    def _image_area_h(self) -> int:
        return max(0, self.win_h - PANEL_H)

    def _window_size(self) -> tuple[int, int]:
        return (self.win_w, self.win_h)

    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.is_alive() and not self.paused

    # -- UI construction --
    def _build_ui(self) -> None:
        win_w, win_h = self._window_size()
        self.manager.clear_and_reset()
        self._remove_buttons.clear()
        self._prompt_entries.clear()
        self._weight_sliders.clear()
        self._row_elements.clear()

        ui = pygame_gui.elements
        stack = Stack(MARGIN, self._image_area_h() + PAD, win_w - 2 * MARGIN)

        # Row 1: transport — Play / Stop | step counter | (Save, right-aligned)
        r = stack.row(CTRL_H)
        self.play_button = ui.UIButton(r.left(110), "Play", self.manager, object_id="#play_button")
        self.stop_button = ui.UIButton(r.left(90), "Stop", self.manager, object_id="#stop_button")
        self.save_button = ui.UIButton(r.right(90), "Save", self.manager, object_id="#save_button")
        self.step_label = ui.UILabel(r.fill(), "step 0 / 0", self.manager, object_id="#step_label")

        # Row 2: steps | width / height / apply / orientation swap
        r = stack.row(CTRL_H)
        ui.UILabel(r.left(50), "Steps", self.manager)
        self.steps_entry = ui.UITextEntryLine(r.left(70), self.manager)
        self.steps_entry.set_text(str(self.steps))
        r.left(16)  # spacer
        ui.UILabel(r.left(24), "W", self.manager)
        self.width_entry = ui.UITextEntryLine(r.left(70), self.manager)
        self.width_entry.set_text(str(self.width))
        ui.UILabel(r.left(24), "H", self.manager)
        self.height_entry = ui.UITextEntryLine(r.left(70), self.manager)
        self.height_entry.set_text(str(self.height))
        self.apply_button = ui.UIButton(r.left(96), "Apply size", self.manager)
        self.swap_button = ui.UIButton(r.left(96), "Flip W/H", self.manager)

        # Row 3: status line (fills the width)
        self.status_label = ui.UILabel(
            stack.row(LABEL_H).fill(), "", self.manager, object_id="#status_label"
        )

        # Row 4: prompt list header + add + hint (hint fills the remaining width)
        r = stack.row(LABEL_H)
        ui.UILabel(r.left(90), "PROMPTS", self.manager, object_id="#section_label")
        self.add_button = ui.UIButton(
            r.left(120), "+ Add prompt", self.manager, object_id="#add_button"
        )
        ui.UILabel(
            r.fill(),
            "weight 0-2 applies instantly · text applies on Enter or click-away · % = mix used",
            self.manager,
            object_id="#hint_label",
        )

        # Scrolling prompt list fills the rest of the panel down to the window bottom.
        list_rect = pygame.Rect(MARGIN, stack.y, win_w - 2 * MARGIN, win_h - PAD - stack.y)
        self.prompt_panel = ui.UIScrollingContainer(list_rect, self.manager)
        self._rebuild_prompt_rows()
        self._sync_enabled()

    def _rebuild_prompt_rows(self) -> None:
        for el in self._row_elements:
            el.kill()
        self._row_elements.clear()
        self._remove_buttons.clear()
        self._prompt_entries.clear()
        self._weight_sliders.clear()

        container = self.prompt_panel
        inner_w = int(container.get_container().get_size()[0])
        ui = pygame_gui.elements
        v_pad = (ROW_PITCH - CTRL_H) // 2  # vertically centre widgets in their row pitch
        for i, prompt in enumerate(self.prompts):
            # Pack: [text fills] [slider] [weight readout] [X]. Right-side widgets are taken
            # first so the text entry flexes into whatever width is left.
            r = Row(0, i * ROW_PITCH + v_pad, inner_w, CTRL_H)
            remove = ui.UIButton(
                r.right(30),
                "×",
                self.manager,
                container=container,
                object_id=ObjectID(object_id="#remove_button", class_id="@remove_button"),
            )
            wlabel = ui.UILabel(r.right(104), "", self.manager, container=container)
            slider = ui.UIHorizontalSlider(
                r.right(150),
                start_value=prompt.weight,
                value_range=(0.0, 2.0),
                manager=self.manager,
                container=container,
            )
            entry = ui.UITextEntryLine(r.fill(), self.manager, container=container)
            entry.set_text(prompt.text)
            self._row_elements += [remove, entry, slider, wlabel]
            self._remove_buttons[remove] = i
            self._prompt_entries[entry] = i
            self._weight_sliders[slider] = i
            prompt._wlabel = wlabel  # type: ignore[attr-defined]  # stash for live updates
            prompt._entry = entry  # type: ignore[attr-defined]
            prompt._label_state = None  # type: ignore[attr-defined]  # last (text, colour) shown
        container.set_scrollable_area_dimensions(
            (inner_w, max(len(self.prompts), 1) * ROW_PITCH + 6)
        )
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        """Update each row's readout: raw weight + normalised share, or a pending badge.

        Mirrors Sampler.set_conditioning (empty rows ignored; remaining weights normalised
        to sum to 1, so the % is exactly the mix the guidance uses). A row whose text box
        differs from the applied prompt shows an amber "edited · Enter" badge instead — this
        is the live "not yet applied" signal. Labels are only mutated when their state
        changes, so this is cheap to call every frame.
        """
        active = [(i, r.weight) for i, r in enumerate(self.prompts) if r.text.strip()]
        total = sum(w for _, w in active)
        shares = {i: w / total for i, w in active} if total > 1e-3 else {}
        for i, row in enumerate(self.prompts):
            wlabel = getattr(row, "_wlabel", None)
            entry = getattr(row, "_entry", None)
            if wlabel is None or entry is None:
                continue
            if entry.get_text() != row.text:
                text, colour = "edited · Enter", PENDING_COLOR
            elif not row.text.strip():
                text, colour = f"{row.weight:.2f}  empty", MUTED_COLOR
            elif not shares:
                text, colour = f"{row.weight:.2f}  off", MUTED_COLOR
            else:
                text, colour = f"{row.weight:.2f}  {shares[i] * 100:.0f}%", READOUT_COLOR
            state = (text, colour)
            if row._label_state == state:  # type: ignore[attr-defined]
                continue
            row._label_state = state  # type: ignore[attr-defined]
            wlabel.text_colour = pygame.Color(*colour)
            wlabel.set_text(text)

    def _sync_enabled(self) -> None:
        """Total-steps + size boxes are editable only when not actively generating."""
        editable = not self.running
        for el in (
            self.steps_entry,
            self.width_entry,
            self.height_entry,
            self.apply_button,
            self.swap_button,
        ):
            (el.enable if editable else el.disable)()
        self.play_button.set_text("Pause" if self.running else "Play")

    def _status(self, text: str) -> None:
        self.status_label.set_text(text)

    # -- prompt snapshot --
    def _prompt_snapshot(self) -> list[tuple[str, float]]:
        return [(r.text, r.weight) for r in self.prompts]

    def _push_prompts(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.set_prompts(self._prompt_snapshot())

    def _commit_prompt_entry(self, entry: pygame_gui.elements.UITextEntryLine) -> None:
        """Apply a prompt text box's contents (on Enter or when focus moves away)."""
        idx = self._prompt_entries.get(entry)
        if idx is None or not (0 <= idx < len(self.prompts)):
            return
        if entry.get_text() == self.prompts[idx].text:
            return
        self.prompts[idx].text = entry.get_text()
        self._refresh_rows()
        self._push_prompts()

    # -- run lifecycle --
    def _start_run(self) -> None:
        self._stop_run()
        self.worker = GenerationWorker(
            self.session,
            width=self.width,
            height=self.height,
            steps=self.steps,
            encode_cache=self._encode_cache,
            cache_lock=self._cache_lock,
        )
        self.worker.set_prompts(self._prompt_snapshot())
        self.paused = False
        self.worker.start()
        self._status("Generating… (first step compiles kernels if --compile)")
        self._sync_enabled()

    def _stop_run(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=5.0)
        self.worker = None
        self.paused = False
        self._sync_enabled()

    def _toggle_play(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self._start_run()
        elif self.paused:
            self.paused = False
            self.worker.resume()
            self._status("Generating…")
            self._sync_enabled()
        else:
            self.paused = True
            self.worker.pause()
            self._status("Paused — adjust prompts, steps, or size")
            self._sync_enabled()

    def _apply_size(self, width: int, height: int) -> None:
        # Changing the output shape requires a fresh run. The window does NOT change — the
        # image is letterboxed into the image region — so orientation flips keep proportions.
        self._stop_run()
        self.width = snap_side(width)
        self.height = snap_side(height)
        self.width_entry.set_text(str(self.width))
        self.height_entry.set_text(str(self.height))
        self._status(f"Size set to {self.width}x{self.height} - press Play")

    def _resize_window(self, w: int, h: int) -> None:
        # Adopt the window's actual new size — do NOT call set_mode (that fights tiling WMs;
        # see __post_init__). The SDL surface tracks the resize on its own; we just relay
        # out the UI. The native minimum size (set at startup) keeps it from going too small.
        if (w, h) == (self.win_w, self.win_h):
            return
        self.win_w, self.win_h = w, h
        surface = pygame.display.get_surface()
        if surface is not None:
            self.screen = surface
        self.manager.set_window_resolution((w, h))
        self._build_ui()

    def _commit_steps(self) -> None:
        try:
            value = int(self.steps_entry.get_text())
        except ValueError:
            self.steps_entry.set_text(str(self.steps))
            return
        self.steps = max(1, min(1000, value))
        self.steps_entry.set_text(str(self.steps))
        # If a run is paused, changing steps abandons it (respacing is fixed per run).
        if self.worker is not None and self.worker.is_alive():
            self._stop_run()
            self._status(f"Steps set to {self.steps} — press Play to start fresh")

    def _open_save_dialog(self) -> None:
        """Freeze the current frame and open a file dialog to choose where to write it."""
        if self._frame_surface is None:
            self._status("Nothing to save yet")
            return
        if self._file_dialog is not None and self._file_dialog.alive():
            return  # a dialog is already open
        self._save_surface = self._frame_surface.copy()  # freeze; generation may advance
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        win_w, win_h = self._window_size()
        rect = pygame.Rect(0, 0, 560, 440)
        rect.center = (win_w // 2, win_h // 2)
        self._file_dialog = UIFileDialog(
            rect,
            self.manager,
            window_title="Save image",
            initial_file_path=str(self.out_dir / f"interactive_{stamp}.png"),
            allow_existing_files_only=False,
        )

    def _write_save(self, path_str: str) -> None:
        """Write the frozen frame to the path the dialog returned (defaulting to .png)."""
        if self._save_surface is None:
            return
        path = Path(path_str)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self._save_surface, str(path))
        self._save_surface = None
        self._status(f"Saved {path}")
        log.info("saved %s", path)

    # -- events --
    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.VIDEORESIZE:
            # Resize events stream while dragging; coalesce to one relayout per frame
            # (see run()) instead of rebuilding the UI on every event.
            self._pending_size = (event.w, event.h)
            return True

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.play_button:
                self._toggle_play()
            elif event.ui_element == self.stop_button:
                self._stop_run()
                self._status("Stopped")
            elif event.ui_element == self.save_button:
                self._open_save_dialog()
            elif event.ui_element == self.add_button:
                self.prompts.append(PromptRow("", 1.0))
                self._rebuild_prompt_rows()
                self._push_prompts()
            elif event.ui_element == self.apply_button:
                self._apply_size(
                    int_or(self.width_entry.get_text(), self.width),
                    int_or(self.height_entry.get_text(), self.height),
                )
            elif event.ui_element == self.swap_button:
                self._apply_size(self.height, self.width)
            elif event.ui_element in self._remove_buttons:
                idx = self._remove_buttons[event.ui_element]
                if 0 <= idx < len(self.prompts):
                    self.prompts.pop(idx)
                    self._rebuild_prompt_rows()
                    self._push_prompts()

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            slider_idx = self._weight_sliders.get(event.ui_element)
            if slider_idx is not None and 0 <= slider_idx < len(self.prompts):
                self.prompts[slider_idx].weight = float(event.value)
                self._refresh_rows()
                self._push_prompts()

        elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element == self.steps_entry:
                self._commit_steps()
            elif event.ui_element in (self.width_entry, self.height_entry):
                pass  # applied via the Apply button
            elif event.ui_element in self._prompt_entries:
                self._commit_prompt_entry(event.ui_element)

        elif event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            if event.ui_element is self._file_dialog:
                self._write_save(event.text)
                self._file_dialog = None

        elif event.type == pygame_gui.UI_WINDOW_CLOSE:
            if event.ui_element is self._file_dialog:  # cancelled
                self._file_dialog = None
                self._save_surface = None

        return True

    # -- drawing --
    def _update_frame_surface(self) -> None:
        if self.worker is None:
            return
        frame = self.worker.latest_frame()
        if frame is None:
            return
        key = (id(frame.image), frame.index)
        if key == self._frame_key:
            return
        self._frame_key = key
        # pygame.surfarray expects (W, H, 3), so swap the first two axes.
        self._frame_surface = pygame.surfarray.make_surface(frame.image.swapaxes(0, 1))
        self.step_label.set_text(f"step {frame.index} / {frame.total}")
        if self.worker.finished:
            self._status("Done — Save, or change settings and Play again")
            self._sync_enabled()

    def _auto_apply_on_blur(self) -> None:
        """Apply a prompt text box when keyboard focus leaves it (no Enter needed)."""
        focus = self.manager.get_focus_set() or set()
        current = next((e for e in self._prompt_entries if e in focus), None)
        if current is self._focused_entry:
            return
        previous = self._focused_entry
        self._focused_entry = current
        if previous is not None and previous.alive():
            self._commit_prompt_entry(previous)

    def _draw(self) -> None:
        win_w, win_h = self._window_size()
        img_h = self._image_area_h()
        self.screen.fill(WINDOW_BG)
        pygame.draw.rect(self.screen, IMAGE_BG, (0, 0, win_w, img_h))
        pygame.draw.rect(self.screen, PANEL_BG, (0, img_h, win_w, win_h - img_h))
        if self._frame_surface is not None:
            iw, ih = self._frame_surface.get_size()
            region = pygame.Rect(0, 0, win_w, img_h)
            scale = min(region.width / iw, region.height / ih)
            size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
            scaled = pygame.transform.smoothscale(self._frame_surface, size)
            self.screen.blit(scaled, scaled.get_rect(center=region.center))
        pygame.draw.line(self.screen, DIVIDER, (0, img_h), (win_w, img_h))

    # -- main loop --
    def run(self) -> None:
        clock = pygame.time.Clock()
        alive = True
        while alive:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self.manager.process_events(event)
                if not self._handle_event(event):
                    alive = False
            # Apply at most one window relayout per frame (resize events are coalesced).
            if self._pending_size is not None:
                w, h = self._pending_size
                self._pending_size = None
                self._resize_window(w, h)
            # Reflect a worker that finished/exited on its own.
            if self.worker is not None and not self.worker.is_alive() and not self.paused:
                self._sync_enabled()
            self._auto_apply_on_blur()
            # Live "edited · Enter" badge while typing (cheap; only mutates on change).
            self._refresh_rows()
            self._update_frame_surface()
            self.manager.update(dt)
            self._draw()
            self.manager.draw_ui(self.screen)
            pygame.display.flip()
        self._stop_run()


def int_or(text: str, fallback: int) -> int:
    try:
        return int(text)
    except ValueError:
        return fallback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100, help="Initial total step count.")
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the UNet/CLIP (~2x faster once warm, but recompiles ~90s on "
        "each new image size).",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU (very slow).")
    # Defaults are repo-root-relative, matching the library, so weights/outputs are shared
    # with it (run from the repo root: `uv run disco-studio`) rather than re-downloaded.
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Where model weights live (defaults to the library's models dir).",
    )
    ap.add_argument("--out", type=Path, default=Path("images_out"), help="Where Save writes PNGs.")
    args = ap.parse_args()

    pygame.init()
    config = RunConfig(
        compile=args.compile,
        cpu=args.cpu,
        models_dir=args.models_dir,
        width=snap_side(args.width),
        height=snap_side(args.height),
        steps=args.steps,
    )
    log.info("loading models (this can take a minute)…")
    session = DiscoSession(config)
    log.info("models loaded")

    app = App(
        session=session,
        out_dir=args.out,
        width=args.width,
        height=args.height,
        steps=args.steps,
    )
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
