"""
BlockProgrammingWidget — Mindstorms-EV3-style visual programming for Mode 3.

Three-pane layout:

    ┌─────────────┬───────────────────────────┬─────────────────┐
    │  PALETTE    │  PROGRAM CANVAS            │  PARAMETERS     │
    │             │                            │                 │
    │  Move Fwd   │  1. Move Fwd  0.50 m       │  Distance: 0.5  │
    │  Move Back  │  2. Turn Left 90°          │                 │
    │  Turn Left  │  3. Find Object  bottle    │                 │
    │  Turn Right │  4. Approach   0.3 m       │                 │
    │  Wait       │  5. Read Gauge             │                 │
    │  Find Obj   │  6. Go To Base             │                 │
    │  Approach   │  …                         │                 │
    │  Read Gauge │                            │                 │
    │  …          │                            │                 │
    └─────────────┴───────────────────────────┴─────────────────┘
            ▶ Run        ■ Stop       ⟲ Clear       💾 Save / Load

Drag a block from the palette into the canvas (or double-click the palette
entry) to add it. Drag inside the canvas to re-order. Click a block to edit
its parameters in the right pane. Hit Run and the robot executes each block
in sequence; the block currently running is highlighted purple.

The runner translates blocks into the right combination of `/qbot3/cmd_vel`,
`/qbot3/precise_cmd`, and BaseSkill instantiations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.shared_state import SharedState
from gui.theme import Tokens
from skills.approach_object import ApproachObjectSkill
from skills.go_to_position import GoToPositionSkill
from skills.line_follower import LineFollowerSkill
from skills.map_room import MapRoomSkill
from skills.read_gauge import ReadGaugeSkill
from skills.return_to_base import ReturnToBaseSkill
from skills.search_object import SearchObjectSkill
from skills.speak_text import SpeakTextSkill
from skills.wall_follower import WallFollowerSkill

logger = logging.getLogger(__name__)


# =====================================================================
# Block specification + catalog
# =====================================================================


@dataclass(frozen=True)
class ParamSpec:
    """One parameter slot on a block."""
    key: str
    label: str
    kind: str                      # "float" | "int" | "str" | "choice" | "bool"
    default: Any
    description: str = ""          # one-sentence help text shown under the editor
    units: str = ""
    minimum: float = -1000.0
    maximum: float = 1000.0
    step: float = 0.1
    choices: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockSpec:
    """Static metadata for a block type."""
    block_id: str
    name: str
    description: str
    color: str                     # hex; used to tint the canvas item
    category: str                  # "Move" / "Turn" / "Wait" / "Sense" / "Skill"
    params: Tuple[ParamSpec, ...] = ()


# Colour palette for block categories — tied to the dark theme accents
_C_MOVE   = Tokens.ACCENT_PRIMARY        # purple
_C_TURN   = Tokens.ACCENT_SECONDARY      # teal
_C_WAIT   = Tokens.TEXT_MUTED            # grey
_C_SENSE  = Tokens.WARNING               # amber
_C_SKILL  = Tokens.SUCCESS               # green


BLOCK_CATALOG: Dict[str, BlockSpec] = {
    # ---- Movement ----
    "move_forward": BlockSpec(
        block_id="move_forward",
        name="Move Forward",
        description="Drive forward by a fixed distance (closed-loop).",
        color=_C_MOVE, category="Move",
        params=(ParamSpec(
            "distance_m", "Distance", "float", 0.5,
            description="How far to drive forward in metres. The Pi's "
                        "closed-loop controller stops within ~2 cm of this target.",
            units="m", minimum=0.01, maximum=5.0, step=0.05),),
    ),
    "move_backward": BlockSpec(
        block_id="move_backward",
        name="Move Backward",
        description="Drive backward by a fixed distance (closed-loop).",
        color=_C_MOVE, category="Move",
        params=(ParamSpec(
            "distance_m", "Distance", "float", 0.5,
            description="How far to reverse in metres. Same closed-loop "
                        "accuracy as Move Forward.",
            units="m", minimum=0.01, maximum=5.0, step=0.05),),
    ),
    # ---- Turning ----
    "turn_left": BlockSpec(
        block_id="turn_left",
        name="Turn Left",
        description="Rotate counter-clockwise by a fixed angle.",
        color=_C_TURN, category="Turn",
        params=(ParamSpec(
            "angle_deg", "Angle", "float", 90.0,
            description="Rotation amount in degrees, counter-clockwise. "
                        "Uses the IMU yaw to verify completion (±2°).",
            units="°", minimum=1.0, maximum=360.0, step=5.0),),
    ),
    "turn_right": BlockSpec(
        block_id="turn_right",
        name="Turn Right",
        description="Rotate clockwise by a fixed angle.",
        color=_C_TURN, category="Turn",
        params=(ParamSpec(
            "angle_deg", "Angle", "float", 90.0,
            description="Rotation amount in degrees, clockwise. "
                        "Uses the IMU yaw to verify completion (±2°).",
            units="°", minimum=1.0, maximum=360.0, step=5.0),),
    ),
    # ---- Pause ----
    "wait": BlockSpec(
        block_id="wait",
        name="Wait",
        description="Pause execution for a fixed time.",
        color=_C_WAIT, category="Wait",
        params=(ParamSpec(
            "seconds", "Duration", "float", 1.0,
            description="How long to pause before running the next block. "
                        "Useful between actions for sensors to settle.",
            units="s", minimum=0.1, maximum=60.0, step=0.1),),
    ),
    # ---- Sensing / search ----
    "find_object": BlockSpec(
        block_id="find_object",
        name="Find Object",
        description="Rotate in place looking for the target YOLO class.",
        color=_C_SENSE, category="Sense",
        params=(
            ParamSpec(
                "target_class", "Object", "str", "bottle",
                description="YOLO class name to search for. Use the "
                            "vocabulary from the COCO dataset (e.g. bottle, "
                            "chair, person, laptop, clock, tv, book, cup)."),
            ParamSpec(
                "confidence", "Min confidence", "float", 0.5,
                description="Reject detections below this score (0.0–1.0). "
                            "Higher = fewer false positives, but may miss "
                            "the target at distance.",
                minimum=0.05, maximum=0.99, step=0.05),
        ),
    ),
    "read_gauge": BlockSpec(
        block_id="read_gauge",
        name="Read Gauge",
        description="Approach and read a gauge using the VLM.",
        color=_C_SENSE, category="Sense",
        params=(
            ParamSpec(
                "target_class", "Gauge class", "str", "gauge",
                description="YOLO class to lock onto before reading "
                            "(commonly 'clock' or 'tv' for COCO; train a "
                            "custom class for industrial gauges)."),
            ParamSpec(
                "approach_distance_m", "Stop at", "float", 0.6,
                description="Distance from the gauge before stopping and "
                            "asking the VLM to read it. Closer = larger crop, "
                            "easier reading; too close clips the dial.",
                units="m", minimum=0.2, maximum=2.0, step=0.05),
        ),
    ),
    # ---- Skills ----
    "approach_object": BlockSpec(
        block_id="approach_object",
        name="Approach Object",
        description="Drive toward a detected object until close.",
        color=_C_SKILL, category="Skill",
        params=(
            ParamSpec(
                "target_class", "Object", "str", "chair",
                description="YOLO class to approach. The robot picks the "
                            "highest-confidence detection of this class in view."),
            ParamSpec(
                "stop_distance_m", "Stop at", "float", 0.5,
                description="Stand-off distance once the object is in front. "
                            "0.4–0.6 m is a good default for objects you'll "
                            "interact with.",
                units="m", minimum=0.2, maximum=2.0, step=0.05),
        ),
    ),
    "go_to_position": BlockSpec(
        block_id="go_to_position",
        name="Go To Position",
        description="Navigate to an (x, y) target in the odom frame.",
        color=_C_SKILL, category="Skill",
        params=(
            ParamSpec(
                "x", "X", "float", 0.0,
                description="Target X coordinate in metres. The X axis points "
                            "in whatever direction the robot was facing when "
                            "the app started (odom frame origin).",
                units="m", minimum=-20.0, maximum=20.0, step=0.1),
            ParamSpec(
                "y", "Y", "float", 0.0,
                description="Target Y coordinate in metres. The Y axis is to "
                            "the robot's left at start-up.",
                units="m", minimum=-20.0, maximum=20.0, step=0.1),
            ParamSpec(
                "tolerance_m", "Tolerance", "float", 0.10,
                description="How close the robot must get to the target before "
                            "the block is considered done. Lower = more "
                            "precise but slower (more course corrections).",
                units="m", minimum=0.02, maximum=1.0, step=0.02),
        ),
    ),
    "return_to_base": BlockSpec(
        block_id="return_to_base",
        name="Go To Base",
        description="Drive to the saved 'base' waypoint.",
        color=_C_SKILL, category="Skill",
        params=(),
    ),
    "map_room": BlockSpec(
        block_id="map_room",
        name="Map Room",
        description="Sweep the room to build a SLAM map.",
        color=_C_SKILL, category="Skill",
        params=(ParamSpec(
            "coverage_threshold_pct", "Target coverage", "float", 80.0,
            description="Stop sweeping once this percent of the explored "
                        "bounding box has been observed. Higher = more "
                        "thorough map, but takes much longer in big rooms.",
            units="%", minimum=10.0, maximum=100.0, step=5.0),),
    ),
    "wall_follow": BlockSpec(
        block_id="wall_follow",
        name="Wall Follow",
        description="Follow a wall for a duration, then stop.",
        color=_C_SKILL, category="Skill",
        params=(
            ParamSpec(
                "wall_side", "Side", "choice", "right",
                description="Which side the robot keeps the wall on. 'right' "
                            "= clockwise loop around a room; 'left' = counter "
                            "-clockwise.",
                choices=("right", "left")),
            ParamSpec(
                "target_distance_m", "Distance", "float", 0.25,
                description="Standoff distance to maintain from the wall. "
                            "0.20–0.35 m works well in indoor corridors.",
                units="m", minimum=0.1, maximum=1.0, step=0.05),
            ParamSpec(
                "duration_s", "Duration", "float", 10.0,
                description="How long to follow before automatically stopping "
                            "and moving to the next block.",
                units="s", minimum=1.0, maximum=300.0, step=1.0),
        ),
    ),
    "line_follow": BlockSpec(
        block_id="line_follow",
        name="Line Follow",
        description="Follow a coloured line for a duration, then stop.",
        color=_C_SKILL, category="Skill",
        params=(ParamSpec(
            "duration_s", "Duration", "float", 10.0,
            description="How long to follow the line before stopping. "
                        "Line colour + bottom-band region are taken from "
                        "config/skills_config.yaml (defaults to a black line).",
            units="s", minimum=1.0, maximum=300.0, step=1.0),),
    ),
    "speak_text": BlockSpec(
        block_id="speak_text",
        name="Speak Text",
        description="Play typed text aloud using OpenAI TTS (text-to-speech).",
        color=_C_SENSE, category="Sense",
        params=(ParamSpec(
            "text", "Text", "str", "Hello, I am QBot.",
            description="The text to speak aloud through the speaker. "
                        "Uses OpenAI TTS (model tts-1, voice nova)."),),
    ),
}


# =====================================================================
# ProgramBlock — one instance in a program
# =====================================================================


@dataclass
class ProgramBlock:
    block_id: str
    params: Dict[str, Any] = field(default_factory=dict)

    def with_defaults(self) -> "ProgramBlock":
        spec = BLOCK_CATALOG.get(self.block_id)
        if spec is None:
            return self
        merged = {p.key: p.default for p in spec.params}
        merged.update(self.params)
        self.params = merged
        return self

    def summary(self) -> str:
        spec = BLOCK_CATALOG.get(self.block_id)
        if spec is None:
            return self.block_id
        bits: List[str] = []
        for p in spec.params:
            v = self.params.get(p.key, p.default)
            if isinstance(v, float):
                bits.append(f"{v:g}{p.units}")
            else:
                bits.append(f"{v}{p.units}")
        return f"{spec.name}  " + "  ·  ".join(bits) if bits else spec.name

    def to_dict(self) -> Dict[str, Any]:
        return {"block_id": self.block_id, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgramBlock":
        b = cls(block_id=str(data["block_id"]),
                params=dict(data.get("params", {})))
        return b.with_defaults()


# =====================================================================
# Drag-drop list widgets
# =====================================================================


_BLOCK_MIME = "application/x-qbot3-block-id"


class _PaletteList(QListWidget):
    """Source list — drag-only, single selection. Each item carries its block_id."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(False)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setSpacing(2)


class _CanvasList(QListWidget):
    """Program canvas — accepts drops from the palette + internal reorder."""

    block_added = pyqtSignal(int, str)        # row, block_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setSpacing(2)

    def dropEvent(self, event: QDropEvent) -> None:
        source = event.source()
        if source is self:
            super().dropEvent(event)
            return
        if isinstance(source, _PaletteList):
            item = source.currentItem()
            if item is None:
                event.ignore()
                return
            block_id = str(item.data(Qt.ItemDataRole.UserRole))
            # Where in the canvas does the user want the new block?
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            target_row = self.indexAt(pos).row()
            row = target_row if target_row != -1 else self.count()
            self.block_added.emit(row, block_id)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


# =====================================================================
# Block-program runner
# =====================================================================


class _ProgramRunner:
    """Translates a list of ProgramBlocks into runtime actions."""

    def __init__(self, state: SharedState, ros: Any, *,
                 vlm_pipeline: Any = None, vlm_model_name: str = "gpt-4o",
                 skills_config: Optional[Dict[str, Any]] = None,
                 voice_io: Any = None) -> None:
        self.state = state
        self.ros = ros
        self.vlm = vlm_pipeline
        self.vlm_model = vlm_model_name
        self.skills_config = skills_config or {}
        self.voice_io = voice_io
        self._cancel = False
        self._active_skill: Any = None

    def cancel(self) -> None:
        self._cancel = True
        if self._active_skill is not None:
            try:
                self._active_skill.abort()
            except Exception:
                pass
        try:
            self.ros.publish_emergency_stop()
        except Exception:
            pass

    async def run(self, blocks: List[ProgramBlock],
                  on_progress: Callable[[int, int, str], None]) -> bool:
        """Execute blocks sequentially. Returns True if every block succeeded."""
        self._cancel = False
        total = len(blocks)
        for idx, block in enumerate(blocks):
            if self._cancel:
                return False
            on_progress(idx, total, "running")
            ok = await self._dispatch(block)
            if not ok:
                on_progress(idx, total, "failed")
                # Stop on first failure — graceful default for the demo
                try:
                    self.ros.publish_cmd_vel(0.0, 0.0)
                except Exception:
                    pass
                return False
            on_progress(idx, total, "done")
        return True

    async def _dispatch(self, block: ProgramBlock) -> bool:
        bid = block.block_id
        p = block.params
        try:
            if bid == "move_forward":
                return await self._precise(distance_m=float(p["distance_m"]))
            if bid == "move_backward":
                return await self._precise(distance_m=-float(p["distance_m"]))
            if bid == "turn_left":
                return await self._precise(angle_rad=math.radians(float(p["angle_deg"])))
            if bid == "turn_right":
                return await self._precise(angle_rad=-math.radians(float(p["angle_deg"])))
            if bid == "wait":
                end = time.monotonic() + float(p["seconds"])
                while time.monotonic() < end:
                    if self._cancel:
                        return False
                    await asyncio.sleep(0.1)
                return True

            # ---- Skill-based blocks ----
            if bid == "find_object":
                skill = SearchObjectSkill(self.state, self.ros,
                                          skills_config=self.skills_config)
                self._active_skill = skill
                result = await skill.run({
                    "target_class": str(p["target_class"]),
                    "confidence_threshold": float(p.get("confidence", 0.5)),
                })
                self._active_skill = None
                return bool(result.success)

            if bid == "approach_object":
                skill = ApproachObjectSkill(self.state, self.ros,
                                            skills_config=self.skills_config)
                self._active_skill = skill
                result = await skill.run({
                    "target_class": str(p["target_class"]),
                    "stop_distance_m": float(p["stop_distance_m"]),
                })
                self._active_skill = None
                return bool(result.success)

            if bid == "read_gauge":
                if self.vlm is None:
                    logger.warning("read_gauge block: VLM pipeline not provided")
                    return False
                skill = ReadGaugeSkill(
                    self.state, self.ros,
                    vlm_pipeline=self.vlm, vlm_model_name=self.vlm_model,
                    skills_config=self.skills_config,
                )
                self._active_skill = skill
                result = await skill.run({
                    "target_class": str(p.get("target_class", "gauge")),
                    "approach_distance_m": float(p["approach_distance_m"]),
                })
                self._active_skill = None
                return bool(result.success)

            if bid == "go_to_position":
                skill = GoToPositionSkill(self.state, self.ros,
                                          skills_config=self.skills_config)
                self._active_skill = skill
                result = await skill.run({
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "tolerance_m": float(p["tolerance_m"]),
                })
                self._active_skill = None
                return bool(result.success)

            if bid == "return_to_base":
                skill = ReturnToBaseSkill(self.state, self.ros,
                                          skills_config=self.skills_config)
                self._active_skill = skill
                result = await skill.run({})
                self._active_skill = None
                return bool(result.success)

            if bid == "map_room":
                skill = MapRoomSkill(self.state, self.ros,
                                     skills_config=self.skills_config)
                self._active_skill = skill
                result = await skill.run({
                    "coverage_threshold_pct": float(p["coverage_threshold_pct"]),
                })
                self._active_skill = None
                return bool(result.success)

            if bid == "wall_follow":
                skill = WallFollowerSkill(self.state, self.ros,
                                          skills_config=self.skills_config)
                self._active_skill = skill
                # Wall-follow runs forever — bound by the user's duration_s
                duration = float(p["duration_s"])
                params = {
                    "wall_side": str(p["wall_side"]),
                    "target_distance_m": float(p["target_distance_m"]),
                }

                async def _bounded() -> Any:
                    try:
                        return await asyncio.wait_for(skill.run(params), timeout=duration)
                    except asyncio.TimeoutError:
                        skill.abort()
                        return None

                await _bounded()
                self._active_skill = None
                return True

            if bid == "line_follow":
                skill = LineFollowerSkill(self.state, self.ros,
                                          skills_config=self.skills_config)
                self._active_skill = skill
                duration = float(p["duration_s"])
                try:
                    await asyncio.wait_for(skill.run({}), timeout=duration)
                except asyncio.TimeoutError:
                    skill.abort()
                self._active_skill = None
                return True

            if bid == "speak_text":
                skill = SpeakTextSkill(
                    self.state, self.ros,
                    voice_io=self.voice_io,
                    skills_config=self.skills_config,
                )
                self._active_skill = skill
                result = await skill.run({"text": str(p.get("text", ""))})
                self._active_skill = None
                return bool(result.success)

            logger.warning("Unknown block_id: %s", bid)
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Block '%s' raised: %s", bid, exc)
            self._active_skill = None
            return False

    async def _precise(self, *, distance_m: float = 0.0,
                       angle_rad: float = 0.0,
                       timeout_s: float = 30.0) -> bool:
        """Issue a /qbot3/precise_cmd goal and wait for /motion/result."""
        with self.state.lock:
            self.state.motion_feedback.last_result = None
            self.state.motion_feedback.status = "idle"
        self.ros.publish_precise_cmd(distance_m=distance_m, angle_rad=angle_rad)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._cancel:
                return False
            await asyncio.sleep(0.05)
            if self.state.motion_feedback.status in ("moving", "turning", "emergency_stop"):
                break

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._cancel:
                return False
            await asyncio.sleep(0.1)
            with self.state.lock:
                status = self.state.motion_feedback.status
                result = self.state.motion_feedback.last_result
            if status in ("idle", "emergency_stop") and result is not None:
                return bool(result) and status != "emergency_stop"
        try:
            self.ros.publish_emergency_stop()
        except Exception:
            pass
        return False


# =====================================================================
# BlockProgrammingWidget
# =====================================================================


_PROGRAMS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "programs",
)


class BlockProgrammingWidget(QWidget):
    """Mode-3 right panel — drag-drop block programming canvas."""

    program_started = pyqtSignal()
    program_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        state: SharedState,
        ros: Any,
        *,
        vlm_pipeline: Any = None,
        vlm_model_name: str = "gpt-4o",
        skills_config: Optional[Dict[str, Any]] = None,
        voice_io: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.ros = ros
        self._runner = _ProgramRunner(
            state, ros,
            vlm_pipeline=vlm_pipeline, vlm_model_name=vlm_model_name,
            skills_config=skills_config,
            voice_io=voice_io,
        )
        self._task: Optional[asyncio.Task] = None
        self._blocks: List[ProgramBlock] = []
        self._selected_param_widgets: Dict[str, Any] = {}

        self._build_ui()

    # ---------------------------------------------------------------
    # UI build
    # ---------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        outer.addLayout(self._build_toolbar())
        outer.addWidget(self._build_three_panes(), 1)
        outer.addWidget(self._build_status_strip())

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._run_btn = QPushButton("Run")
        self._run_btn.setProperty("variant", "primary")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.clicked.connect(self._on_run)
        bar.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("variant", "danger")
        self._stop_btn.setMinimumHeight(34)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        bar.addWidget(self._stop_btn)

        bar.addSpacing(12)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setProperty("variant", "ghost")
        self._clear_btn.clicked.connect(self._on_clear)
        bar.addWidget(self._clear_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setProperty("variant", "ghost")
        self._save_btn.clicked.connect(self._on_save)
        bar.addWidget(self._save_btn)

        self._load_btn = QPushButton("Load")
        self._load_btn.setProperty("variant", "ghost")
        self._load_btn.clicked.connect(self._on_load)
        bar.addWidget(self._load_btn)

        bar.addStretch(1)
        return bar

    def _build_three_panes(self) -> QSplitter:
        split = QSplitter(Qt.Orientation.Horizontal)

        # ---- Palette ----
        palette_card = self._wrap_card("BLOCKS", self._build_palette())
        palette_card.setMinimumWidth(160)
        split.addWidget(palette_card)

        # ---- Canvas ----
        self._canvas = _CanvasList()
        self._canvas.block_added.connect(self._insert_new_block)
        self._canvas.itemSelectionChanged.connect(self._on_canvas_selection)
        self._canvas.model().rowsMoved.connect(self._on_rows_moved)
        canvas_card = self._wrap_card("PROGRAM", self._canvas)
        split.addWidget(canvas_card)

        # ---- Parameters ----
        self._params_panel = QWidget()
        self._params_layout = QVBoxLayout(self._params_panel)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.setSpacing(8)
        self._params_placeholder = QLabel("Select a block to edit its parameters")
        self._params_placeholder.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; font-size: 11px;"
        )
        self._params_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._params_layout.addWidget(self._params_placeholder)
        self._params_layout.addStretch(1)
        params_card = self._wrap_card("PARAMETERS", self._params_panel)
        params_card.setMinimumWidth(180)
        split.addWidget(params_card)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 3)
        return split

    def _build_palette(self) -> _PaletteList:
        palette = _PaletteList()
        for spec in BLOCK_CATALOG.values():
            item = QListWidgetItem(spec.name)
            item.setData(Qt.ItemDataRole.UserRole, spec.block_id)
            item.setToolTip(f"{spec.name}\n{spec.description}\n\nDrag onto the program.")
            color = QColor(spec.color)
            color.setAlpha(48)
            item.setBackground(color)
            palette.addItem(item)
        palette.itemDoubleClicked.connect(self._on_palette_double_click)
        return palette

    def _wrap_card(self, title: str, content: QWidget) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(6)
        cap = QLabel(title)
        cap.setProperty("role", "caption")
        v.addWidget(cap)
        v.addWidget(content, 1)
        return card

    def _build_status_strip(self) -> QFrame:
        strip = QFrame()
        strip.setProperty("role", "card")
        h = QHBoxLayout(strip)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(12)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(
            f"color: {Tokens.TEXT_SECONDARY}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        h.addWidget(self._status_label, 1)

        self._step_label = QLabel("0 / 0")
        self._step_label.setStyleSheet(
            f"color: {Tokens.TEXT_MUTED}; "
            f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 11px;"
        )
        h.addWidget(self._step_label)
        return strip

    # ---------------------------------------------------------------
    # Block management
    # ---------------------------------------------------------------

    def _insert_new_block(self, row: int, block_id: str) -> None:
        block = ProgramBlock(block_id=block_id).with_defaults()
        idx = max(0, min(row, len(self._blocks)))
        self._blocks.insert(idx, block)
        self._refresh_canvas(select_row=idx)

    def _on_palette_double_click(self, item: QListWidgetItem) -> None:
        block_id = str(item.data(Qt.ItemDataRole.UserRole))
        self._insert_new_block(len(self._blocks), block_id)

    def _on_rows_moved(self, *_args: Any) -> None:
        # Re-derive the in-memory list from canvas item order.
        new_blocks: List[ProgramBlock] = []
        for i in range(self._canvas.count()):
            item = self._canvas.item(i)
            block = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(block, ProgramBlock):
                new_blocks.append(block)
        self._blocks = new_blocks
        self._refresh_canvas(select_row=self._canvas.currentRow())

    def _refresh_canvas(self, *, select_row: int = -1) -> None:
        self._canvas.blockSignals(True)
        self._canvas.clear()
        for i, block in enumerate(self._blocks):
            spec = BLOCK_CATALOG.get(block.block_id)
            label = f"{i + 1}.  {block.summary()}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, block)
            if spec is not None:
                color = QColor(spec.color)
                color.setAlpha(64)
                item.setBackground(color)
                item.setToolTip(spec.description)
            self._canvas.addItem(item)
        self._canvas.blockSignals(False)
        if select_row >= 0:
            self._canvas.setCurrentRow(min(select_row, self._canvas.count() - 1))
        self._step_label.setText(f"0 / {len(self._blocks)}")

    def _on_canvas_selection(self) -> None:
        row = self._canvas.currentRow()
        if row < 0 or row >= len(self._blocks):
            self._show_param_placeholder()
            return
        self._build_param_editor(self._blocks[row])

    def _show_param_placeholder(self) -> None:
        self._clear_param_layout()
        self._params_layout.addWidget(self._params_placeholder)
        self._params_layout.addStretch(1)
        self._params_placeholder.show()

    def _clear_param_layout(self) -> None:
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._params_placeholder:
                widget.deleteLater()
        self._params_placeholder.hide()
        self._selected_param_widgets = {}

    def _build_param_editor(self, block: ProgramBlock) -> None:
        spec = BLOCK_CATALOG.get(block.block_id)
        if spec is None:
            return
        self._clear_param_layout()

        title = QLabel(spec.name)
        title.setProperty("role", "heading")
        self._params_layout.addWidget(title)

        desc = QLabel(spec.description)
        desc.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 11px;")
        desc.setWordWrap(True)
        self._params_layout.addWidget(desc)

        if not spec.params:
            empty = QLabel("(no parameters)")
            empty.setStyleSheet(f"color: {Tokens.TEXT_MUTED}; font-size: 11px;")
            self._params_layout.addWidget(empty)
            self._params_layout.addStretch(1)
            return

        # One vertical card per parameter — clearer than a tight QFormLayout
        # because we want the description hint visible right under each editor.
        for param in spec.params:
            self._params_layout.addWidget(self._build_param_card(block, param))

        # Delete button
        del_btn = QPushButton("Delete block")
        del_btn.setProperty("variant", "danger")
        del_btn.clicked.connect(self._on_delete_selected)
        self._params_layout.addWidget(del_btn)
        self._params_layout.addStretch(1)

    def _build_param_card(self, block: ProgramBlock, param: ParamSpec) -> QWidget:
        """One self-contained editor row: bold label + units + editor + help."""
        card = QFrame()
        card.setProperty("role", "card")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)

        # Top line: label (left) + units pill (right)
        header = QHBoxLayout()
        header.setSpacing(6)

        label = QLabel(param.label)
        label.setStyleSheet(
            f"color: {Tokens.TEXT_PRIMARY}; font-weight: 600; font-size: 12px;"
        )
        header.addWidget(label)
        header.addStretch(1)

        if param.units:
            units = QLabel(param.units)
            units.setStyleSheet(
                f"color: {Tokens.ACCENT_SECONDARY}; "
                f"font-family: {Tokens.FONT_FAMILY_MONO}; font-size: 10px; "
                f"background: rgba(0,212,170,30); border-radius: 4px; padding: 1px 6px;"
            )
            header.addWidget(units)

        v.addLayout(header)

        # The editor widget itself
        widget = self._make_param_widget(param, block.params.get(param.key, param.default))
        widget.setToolTip(param.description or param.label)
        self._selected_param_widgets[param.key] = widget
        v.addWidget(widget)

        # One-line help text
        if param.description:
            hint = QLabel(param.description)
            hint.setStyleSheet(
                f"color: {Tokens.TEXT_MUTED}; font-size: 10px; line-height: 130%;"
            )
            hint.setWordWrap(True)
            v.addWidget(hint)

        return card

    def _make_param_widget(self, spec: ParamSpec, value: Any) -> QWidget:
        if spec.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(spec.minimum, spec.maximum)
            w.setSingleStep(spec.step)
            w.setDecimals(3 if spec.step < 0.01 else 2)
            w.setValue(float(value))
            w.valueChanged.connect(lambda v, k=spec.key: self._on_param_changed(k, v))
            return w
        if spec.kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.minimum), int(spec.maximum))
            w.setValue(int(value))
            w.valueChanged.connect(lambda v, k=spec.key: self._on_param_changed(k, v))
            return w
        if spec.kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            cb.toggled.connect(lambda v, k=spec.key: self._on_param_changed(k, v))
            return cb
        if spec.kind == "choice":
            combo = QComboBox()
            combo.addItems(list(spec.choices))
            if str(value) in spec.choices:
                combo.setCurrentText(str(value))
            combo.currentTextChanged.connect(
                lambda v, k=spec.key: self._on_param_changed(k, v))
            return combo
        # default: string
        le = QLineEdit(str(value))
        le.editingFinished.connect(
            lambda k=spec.key, w=le: self._on_param_changed(k, w.text()))
        return le

    def _on_param_changed(self, key: str, value: Any) -> None:
        row = self._canvas.currentRow()
        if row < 0 or row >= len(self._blocks):
            return
        self._blocks[row].params[key] = value
        # Update the canvas label without rebuilding everything else
        spec = BLOCK_CATALOG.get(self._blocks[row].block_id)
        item = self._canvas.item(row)
        if item is not None:
            item.setText(f"{row + 1}.  {self._blocks[row].summary()}")
            if spec is not None:
                color = QColor(spec.color)
                color.setAlpha(64)
                item.setBackground(color)
            item.setData(Qt.ItemDataRole.UserRole, self._blocks[row])

    def _on_delete_selected(self) -> None:
        row = self._canvas.currentRow()
        if row < 0 or row >= len(self._blocks):
            return
        del self._blocks[row]
        self._refresh_canvas(select_row=min(row, len(self._blocks) - 1))
        if not self._blocks:
            self._show_param_placeholder()

    # ---------------------------------------------------------------
    # Toolbar handlers
    # ---------------------------------------------------------------

    def _on_clear(self) -> None:
        if not self._blocks:
            return
        self._blocks.clear()
        self._refresh_canvas(select_row=-1)
        self._show_param_placeholder()
        self._status_label.setText("Cleared")

    def _on_save(self) -> None:
        os.makedirs(_PROGRAMS_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Program", _PROGRAMS_DIR, "QBot3 Program (*.json)",
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        payload = [b.to_dict() for b in self._blocks]
        try:
            with open(path, "w") as f:
                json.dump({"blocks": payload, "version": 1}, f, indent=2)
            self._status_label.setText(f"Saved {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_load(self) -> None:
        os.makedirs(_PROGRAMS_DIR, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Program", _PROGRAMS_DIR, "QBot3 Program (*.json)",
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            blocks = [ProgramBlock.from_dict(d) for d in data.get("blocks", [])]
            self._blocks = blocks
            self._refresh_canvas(select_row=0 if blocks else -1)
            self._status_label.setText(f"Loaded {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _on_run(self) -> None:
        if self._task is not None and not self._task.done():
            return
        if not self._blocks:
            self._status_label.setText("Add some blocks first")
            return
        # Block program runner uses the same skill stack as the AI mode,
        # so the same calibration gate applies.
        with self.state.lock:
            calibrated = self.state.imu_calibrated
        if not calibrated:
            self._status_label.setText(
                "Gyroscope calibration in progress — wait for the banner"
            )
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._task = loop.create_task(self._run_async())
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Running…")
        self.program_started.emit()

    def _on_stop(self) -> None:
        self._runner.cancel()
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Stopping…")

    async def _run_async(self) -> None:
        success = False
        message = "stopped"
        try:
            success = await self._runner.run(list(self._blocks), self._on_runner_progress)
            message = "complete" if success else "stopped or failed"
        except asyncio.CancelledError:
            success = False
            message = "cancelled"
        except Exception as exc:
            logger.exception("Program crashed: %s", exc)
            success = False
            message = f"error: {type(exc).__name__}"
        finally:
            try:
                self.ros.publish_cmd_vel(0.0, 0.0)
            except Exception:
                pass
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._status_label.setText(message)
            self._highlight_block(-1)
            self.program_finished.emit(success, message)
            self._task = None

    def _on_runner_progress(self, idx: int, total: int, status: str) -> None:
        self._step_label.setText(f"{idx + 1} / {total}  ·  {status}")
        if status in ("running",):
            self._highlight_block(idx)
        elif status == "done":
            self._mark_block_done(idx)

    def _highlight_block(self, row: int) -> None:
        for i in range(self._canvas.count()):
            item = self._canvas.item(i)
            if item is None:
                continue
            spec = BLOCK_CATALOG.get(self._blocks[i].block_id) if i < len(self._blocks) else None
            if i == row:
                color = QColor(Tokens.ACCENT_PRIMARY)
                color.setAlpha(180)
            elif spec is not None:
                color = QColor(spec.color)
                color.setAlpha(64)
            else:
                color = QColor(Tokens.SURFACE_ELEVATED)
            item.setBackground(color)

    def _mark_block_done(self, row: int) -> None:
        if row < 0 or row >= self._canvas.count():
            return
        item = self._canvas.item(row)
        if item is None:
            return
        color = QColor(Tokens.SUCCESS)
        color.setAlpha(80)
        item.setBackground(color)

    # ---------------------------------------------------------------
    # Public hook for main_window
    # ---------------------------------------------------------------

    def set_vlm(self, pipeline: Any, model_name: str = "gpt-4o") -> None:
        """Update the VLM reference if the user changes models in settings."""
        self._runner.vlm = pipeline
        self._runner.vlm_model = model_name
