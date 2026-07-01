"""
Skills package — concrete BaseSkill subclasses.

`SKILL_CLASSES` is the canonical name->class registry used by:
    * the GUI skill cards (mode 3 panel)
    * the LLM planner (lists allowed skill names in its context)
    * the AI mode dispatcher (instantiates a skill from an LLM `skill_command`)
"""

from __future__ import annotations

from typing import Any, Dict, Type

from skills.base_skill import BaseSkill, SkillAborted, SkillResult
from skills.approach_object import ApproachObjectSkill
from skills.explore_room import ExploreRoomSkill
from skills.go_to_object import GoToObjectSkill
from skills.go_to_position import GoToPositionSkill
from skills.line_follower import LineFollowerSkill
from skills.map_room import MapRoomSkill
from skills.read_gauge import ReadGaugeSkill
from skills.return_to_base import ReturnToBaseSkill
from skills.search_object import SearchObjectSkill
from skills.seek_object import SeekObjectSkill
from skills.sequential_approach import SequentialApproachSkill
from skills.vlm_reach import VlmReachSkill
from skills.speak_text import SpeakTextSkill
from skills.wall_follower import WallFollowerSkill

SKILL_CLASSES: Dict[str, Type[BaseSkill]] = {
    WallFollowerSkill.name: WallFollowerSkill,
    LineFollowerSkill.name: LineFollowerSkill,
    ReturnToBaseSkill.name: ReturnToBaseSkill,
    SearchObjectSkill.name: SearchObjectSkill,
    SeekObjectSkill.name: SeekObjectSkill,
    ApproachObjectSkill.name: ApproachObjectSkill,
    GoToPositionSkill.name: GoToPositionSkill,
    GoToObjectSkill.name: GoToObjectSkill,
    MapRoomSkill.name: MapRoomSkill,
    ReadGaugeSkill.name: ReadGaugeSkill,
    SequentialApproachSkill.name: SequentialApproachSkill,
    ExploreRoomSkill.name: ExploreRoomSkill,
    VlmReachSkill.name: VlmReachSkill,
    SpeakTextSkill.name: SpeakTextSkill,
}


# Skills the LLM planner is NOT allowed to pick. They stay registered in
# SKILL_CLASSES so:
#   * the block-programming canvas (Mode 3) can still drop them on a
#     program for manual ops,
#   * the host code can still instantiate them by name for debugging.
# But `planner_visible_skills()` filters them out of the planner prompt
# so the LLM never reaches for them. We removed them from the planner's
# surface because the wall-follower + line-follower behaviours have been
# observed to misfire in the current environment and they are not
# helpful for the inspection / approach tasks the planner actually has
# to solve. They will be reinstated once the underlying behaviour is
# fixed.
PLANNER_HIDDEN_SKILLS: set = {
    "wall_follower",
    "line_follower",
}


def planner_visible_skills() -> Dict[str, Type[BaseSkill]]:
    """Subset of SKILL_CLASSES that the LLM planner is allowed to choose
    from. Hides legacy / unreliable skills behind PLANNER_HIDDEN_SKILLS
    without removing them entirely from the codebase."""
    return {
        n: c for n, c in SKILL_CLASSES.items()
        if n not in PLANNER_HIDDEN_SKILLS
    }


# Parameter schema for each skill. `required` params have no defaults — the
# planner MUST supply them in `skill_command.parameters` or the skill returns
# failure immediately. `optional` params fall back to skills_config.yaml or
# hard-coded defaults inside the skill.
#
# Used by:
#   * skill_metadata() below — surfaces the schema to the LLM planner
#   * GUI block-programming pre-fill (future)
SKILL_PARAM_SCHEMAS: Dict[str, Dict[str, Dict[str, str]]] = {
    "wall_follower": {
        "required": {},
        "optional": {
            "wall_side": "'left' or 'right' (which side to track), default 'right'",
            "target_distance_m": "desired distance to wall in meters, default 0.25",
            "speed_mps": "forward speed, default 0.15",
        },
    },
    "line_follower": {
        "required": {},
        "optional": {
            "speed_mps": "forward speed, default 0.12",
            "line_color_hsv_lower": "[H,S,V] lower bound, default [0,0,0] (black)",
            "line_color_hsv_upper": "[H,S,V] upper bound, default [180,255,60] (black)",
        },
    },
    "return_to_base": {
        "required": {},
        "optional": {
            "position_tolerance_m": "margin of error — stop within this radius, default 0.15",
        },
    },
    "search_object": {
        "required": {
            "target_class": "COCO class name to look for — e.g. 'bottle', 'chair', "
                            "'person', 'cup', 'laptop', 'tv', 'clock', 'book'. "
                            "MUST be set or the skill fails immediately.",
        },
        "optional": {
            "rotation_speed_radps": "rotate speed, default 0.6",
            "step_angle_deg": "degrees per scan step, default 30",
            "max_full_rotations": "give up after N rotations, default 2",
            "confidence_threshold": "min YOLO confidence, default 0.5",
        },
    },
    "seek_object": {
        "required": {
            "target_class": "Class name to seek (any layer). Accepts prefixed "
                            "names: 'bottle', 'yw_gun', 'person'. "
                            "Use this skill when the VLM scene text mentions a "
                            "target but no detection box has appeared yet — "
                            "typically because the object is too far. The robot "
                            "drives forward (with optional bearing bias) until "
                            "any detection layer produces a matching box, then "
                            "hands off to approach_object.",
        },
        "optional": {
            "target_keywords": "list of additional substrings to match against "
                               "any detection class — useful when unsure which "
                               "prefixed form will appear, e.g. ['gun','pistol','rifle']",
            "bearing_hint": "'center' | 'left' | 'right' | 'slight_left' | "
                            "'slight_right' — small angular bias from the VLM's "
                            "spatial cue, default 'center'",
            "approach_speed_mps": "forward speed while seeking, default 0.12",
            "min_confidence": "detection confidence needed to declare acquired, default 0.30",
            "max_distance_m": "hard cap on travel distance, default 3.0",
            "max_duration_s": "hard cap on time, default 30",
            "min_obstacle_distance_m": "abort if depth shows obstacle closer than this, default 0.30",
        },
    },
    "approach_object": {
        "required": {
            "target_class": "COCO class name to approach — e.g. 'bottle', 'chair'. "
                            "MUST be set or the skill fails immediately.",
        },
        "optional": {
            "stop_distance_m": "halt at this distance from target, default 0.5",
            "approach_speed_mps": "forward speed when far from target, default 0.15",
            "slow_radius_m": "smoothstep deceleration zone ahead of stop, default 0.6",
            "min_approach_speed_mps": "creep speed at the stop band, default 0.04",
            "alignment_deadband_px": "pixel error below which angular_z=0, default 25",
            "min_confidence": "ignore detections below this YOLO confidence, default 0.35",
        },
    },
    "go_to_position": {
        "required": {
            "x": "world-frame X target in meters (odom frame)",
            "y": "world-frame Y target in meters (odom frame)",
        },
        "optional": {
            "tolerance_m": "margin of error — accept arrival within this radius, default 0.15",
            "max_legs": "abort after this many drive legs to prevent infinite hunting, default 12",
        },
    },
    "go_to_object": {
        "required": {
            "target_name": "class name of a remembered object — e.g. 'bottle', "
                           "'chair', 'person'. Looked up in the RAG object "
                           "memory; fails immediately if no instance has been "
                           "seen yet.",
        },
        "optional": {
            "stop_distance_m": "halt this far short of the object, default 0.5",
            "instance_index": "n-th most-confident instance, default 0",
            "face_target": "rotate to face the object at the end, default True",
            "tolerance_m": "margin of error for the navigation phase, default 0.15",
        },
    },
    "map_room": {
        "required": {},
        "optional": {
            "coverage_threshold_pct": "stop after this %% coverage, default 80",
            "spacing_m": "distance between sweep lines, default 0.4",
        },
    },
    "read_gauge": {
        "required": {},
        "optional": {
            "target_class": "YOLO class of the gauge to read, default 'gauge'",
            "approach_distance_m": "stop this far from the gauge before reading, default 0.6",
        },
    },
    "sequential_approach": {
        "required": {
            "target_sequence": "list of COCO class names to visit IN ORDER — e.g. "
                               "['chair','chair','person']. MUST be a non-empty list.",
        },
        "optional": {
            "stop_distance_m": "halt at this distance per target, default 0.5",
            "per_target_timeout_s": "per-target time budget, default 60",
        },
    },
    "explore_room": {
        "required": {
            "target_class": "name of the object the operator is looking for "
                            "(e.g. 'bottle', 'chair'). The skill maintains a "
                            "TARGET-BELIEF probability grid combining VLM "
                            "scene text, YOLO detections, the SLAM occupancy "
                            "grid, and the robot's trajectory; on each iter "
                            "it drives to the highest-belief reachable cell "
                            "and re-scans. Exits success as soon as a "
                            "matching detection appears, exits failure when "
                            "the belief is exhausted. Use this for MEDIUM "
                            "difficulty tasks where the target is occluded "
                            "or out of sight from the start pose.",
        },
        "optional": {
            "target_keywords": "list of substrings to also match (e.g. "
                               "['bottle','water bottle']), useful for "
                               "prefixed forms like yw_bottle.",
            "max_iterations": "give up after this many exploration legs, default 8",
            "max_duration_s": "wall-clock cap in seconds, default 120",
            "min_confidence": "YOLO confidence to declare acquired, default 0.30",
            "scan_after_arrival": "pivot ±60° after each leg to widen the FOV, default True",
        },
    },
    "vlm_reach": {
        "required": {
            "target_name": "object the VLM has spotted but YOLO hasn't. The "
                           "skill uses a GPT-4o-mini → GPT-4o visual-"
                           "grounding cascade to lock a precise image "
                           "point, deprojects it to a 3-D world position "
                           "via depth + intrinsics, picks a SAFE viewpoint "
                           "short of the target by snapping to a free "
                           "SLAM cell, then drives there. Use whenever "
                           "the VLM TASK line emits `VLM-sees-target` "
                           "and no matching YOLO box exists.",
        },
        "optional": {
            "stop_distance_m": "halt this far from the target along the "
                               "robot→target line, default 0.5",
            "max_iterations": "ground+drive cycles before giving up, default 4",
            "max_duration_s": "wall-clock cap, default 90",
            "success_confidence": "grounding confidence at close range to "
                                  "declare success, default 0.55",
            "viewpoint_snap_m": "max radius to search the free-space "
                                "grid for the nearest safe viewpoint, "
                                "default 0.40",
        },
    },
    "speak_text": {
        "required": {
            "text": "the text string to speak aloud — e.g. 'Hello, I am QBot.'",
        },
        "optional": {},
    },
}


def skill_metadata() -> Dict[str, Dict[str, Any]]:
    """Lightweight metadata for the GUI skill cards and the LLM planner.

    Includes parameter schemas so the planner knows exactly which params are
    required vs optional and what each one does — without this, the LLM picks
    a skill but emits empty `parameters`, causing skills like search_object to
    fail instantly (no target_class) and the AI loop to spin forever.
    """
    md: Dict[str, Dict[str, Any]] = {}
    for name, cls in SKILL_CLASSES.items():
        schema = SKILL_PARAM_SCHEMAS.get(name, {"required": {}, "optional": {}})
        md[name] = {
            "name": cls.name,
            "description": cls.description,
            "icon": cls.icon,
            "required_params": dict(schema.get("required", {})),
            "optional_params": dict(schema.get("optional", {})),
        }
    return md


__all__ = [
    "BaseSkill",
    "SkillResult",
    "SkillAborted",
    "SKILL_CLASSES",
    "SKILL_PARAM_SCHEMAS",
    "PLANNER_HIDDEN_SKILLS",
    "planner_visible_skills",
    "skill_metadata",
    "WallFollowerSkill",
    "LineFollowerSkill",
    "ReturnToBaseSkill",
    "SearchObjectSkill",
    "SeekObjectSkill",
    "ApproachObjectSkill",
    "GoToPositionSkill",
    "GoToObjectSkill",
    "MapRoomSkill",
    "ReadGaugeSkill",
    "SequentialApproachSkill",
    "ExploreRoomSkill",
    "VlmReachSkill",
    "SpeakTextSkill",
]
