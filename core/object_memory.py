"""
Object Memory — persistent world-frame map of detected objects.

Every YOLO detection with a valid ``position_3d`` (camera optical frame)
is transformed into the world (odom) frame and merged into a per-class
list of remembered objects. Each entry stores ``(name, x, y, z, theta)``
plus confidence and hit-count metadata so the planner can pick the
most-reliable instance when the operator says "go to the bottle".

This is the lightweight RAG layer the project asked for: no embeddings
yet, just a typed table keyed by COCO class name. The list is small
enough to load entirely into the planner's context every cycle.

Persisted to ``config/object_memory.json`` so a robot that has explored
a room remembers it after restart.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Optical-frame Detection.position_3d uses (X right, Y down, Z forward).
# Converting to base_link (X forward, Y left, Z up):
#     X_base =  Z_optical
#     Y_base = -X_optical
#     Z_base = -Y_optical
def _optical_to_base(x_opt: float, y_opt: float, z_opt: float
                     ) -> Tuple[float, float, float]:
    return (z_opt, -x_opt, -y_opt)


@dataclass
class ObjectMemoryEntry:
    """One remembered object instance in the world frame."""
    name: str
    x: float
    y: float
    z: float
    theta_rad: float                  # robot heading at the most-recent observation
    confidence: float = 0.0
    hits: int = 0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ObjectMemoryEntry":
        return cls(
            name=str(d.get("name", "")),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            theta_rad=float(d.get("theta_rad", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            hits=int(d.get("hits", 0)),
            first_seen_ts=float(d.get("first_seen_ts", 0.0)),
            last_seen_ts=float(d.get("last_seen_ts", 0.0)),
        )


class ObjectMemory:
    """Thread-safe per-class table of remembered objects.

    Merge policy: a new detection that lands within ``merge_radius_m`` of
    an existing same-class entry updates that entry with a confidence-
    weighted running average. Detections that land outside the merge
    radius become a new instance (so multiple chairs in the same room are
    tracked separately).
    """

    def __init__(
        self,
        *,
        store_path: Optional[Path] = None,
        merge_radius_m: float = 0.5,
        min_confidence: float = 0.4,
        max_instances_per_class: int = 12,
    ) -> None:
        self._lock = threading.RLock()
        self._by_class: Dict[str, List[ObjectMemoryEntry]] = {}
        self.store_path = Path(store_path) if store_path else None
        self.merge_radius_m = float(merge_radius_m)
        self.min_confidence = float(min_confidence)
        self.max_instances_per_class = int(max_instances_per_class)

    # ---------------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------------

    def update_from_detections(
        self,
        detections: Iterable,
        robot_pose: Tuple[float, float, float],
    ) -> int:
        """Fold every detection with a valid position_3d into memory.

        ``robot_pose`` is ``(x, y, yaw_rad)`` in the odom/world frame.
        Returns the number of entries created or updated.
        """
        rx, ry, ryaw = robot_pose
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        touched = 0
        ts = time.monotonic()

        with self._lock:
            for det in detections:
                pos = getattr(det, "position_3d", None)
                if pos is None:
                    continue
                conf = float(getattr(det, "confidence", 0.0))
                if conf < self.min_confidence:
                    continue
                # Optical → base_link → world
                xb, yb, zb = _optical_to_base(float(pos[0]),
                                              float(pos[1]),
                                              float(pos[2]))
                wx = rx + cos_y * xb - sin_y * yb
                wy = ry + sin_y * xb + cos_y * yb
                wz = zb
                self._merge_or_insert(
                    name=str(getattr(det, "class_name", "")).strip().lower(),
                    x=wx, y=wy, z=wz,
                    theta_rad=ryaw,
                    confidence=conf,
                    ts=ts,
                )
                touched += 1
        return touched

    def _merge_or_insert(
        self,
        *,
        name: str,
        x: float,
        y: float,
        z: float,
        theta_rad: float,
        confidence: float,
        ts: float,
    ) -> None:
        if not name:
            return
        bucket = self._by_class.setdefault(name, [])
        # Find nearest same-class entry within the merge radius
        nearest_idx = -1
        nearest_d2 = self.merge_radius_m ** 2
        for i, e in enumerate(bucket):
            dx = e.x - x
            dy = e.y - y
            d2 = dx * dx + dy * dy
            if d2 <= nearest_d2:
                nearest_d2 = d2
                nearest_idx = i

        if nearest_idx >= 0:
            e = bucket[nearest_idx]
            # Confidence-weighted running average. Old observations carry
            # weight ``e.confidence * e.hits``; the new one carries ``confidence``.
            w_old = max(0.01, e.confidence) * max(1, e.hits)
            w_new = max(0.01, confidence)
            w = w_old + w_new
            e.x = (e.x * w_old + x * w_new) / w
            e.y = (e.y * w_old + y * w_new) / w
            e.z = (e.z * w_old + z * w_new) / w
            e.theta_rad = theta_rad
            e.confidence = max(e.confidence, confidence)
            e.hits += 1
            e.last_seen_ts = ts
            return

        # New instance — append, then trim by lowest confidence if overflowed
        bucket.append(ObjectMemoryEntry(
            name=name, x=x, y=y, z=z, theta_rad=theta_rad,
            confidence=confidence, hits=1,
            first_seen_ts=ts, last_seen_ts=ts,
        ))
        if len(bucket) > self.max_instances_per_class:
            bucket.sort(key=lambda e: (e.confidence, e.last_seen_ts), reverse=True)
            del bucket[self.max_instances_per_class:]

    # ---------------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------------

    def find(self, name: str, *, instance: int = 0
             ) -> Optional[ObjectMemoryEntry]:
        """Return the n-th most-confident remembered instance of ``name``."""
        if not name:
            return None
        with self._lock:
            bucket = self._by_class.get(name.strip().lower(), [])
            if not bucket:
                return None
            ranked = sorted(
                bucket,
                key=lambda e: (e.confidence, e.last_seen_ts),
                reverse=True,
            )
            if instance < 0 or instance >= len(ranked):
                return None
            return ranked[instance]

    def find_nearest(self, name: str, x: float, y: float
                     ) -> Optional[ObjectMemoryEntry]:
        with self._lock:
            bucket = self._by_class.get(name.strip().lower(), [])
            if not bucket:
                return None
            return min(
                bucket,
                key=lambda e: (e.x - x) ** 2 + (e.y - y) ** 2,
            )

    def list_all(self) -> List[ObjectMemoryEntry]:
        with self._lock:
            out: List[ObjectMemoryEntry] = []
            for bucket in self._by_class.values():
                out.extend(bucket)
            return out

    def class_names(self) -> List[str]:
        with self._lock:
            return sorted(self._by_class.keys())

    def count(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._by_class.values())

    # ---------------------------------------------------------------
    # Removal
    # ---------------------------------------------------------------

    def forget(self, name: str) -> int:
        with self._lock:
            bucket = self._by_class.pop(name.strip().lower(), [])
            return len(bucket)

    def clear(self) -> None:
        with self._lock:
            self._by_class.clear()

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    def to_dict(self) -> Dict[str, List[Dict[str, float]]]:
        with self._lock:
            return {
                name: [e.to_dict() for e in bucket]
                for name, bucket in self._by_class.items()
            }

    def load_dict(self, payload: Dict[str, List[Dict[str, float]]]) -> None:
        with self._lock:
            self._by_class.clear()
            for name, bucket in (payload or {}).items():
                if not isinstance(bucket, list):
                    continue
                key = str(name).strip().lower()
                self._by_class[key] = [
                    ObjectMemoryEntry.from_dict(e) for e in bucket
                    if isinstance(e, dict)
                ]

    def save(self) -> bool:
        if self.store_path is None:
            return False
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.store_path.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as exc:
            logger.warning("ObjectMemory.save failed: %s", exc)
            return False

    def load(self) -> bool:
        if self.store_path is None or not self.store_path.exists():
            return False
        try:
            with self.store_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.load_dict(payload)
            logger.info("Loaded %d objects from %s", self.count(), self.store_path)
            return True
        except Exception as exc:
            logger.warning("ObjectMemory.load failed: %s", exc)
            return False

    # ---------------------------------------------------------------
    # Planner context formatting
    # ---------------------------------------------------------------

    def format_for_planner(self, *, max_entries: int = 16) -> str:
        """Compact human-readable summary the LLM can consume."""
        entries = self.list_all()
        if not entries:
            return "(no remembered objects)"
        entries.sort(key=lambda e: (e.confidence, e.last_seen_ts), reverse=True)
        lines: List[str] = []
        for e in entries[:max_entries]:
            lines.append(
                f"- {e.name}: x={e.x:+.2f} y={e.y:+.2f} z={e.z:+.2f} "
                f"yaw={math.degrees(e.theta_rad):+.0f}° "
                f"conf={e.confidence:.2f} hits={e.hits}"
            )
        if len(entries) > max_entries:
            lines.append(f"  ...and {len(entries) - max_entries} more")
        return "\n".join(lines)
