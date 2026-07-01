"""
TargetBeliefMap — 2D probability grid representing where the target
object is most likely to be, used by the new `explore_room` skill for
semantic active search.

The belief grid is co-registered with the SLAM log-odds grid: same
size, same resolution, same origin cell. That lets the explore skill
combine belief (where might it be) with occupancy (where can I
actually drive) cheaply via element-wise masking.

Information sources folded into the belief each iteration:

  1. VLM bearing hint (`update_vlm_bearing`)
     The VLM's TASK section yields a coarse spatial cue ("the target
     might be behind the chair on the left"). We project that into a
     forward cone from the robot's current pose and bump cells inside
     the cone. Strength of the bump scales with VLM confidence.

  2. YOLO negative evidence (`update_yolo_negative`)
     If the robot has just looked at some part of the room and YOLO
     found no matching detection, the target almost certainly isn't
     in the visible cone. We DECAY belief inside the camera FOV cone,
     out to max_range_m.

  3. Visited regions (`update_visited`)
     Cells the robot has been physically near are unlikely to still
     hide the target (we've already inspected them with the camera).
     Decay a disc around every recent trajectory sample.

  4. Frontier prior (`apply_frontier_prior`)
     Cells at the edge of known-free space (next to unknown) get a
     small positive prior: "the target could still be lurking in the
     unexplored region just past here". This is what gives the search
     direction even before the VLM has weighed in.

  5. Reachability mask (applied in `next_goal`)
     The chosen cell MUST be in known-free space AND have a clear
     straight-line ray from the robot pose. We can't drive into an
     occupied cell or through a wall.

`next_goal` returns the world (x, y) of the highest-scoring reachable
candidate, or None if no candidate is found.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Log-belief clip range — keeps any single update from saturating the
# grid for the next 20 iterations.
LOGB_MIN = -4.0
LOGB_MAX = 4.0


@dataclass
class BeliefGoal:
    """Return type from `next_goal()`."""

    world_xy: Tuple[float, float]
    cell_rc: Tuple[int, int]
    belief: float
    distance_m: float
    score: float


class TargetBeliefMap:
    """Probability-of-target log-grid co-registered with the SLAM grid."""

    def __init__(
        self,
        *,
        size_cells: int,
        resolution_m: float,
        origin_cell: Tuple[int, int],
    ) -> None:
        self.size = int(size_cells)
        self.resolution = float(resolution_m)
        self._origin_cell = (int(origin_cell[0]), int(origin_cell[1]))
        self.log_belief = np.zeros((self.size, self.size), dtype=np.float32)
        # Track which cells the robot has physically been near. This is
        # what makes the belief monotonically decay over the visited
        # corridor — once we've walked past the right hand wall, it
        # stops being a candidate goal.
        self.visited_mask = np.zeros((self.size, self.size), dtype=bool)
        # Last applied frontier prior (so we can subtract the previous
        # prior before applying the new one — frontiers move every cycle
        # as the robot explores).
        self._frontier_prior_buf = np.zeros((self.size, self.size),
                                            dtype=np.float32)

    # ---------------------------------------------------------------
    # Public mutation API
    # ---------------------------------------------------------------

    def reset(self) -> None:
        self.log_belief.fill(0.0)
        self.visited_mask.fill(False)
        self._frontier_prior_buf.fill(0.0)

    def update_vlm_bearing(
        self,
        robot_xy: Tuple[float, float],
        robot_yaw_rad: float,
        bearing_label: str,
        *,
        distance_band: str = "mid",
        confidence: float = 0.6,
        cone_half_rad: float = math.radians(35.0),
        max_range_m: float = 4.0,
    ) -> None:
        """Bump cells in a directional cone implied by a coarse VLM
        spatial cue. `bearing_label` is one of the strings the prompt
        constrains the VLM to: center / left / right / slight_left /
        slight_right. `distance_band` is near / mid / far.
        """
        # Bearing offsets (degrees) match the planner-prompt vocabulary.
        bearings = {
            "center": 0.0,
            "slight_left": 20.0,
            "left": 60.0,
            "slight_right": -20.0,
            "right": -60.0,
        }
        deg = bearings.get(bearing_label.strip().lower(), 0.0)
        heading = robot_yaw_rad + math.radians(deg)
        # Distance band controls where in the cone we boost most.
        bands = {"near": 0.8, "mid": 1.6, "far": 3.0}
        focus_dist = bands.get(distance_band.strip().lower(), 1.6)
        weight = max(0.2, min(3.5, 2.4 * float(confidence)))
        self._bump_cone(
            robot_xy, heading,
            cone_half_rad=cone_half_rad,
            max_range_m=max_range_m,
            focus_distance_m=focus_dist,
            delta=weight,
        )

    def update_yolo_negative(
        self,
        robot_xy: Tuple[float, float],
        robot_yaw_rad: float,
        *,
        fov_half_rad: float,
        max_range_m: float = 4.0,
        weight: float = -0.6,
    ) -> None:
        """Decay belief inside the camera FOV cone when YOLO found
        nothing matching. The robot looked here and the target isn't
        there."""
        self._bump_cone(
            robot_xy, robot_yaw_rad,
            cone_half_rad=fov_half_rad,
            max_range_m=max_range_m,
            focus_distance_m=max_range_m * 0.5,
            delta=weight,
        )

    def update_visited(
        self,
        robot_xy: Tuple[float, float],
        *,
        radius_m: float = 0.4,
        weight: float = -1.2,
    ) -> None:
        """Mark a disc around the robot as visited and decay belief
        there. We've physically been in this spot — even if the camera
        cone didn't cover it, anything within radius_m is unlikely to
        still be hiding the target."""
        radius_cells = max(1, int(round(radius_m / self.resolution)))
        cr, cc = self._world_to_cell(*robot_xy)
        r0 = max(0, cr - radius_cells)
        r1 = min(self.size, cr + radius_cells + 1)
        c0 = max(0, cc - radius_cells)
        c1 = min(self.size, cc + radius_cells + 1)
        if r1 <= r0 or c1 <= c0:
            return
        ys, xs = np.ogrid[r0:r1, c0:c1]
        mask = (ys - cr) ** 2 + (xs - cc) ** 2 <= radius_cells ** 2
        self.log_belief[r0:r1, c0:c1][mask] += weight
        self.visited_mask[r0:r1, c0:c1][mask] = True
        np.clip(self.log_belief, LOGB_MIN, LOGB_MAX, out=self.log_belief)

    def apply_frontier_prior(
        self,
        frontier_cells: np.ndarray,
        *,
        weight: float = 0.4,
    ) -> None:
        """Boost the log-belief at frontier cells (free cells touching
        unknown cells). Subtract whatever frontier prior we applied last
        cycle so the boost doesn't accumulate."""
        # Roll back the previous prior in one shot.
        self.log_belief -= self._frontier_prior_buf
        self._frontier_prior_buf.fill(0.0)
        if frontier_cells.size > 0:
            rows = frontier_cells[:, 0]
            cols = frontier_cells[:, 1]
            self._frontier_prior_buf[rows, cols] = float(weight)
            self.log_belief += self._frontier_prior_buf
        np.clip(self.log_belief, LOGB_MIN, LOGB_MAX, out=self.log_belief)

    # ---------------------------------------------------------------
    # Goal selection
    # ---------------------------------------------------------------

    def next_goal(
        self,
        robot_xy: Tuple[float, float],
        free_mask: np.ndarray,
        frontier_cells: np.ndarray,
        *,
        prefer_frontiers: bool = True,
        min_goal_distance_m: float = 0.4,
        max_goal_distance_m: float = 5.0,
    ) -> Optional[BeliefGoal]:
        """Pick the best reachable cell to drive to next.

        `free_mask` is True where the SLAM grid says the cell is
        known-free (so it's actually drivable). `frontier_cells` is
        used as the candidate set when `prefer_frontiers=True`,
        otherwise we score every free cell with non-negative belief.
        """
        if free_mask is None or free_mask.size == 0:
            return None

        # Candidate set
        if prefer_frontiers and frontier_cells.size > 0:
            rows = frontier_cells[:, 0]
            cols = frontier_cells[:, 1]
            # frontiers are by definition in free space, but if anyone
            # passes us a stale set we filter again.
            keep = free_mask[rows, cols]
            rows = rows[keep]
            cols = cols[keep]
            if rows.size == 0:
                # Fall back to any free cell.
                return self.next_goal(
                    robot_xy, free_mask, np.zeros((0, 2), dtype=np.int32),
                    prefer_frontiers=False,
                    min_goal_distance_m=min_goal_distance_m,
                    max_goal_distance_m=max_goal_distance_m,
                )
        else:
            ys, xs = np.where(free_mask)
            if ys.size == 0:
                return None
            rows = ys.astype(np.int32)
            cols = xs.astype(np.int32)

        # Distance from robot in cells
        cr, cc = self._world_to_cell(*robot_xy)
        dr = rows - cr
        dc = cols - cc
        d_cells = np.sqrt(dr * dr + dc * dc)
        d_m = d_cells * self.resolution

        # Distance filter
        in_band = (d_m >= min_goal_distance_m) & (d_m <= max_goal_distance_m)
        if not in_band.any():
            # Loosen the upper band if nothing qualifies — better to
            # over-shoot than to return nothing.
            in_band = d_m >= min_goal_distance_m
            if not in_band.any():
                return None
        rows = rows[in_band]
        cols = cols[in_band]
        d_m = d_m[in_band]

        belief = self.log_belief[rows, cols]
        # Score: belief - distance_penalty. Distance penalty is gentle
        # so the planner doesn't lock onto the nearest candidate when a
        # much higher-belief one is 1 m further away.
        score = belief - 0.30 * d_m
        idx = int(np.argmax(score))
        best_r = int(rows[idx])
        best_c = int(cols[idx])
        wx, wy = self._cell_to_world(best_r, best_c)
        return BeliefGoal(
            world_xy=(wx, wy),
            cell_rc=(best_r, best_c),
            belief=float(belief[idx]),
            distance_m=float(d_m[idx]),
            score=float(score[idx]),
        )

    # ---------------------------------------------------------------
    # Heatmap (for the GUI overlay / mission report)
    # ---------------------------------------------------------------

    def get_heatmap(self) -> np.ndarray:
        """Return the (H, W) log-belief grid as a float32 array. The
        SLAM viewer can blend this on top of the occupancy render."""
        return self.log_belief.copy()

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cr0, cc0 = self._origin_cell
        cc = int(round(cc0 + x / self.resolution))
        cr = int(round(cr0 + y / self.resolution))
        cc = max(0, min(self.size - 1, cc))
        cr = max(0, min(self.size - 1, cr))
        return cr, cc

    def _cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        cr0, cc0 = self._origin_cell
        x = (col - cc0) * self.resolution
        y = (row - cr0) * self.resolution
        return (x, y)

    def _bump_cone(
        self,
        robot_xy: Tuple[float, float],
        heading_rad: float,
        *,
        cone_half_rad: float,
        max_range_m: float,
        focus_distance_m: float,
        delta: float,
    ) -> None:
        """Apply `delta` to every cell inside a directional cone from
        the robot, weighted by a gaussian peaked at focus_distance_m
        along the cone axis. Vectorised over the bounding box."""
        if abs(delta) < 1e-6 or max_range_m <= 0:
            return
        cr0, cc0 = self._origin_cell
        rx, ry = robot_xy
        # Bounding box of the cone (cheap conservative one — the
        # circumscribed square at max_range).
        rcells = max(1, int(round(max_range_m / self.resolution)))
        cr, cc = self._world_to_cell(rx, ry)
        r0 = max(0, cr - rcells)
        r1 = min(self.size, cr + rcells + 1)
        c0 = max(0, cc - rcells)
        c1 = min(self.size, cc + rcells + 1)
        if r1 <= r0 or c1 <= c0:
            return
        ys, xs = np.ogrid[r0:r1, c0:c1]
        # Translate to world frame
        wx = (xs - cc0) * self.resolution
        wy = (ys - cr0) * self.resolution
        dx = wx - rx
        dy = wy - ry
        dist = np.sqrt(dx * dx + dy * dy)
        bearing = np.arctan2(dy, dx)
        dtheta = ((bearing - heading_rad + math.pi) % (2 * math.pi)) - math.pi
        in_cone = (dist > 0.05) & (dist <= max_range_m) & (np.abs(dtheta) <= cone_half_rad)
        # Gaussian falloff along the range axis around the focus distance
        sigma = max(0.4, max_range_m / 3.0)
        weight = np.exp(-((dist - focus_distance_m) ** 2) / (2.0 * sigma * sigma))
        contrib = (in_cone.astype(np.float32) * weight.astype(np.float32) * float(delta))
        self.log_belief[r0:r1, c0:c1] += contrib
        np.clip(self.log_belief, LOGB_MIN, LOGB_MAX, out=self.log_belief)
