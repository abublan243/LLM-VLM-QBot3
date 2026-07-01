"""
Performance Monitor — collects pipeline latency, FPS, and resource metrics.

Polls every `monitor_interval_ms`, emits a `metrics_updated(dict)` Qt signal so
the Performance widget can refresh without doing its own bookkeeping.

Tracked (as enumerated in the project spec):
    vlm_latency_ms        EMA over recent VLM API calls
    llm_latency_ms        EMA over recent LLM API calls
    motor_cmd_latency_ms  EMA from LLM-decision → ROS publish
    yolo_fps              counts in a 1 s sliding window
    camera_fps            counts of incoming RGB frames in 1 s
    websocket_rtt_ms      service-call ping to /rosapi/get_time
    battery_percent       direct from SharedState
    task_success_rate     completed/attempted on the running session
    tokens_used_session   running sum across all model calls
    cpu_percent           psutil
    memory_percent        psutil

Thread safety:
    All record_*() methods are safe to call from any thread (background
    workers, ROS callbacks, asyncio tasks). They lock the same RLock.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import RLock
from typing import Any, Deque, Dict, Optional

import psutil
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.shared_state import SharedState

logger = logging.getLogger(__name__)


def _ema(prev: float, sample: float, alpha: float = 0.3) -> float:
    """Exponential moving average (alpha = how much weight on the latest sample)."""
    if prev <= 0.0:
        return sample
    return alpha * sample + (1.0 - alpha) * prev


class PerformanceMonitor(QObject):
    """Aggregates pipeline / system metrics and emits them on a fixed cadence."""

    metrics_updated = pyqtSignal(dict)

    def __init__(
        self,
        state: Optional[SharedState] = None,
        *,
        interval_ms: int = 500,
        ros_bridge: Optional[Any] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state or SharedState.instance()
        self._lock = RLock()
        self._interval_ms = interval_ms
        self._ros_bridge = ros_bridge

        # ---- metrics ----
        self.vlm_latency_ms: float = 0.0
        self.llm_latency_ms: float = 0.0
        self.motor_cmd_latency_ms: float = 0.0
        self.websocket_rtt_ms: float = 0.0
        self.tokens_used_session: int = 0
        self.task_attempted: int = 0
        self.task_succeeded: int = 0

        # FPS sliding windows (timestamps of recent events)
        self._yolo_events: Deque[float] = deque(maxlen=120)
        self._camera_events: Deque[float] = deque(maxlen=120)
        self._fps_window_s = 1.0

        # psutil snapshot (cached so we don't hit the /proc tree more than needed)
        self._cpu_percent: float = 0.0
        self._memory_percent: float = 0.0
        psutil.cpu_percent(interval=None)  # prime the per-process counter

        # ---- timer ----
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def start(self) -> None:
        self._timer.start()
        logger.debug("PerformanceMonitor started @ %d ms", self._interval_ms)

    def stop(self) -> None:
        self._timer.stop()

    def attach_ros_bridge(self, ros_bridge: Any) -> None:
        """Late-bind the ROSBridge instance so we can ping it for RTT."""
        self._ros_bridge = ros_bridge

    # ---------------------------------------------------------------
    # record_* — call sites in the AI pipeline
    # ---------------------------------------------------------------

    def record_vlm_call(self, latency_ms: float, tokens: int = 0) -> None:
        with self._lock:
            self.vlm_latency_ms = _ema(self.vlm_latency_ms, max(0.0, latency_ms))
            self.tokens_used_session += max(0, tokens)

    def record_llm_call(self, latency_ms: float, tokens: int = 0) -> None:
        with self._lock:
            self.llm_latency_ms = _ema(self.llm_latency_ms, max(0.0, latency_ms))
            self.tokens_used_session += max(0, tokens)

    def record_motor_cmd(self, latency_ms: float) -> None:
        with self._lock:
            self.motor_cmd_latency_ms = _ema(self.motor_cmd_latency_ms, max(0.0, latency_ms))

    def record_yolo_inference(self) -> None:
        with self._lock:
            self._yolo_events.append(time.monotonic())

    def record_camera_frame(self) -> None:
        with self._lock:
            self._camera_events.append(time.monotonic())

    def record_task_outcome(self, success: bool) -> None:
        with self._lock:
            self.task_attempted += 1
            if success:
                self.task_succeeded += 1

    # ---------------------------------------------------------------
    # Internal — periodic emission
    # ---------------------------------------------------------------

    def _fps(self, deque_obj: Deque[float]) -> float:
        now = time.monotonic()
        cutoff = now - self._fps_window_s
        while deque_obj and deque_obj[0] < cutoff:
            deque_obj.popleft()
        return len(deque_obj) / self._fps_window_s

    def _measure_topic_age(self) -> None:
        """For native DDS we don't have a WebSocket RTT, so we report
        the age (ms since last receipt) of the IMU topic instead. Stored
        in the same `websocket_rtt_ms` field for back-compat with the
        performance panel; the field is now a generic 'connection latency'.
        """
        with self.state.lock:
            ts = self.state.imu.monotonic_ts
        if ts <= 0.0:
            return
        age_ms = (time.monotonic() - ts) * 1000.0
        with self._lock:
            self.websocket_rtt_ms = _ema(self.websocket_rtt_ms, age_ms)

    def _tick(self) -> None:
        with self._lock:
            yolo_fps = self._fps(self._yolo_events)
            camera_fps = self._fps(self._camera_events)

            # System metrics
            try:
                self._cpu_percent = psutil.cpu_percent(interval=None)
                self._memory_percent = psutil.virtual_memory().percent
            except Exception:
                pass

            attempted = self.task_attempted
            succeeded = self.task_succeeded
            success_rate = (succeeded / attempted) if attempted > 0 else 0.0

            payload: Dict[str, Any] = {
                "vlm_latency_ms": self.vlm_latency_ms,
                "llm_latency_ms": self.llm_latency_ms,
                "motor_cmd_latency_ms": self.motor_cmd_latency_ms,
                "yolo_fps": yolo_fps,
                "camera_fps": camera_fps,
                "websocket_rtt_ms": self.websocket_rtt_ms,
                "battery_percent": self.state.battery.percent,
                "task_attempted": attempted,
                "task_succeeded": succeeded,
                "task_success_rate": success_rate,
                "tokens_used_session": self.tokens_used_session,
                "cpu_percent": self._cpu_percent,
                "memory_percent": self._memory_percent,
            }

        self._measure_topic_age()
        self.metrics_updated.emit(payload)
