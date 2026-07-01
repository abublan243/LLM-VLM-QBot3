#!/usr/bin/env python3
"""
QBot3 Control — application entry point.

Boot sequence:
    1. Load config (settings.json + skills_config.yaml)
    2. Configure logging
    3. Create QApplication + install qasync event loop
    4. Instantiate all core / AI / mode objects
    5. Build and show MainWindow
    6. Connect to ROS bridge (or start synthetic sensor generator)
    7. Run the event loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
from typing import Any, Dict

import yaml

# ── must be early: set PYTHONPATH so relative imports resolve ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _configure_logging(cfg: Dict[str, Any]) -> None:
    level_str = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_file = cfg.get("file", "logs/qbot3_control.log")
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handlers: list = [logging.StreamHandler(sys.stdout)]
    try:
        from logging.handlers import RotatingFileHandler
        max_bytes = int(cfg.get("max_file_size_mb", 10)) * 1024 * 1024
        backup = int(cfg.get("backup_count", 5))
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup)
        handlers.append(fh)
    except Exception:
        pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)


def main() -> None:
    # ── 1. Load config ──
    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.json")
    skills_path = os.path.join(PROJECT_ROOT, "config", "skills_config.yaml")

    settings = _load_json(settings_path)
    skills_config = _load_yaml(skills_path)

    # ── 2. Logging ──
    _configure_logging(settings.get("logging", {}))
    logger.info("QBot3 Control starting")

    # ── 3. Qt + qasync ──
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    import qasync
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    from gui.theme import apply_theme
    apply_theme(app)

    # ── 4. Core objects ──
    from core.shared_state import SharedState
    state = SharedState.instance()

    from core.performance_monitor import PerformanceMonitor
    perf = PerformanceMonitor(state, interval_ms=settings.get("performance", {}).get("monitor_interval_ms", 500))

    robot_cfg = settings.get("robot", {})
    topics_cfg = settings.get("topics", {})

    # Native DDS — set RMW + DOMAIN before rclpy.init()
    rmw = robot_cfg.get("rmw_implementation", "rmw_cyclonedds_cpp")
    domain_id = robot_cfg.get("ros_domain_id", 0)
    if rmw and not os.environ.get("RMW_IMPLEMENTATION"):
        os.environ["RMW_IMPLEMENTATION"] = rmw
    if not os.environ.get("ROS_DOMAIN_ID"):
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)

    from core.ros_bridge import ROSBridge
    ros = ROSBridge(
        topics=topics_cfg,
        state=state,
        domain_id=domain_id,
        rmw_implementation=rmw,
        node_name=robot_cfg.get("node_name", "qbot3_host"),
    )
    perf.attach_ros_bridge(ros)

    from core.sensor_processor import start_latch_decay_thread
    latch_stop = threading.Event()
    start_latch_decay_thread(state, stop_event=latch_stop)

    # ── Object memory (RAG): SESSION-SCOPED world-frame object list ──
    # Starts EMPTY every launch (no load()): memory is just an opportunistic
    # aid for things the robot happens to see while driving/searching this
    # run, not a persistent map. Reset on app start by design.
    from core.object_memory import ObjectMemory
    _cal_cfg = settings.get("calibration", {}) or {}
    object_memory = ObjectMemory(
        merge_radius_m=float(_cal_cfg.get("object_merge_radius_m", 0.5)),
    )
    state.object_memory = object_memory

    from ai.slam_manager import SlamManager
    cal = settings.get("calibration", {})
    slam = SlamManager(
        state, ros_bridge=ros,
        camera_height_m=cal.get("camera_height_m", 0.10),
        camera_pitch_deg=cal.get("camera_pitch_deg", 0.0),
        obstacle_min_height_m=cal.get("obstacle_min_height_m", 0.05),
        obstacle_max_height_m=cal.get("obstacle_max_height_m", 1.50),
    )
    slam.start()
    # Round 20: skills need access to the SLAM grid (bumper-hit marking,
    # explore_room frontier queries, vlm_reach reachability checks). The
    # cleanest path is to publish a reference on SharedState the same way
    # object_memory was attached.
    state.slam_manager = slam

    # ── 5. AI objects ──
    api_keys = settings.get("api_keys", {})
    ai_defaults = settings.get("ai_defaults", {})

    # ── YOLO-World open-vocabulary detector (second layer) ──
    # Accepts arbitrary text prompts as class names — perfect for the
    # cases where COCO doesn't have the class ("gun", "bottle", "helmet")
    # or misses things at distance. Weights auto-download from the
    # Ultralytics hub on first use.
    yw_cfg = settings.get("yolo_world", {}) or {}
    yw_weights = yw_cfg.get("weights", "yolov8s-world.pt")
    if (not os.path.isabs(yw_weights) and "/" in yw_weights):
        yw_weights = os.path.join(PROJECT_ROOT, yw_weights)
    from ai.yolo_world_detector import YoloWorldDetector
    yolo_world_detector = YoloWorldDetector(
        weights=yw_weights,
        confidence=float(yw_cfg.get("confidence", 0.20)),
        class_prefix=str(yw_cfg.get("class_prefix", "yw_")),
        prompt_classes=yw_cfg.get("classes"),
        enabled=bool(yw_cfg.get("enabled", True)),
    )

    from ai.vlm_pipeline import VLMPipeline
    vlm = VLMPipeline(
        state,
        weights=ai_defaults.get("yolo_model", "yolo11l.pt"),
        confidence=ai_defaults.get("yolo_confidence", 0.45),
        openai_key=api_keys.get("openai", ""),
        google_key=api_keys.get("google", ""),
        max_tokens=ai_defaults.get("vlm_max_tokens", 1024),
        performance_monitor=perf,
        yolo_world_detector=yolo_world_detector,
    )
    # Honour a persisted COCO-layer toggle from a previous session.
    # YOLO-World reads its .enabled from its own config block inside
    # YoloWorldDetector init.
    vlm.coco_enabled = bool(ai_defaults.get("coco_enabled", True))

    from ai.llm_planner import LLMPlanner
    llm = LLMPlanner(
        state,
        openai_key=api_keys.get("openai", ""),
        google_key=api_keys.get("google", ""),
        temperature=ai_defaults.get("llm_temperature", 0.2),
        max_tokens=ai_defaults.get("llm_max_tokens", 1024),
        performance_monitor=perf,
    )

    # ── 6. Modes ──
    safety_cfg = settings.get("safety", {})

    from modes.mode_manual import ManualMode
    mode_manual = ManualMode(
        state, ros,
        max_linear=safety_cfg.get("max_linear_speed", 0.3),
        max_angular=safety_cfg.get("max_angular_speed", 1.5),
    )

    # ── VLM visual ROI grounding cascade (Round 20) ──
    # GPT-4o-mini → GPT-4o, strict JSON output. Used by the new
    # `vlm_reach` skill when the VLM perceives the target but no YOLO
    # box exists for it. Self-disables gracefully when the OpenAI key
    # is empty.
    from ai.vlm_grounding import VlmGrounding
    vlm_grounding = VlmGrounding(
        openai_key=api_keys.get("openai", ""),
        mini_confidence_floor=float(
            ai_defaults.get("grounding_mini_confidence_floor", 0.55)
        ),
    )

    from modes.mode_skills import SkillsMode
    mode_skills = SkillsMode(
        state, ros,
        skills_config=skills_config,
        vlm_pipeline=vlm,
        vlm_model_name=ai_defaults.get("vlm_model", "gpt-4o"),
        vlm_grounding=vlm_grounding,
    )

    # ── 6b. Voice I/O (optional — loads in disabled state if sounddevice
    # or the openai SDK isn't installed, or if no API key is set) ──
    voice_cfg = settings.get("voice", {})
    from core.voice_io import VoiceIO
    voice = VoiceIO(
        openai_key=api_keys.get("openai", ""),
        stt_model=voice_cfg.get("stt_model", "whisper-1"),
        tts_model=voice_cfg.get("tts_model", "tts-1"),
        tts_voice=voice_cfg.get("tts_voice", "nova"),
        language=voice_cfg.get("language") or None,
    )

    # Inject voice_io into SkillsMode now that VoiceIO is ready
    mode_skills._voice_io = voice

    from modes.mode_ai import ModeAI
    reports_cfg = settings.get("reports", {}) or {}
    mode_ai = ModeAI(
        state, ros, vlm, llm,
        skills_config=skills_config,
        performance_monitor=perf,
        slam_manager=slam,
        llm_model_name=ai_defaults.get("llm_model", "gpt-4o-mini"),
        vlm_model_name=ai_defaults.get("vlm_model", "gpt-4o"),
        execution_type=ai_defaults.get("execution_type", "high_level"),
        voice_io=voice,
        speak_responses=bool(voice_cfg.get("speak_responses", False)),
        speak_section=str(voice_cfg.get("speak_section", "next_observation")),
        max_speak_chars=int(voice_cfg.get("max_speak_chars", 240)),
        generate_reports=bool(reports_cfg.get("auto_generate", True)),
        vlm_grounding=vlm_grounding,
    )

    # ── 7. GUI ──
    from gui.main_window import MainWindow
    window = MainWindow(
        state, ros, perf, slam, vlm, llm,
        mode_manual, mode_skills, mode_ai,
        settings=settings,
        skills_config=skills_config,
        voice_io=voice,
    )
    window.show()

    perf.start()

    # ── 8. ROS bridge or simulation ──
    sim_cfg = settings.get("simulation", {})

    def _on_connection_changed(connected: bool, _msg: str) -> None:
        if not connected and sim_cfg.get("enable_when_disconnected", True):
            _start_simulation()

    _sim_gen = None

    def _start_simulation() -> None:
        nonlocal _sim_gen
        if _sim_gen is not None:
            return
        logger.info("Starting synthetic sensor generator (simulation mode)")
        from core.sensor_processor import SyntheticSensorGenerator
        _sim_gen = SyntheticSensorGenerator(
            state,
            camera_fps=sim_cfg.get("synthetic_camera_fps", 15),
            imu_hz=sim_cfg.get("synthetic_imu_hz", 50),
        )
        _sim_gen.start()

    ros.connection_changed.connect(_on_connection_changed)

    try:
        ros.start()
    except Exception as exc:
        logger.warning("Initial ROS connection failed: %s — starting simulation", exc)
        if sim_cfg.get("enable_when_disconnected", True):
            _start_simulation()

    # Start YOLO warm-up + continuous detection so the Vision tab + SLAM
    # marker overlay always have fresh detections, regardless of mode.
    loop.create_task(vlm.ensure_yolo_loaded())
    detection_hz = settings.get("ai_defaults", {}).get("yolo_continuous_hz", 3.0)
    loop.create_task(vlm.start_continuous_detection(hz=detection_hz))

    # ── 9. Run ──
    # Graceful shutdown on Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    with loop:
        loop.run_forever()

    # ── 10. Cleanup ──
    logger.info("Shutting down…")
    latch_stop.set()
    if _sim_gen is not None:
        _sim_gen.stop()
    ros.stop()
    slam.stop()
    perf.stop()
    # Object memory is session-scoped — nothing to persist; it is discarded
    # on exit and starts empty on the next launch by design.
    logger.info("QBot3 Control stopped")


if __name__ == "__main__":
    main()
