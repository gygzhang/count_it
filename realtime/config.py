import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from count_cv import DEFAULT_PARAMS


@dataclass
class CameraConfig:
    expected_model: str = "MV-CS004-10UM"
    expected_serial: str = "DA8557576"
    expected_transport: str = "USB3"
    buffer_nodes: int = 64
    processing_queue_size: int = 512
    read_timeout_ms: int = 100
    exposure_us: float = 500.0
    gain_db: float = 0.0


@dataclass
class RecordingConfig:
    queue_size: int = 256
    codec: str = "MJPG"
    nominal_fps: float = 525.5
    output_dir: str = "recordings"


@dataclass
class UiConfig:
    preview_fps: float = 30.0


@dataclass
class FullBinConfig:
    enabled: bool = True
    target_count: int = 100
    line_selector: str = "Line1"
    user_output_selector: str = "UserOutput0"
    active_high: bool = False


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    full_bin: FullBinConfig = field(default_factory=FullBinConfig)
    counting: Dict[str, Any] = field(default_factory=lambda: {
        **DEFAULT_PARAMS,
        "method": "otsu",
        "thresh_lo": 15,
        "thresh_hi": 255,
        "roi": "0,0,720,440",
        "min_area_frac": 0.013,
        "max_area_frac": 0.12,
        "morph_kernel": 5,
        "morph_iter": 1,
        "merge_dist": 40.0,
        "max_dist": 80.0,
        "track_ttl": 6,
        "min_hits": 3,
        "axis": "x",
        "flow": "both",
        "line": 0.5,
        "line_band": 0.04,
    })


def _merge_dataclass(obj, values):
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_config(path="config/realtime.json"):
    config = AppConfig()
    file = Path(path)
    if not file.exists() and getattr(sys, "frozen", False):
        external = Path(sys.executable).resolve().parent / path
        bundled = Path(getattr(sys, "_MEIPASS", "")) / path
        if external.exists():
            file = external
        elif bundled.exists():
            file = bundled
    if not file.exists():
        return config
    data = json.loads(file.read_text(encoding="utf-8"))
    _merge_dataclass(config.camera, data.get("camera", {}))
    _merge_dataclass(config.recording, data.get("recording", {}))
    _merge_dataclass(config.ui, data.get("ui", {}))
    _merge_dataclass(config.full_bin, data.get("full_bin", {}))
    config.counting.update(data.get("counting", {}))
    return config
