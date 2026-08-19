from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraDevice:
    id: str
    model: str
    serial: str
    transport: str

    @property
    def label(self):
        tail = f" / {self.serial}" if self.serial else ""
        return f"{self.model}{tail} ({self.transport})"


@dataclass
class FramePacket:
    image: np.ndarray
    frame_no: int
    captured_ns: int
    camera_timestamp: int = 0


@dataclass
class CountResult:
    frame: Optional[np.ndarray]
    frame_no: int
    count: int
    detections: int
    process_ms: float


@dataclass
class RuntimeStats:
    running: bool = False
    count: int = 0
    acquired: int = 0
    processed: int = 0
    camera_frame_gaps: int = 0
    processing_queue_drops: int = 0
    recording_queue_drops: int = 0
    processing_queue_depth: int = 0
    acquisition_fps: float = 0.0
    processing_fps: float = 0.0
    process_ms: float = 0.0
    full_bin: bool = False
    full_bin_target: int = 0
    counting_paused: bool = False
    io_output_active: bool = False
    io_error: str = ""
    last_frame_no: Optional[int] = None
    error: str = ""
    started_ns: int = 0
    extra: dict = field(default_factory=dict)
