"""Non-blocking recording; recorder overload never blocks counting."""
import json
import queue
import threading
import time
from pathlib import Path

import cv2


class AsyncRecorder:
    def __init__(self, path, fps, codec="MJPG", queue_size=256):
        self.path = Path(path)
        self.fps = float(fps)
        self.codec = codec
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = None
        self.writer = None
        self.written = 0
        self.dropped = 0
        self.first_frame_no = None
        self.last_frame_no = None
        self.error = ""
        self.started_ns = 0

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_ns = time.time_ns()
        self.thread = threading.Thread(target=self._run, name="video-recorder", daemon=True)
        self.thread.start()

    def submit(self, packet):
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    packet = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                frame = packet.image
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if self.writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*self.codec)
                    self.writer = cv2.VideoWriter(
                        str(self.path), fourcc, self.fps, (w, h), True
                    )
                    if not self.writer.isOpened():
                        raise RuntimeError(f"无法创建录像文件: {self.path}")
                self.writer.write(frame)
                self.written += 1
                self.first_frame_no = (packet.frame_no if self.first_frame_no is None
                                       else self.first_frame_no)
                self.last_frame_no = packet.frame_no
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.writer is not None:
                self.writer.release()
                self.writer = None

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10.0)
        metadata = {
            "video": self.path.name,
            "fps": self.fps,
            "codec": self.codec,
            "written_frames": self.written,
            "queue_dropped_frames": self.dropped,
            "first_camera_frame_no": self.first_frame_no,
            "last_camera_frame_no": self.last_frame_no,
            "started_unix_ns": self.started_ns,
            "error": self.error,
        }
        meta_path = self.path.with_suffix(self.path.suffix + ".json")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                             encoding="utf-8")
