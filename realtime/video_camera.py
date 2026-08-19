"""Video-file camera used for UI development and hardware-free testing."""
import time

import cv2

from .camera_base import CameraSource
from .types import CameraDevice, FramePacket


class VideoCamera(CameraSource):
    def __init__(self, path, loop=True, realtime=True):
        self.path = str(path)
        self.loop = loop
        self.realtime = realtime
        self.cap = None
        self.running = False
        self.frame_no = 0
        self.fps = 30.0
        self._next_ns = 0

    def enumerate_devices(self):
        return [CameraDevice("video", "视频模拟相机", self.path, "FILE")]

    def open(self, device_id="video"):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开模拟视频: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_no = 0

    def start(self):
        if self.cap is None:
            self.open()
        self.running = True
        self._next_ns = time.perf_counter_ns()

    def read(self, timeout_ms=100):
        if not self.running:
            return None
        if self.realtime and self.fps > 0:
            self._next_ns += int(1e9 / self.fps)
            delay = self._next_ns - time.perf_counter_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_no = 0
            ok, frame = self.cap.read()
        if not ok:
            self.running = False
            return None
        packet = FramePacket(frame, self.frame_no, time.perf_counter_ns())
        self.frame_no += 1
        return packet

    def stop(self):
        self.running = False

    def close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
