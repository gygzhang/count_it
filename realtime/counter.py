"""Stateful streaming wrapper around the existing Detector and Tracker."""
import time

import cv2

from count_cv import DEFAULT_PARAMS, Detector, Track, Tracker, is_warming, scaled

from .types import CountResult


class StreamingCounter:
    def __init__(self, params=None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.detector = None
        self.tracker = None
        self.method = self.params["method"]
        self.width = 0
        self.height = 0
        self.index = 0

    def reset(self, frame):
        if frame.ndim == 2:
            h0, w0 = frame.shape
        else:
            h0, w0 = frame.shape[:2]
        scale = self.params["scale"]
        self.width, self.height = int(w0 * scale), int(h0 * scale)
        if self.method in ("auto", "refbg"):
            raise ValueError(
                "实时模式请明确使用 thresh/color/bgsub；refbg 请先扩展为固定背景图模式"
            )
        Track._next_id = 0
        self.detector = Detector(
            self.params, self.method, self.width, self.height, ref_gray=None
        )
        self.tracker = Tracker(self.params, self.width, self.height)
        self.index = 0

    def clear_count(self):
        """Clear total and active tracks while retaining detector/background state."""
        if self.tracker is not None:
            Track._next_id = 0
            self.tracker = Tracker(self.params, self.width, self.height)

    @staticmethod
    def _bgr(frame):
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def process(self, frame, frame_no=0, annotate=False):
        if self.detector is None:
            self.reset(frame)
        work = scaled(frame, self.params["scale"], self.width, self.height)
        started = time.perf_counter()
        detections = self.detector.detect(work)
        warming = is_warming(self.method, self.index, self.params["warmup"])
        self.tracker.update(detections, warming)
        process_ms = (time.perf_counter() - started) * 1000.0
        rendered = self.render(work, detections) if annotate else None
        self.index += 1
        return CountResult(
            frame=rendered,
            frame_no=frame_no,
            count=self.tracker.count,
            detections=len(detections),
            process_ms=process_ms,
        )

    def render(self, frame, detections):
        vis = self._bgr(frame).copy()
        tracker = self.tracker
        if tracker.axis == "x":
            cv2.line(vis, (tracker.line_pos, 0),
                     (tracker.line_pos, self.height), (0, 0, 255), 2)
        else:
            cv2.line(vis, (0, tracker.line_pos),
                     (self.width, tracker.line_pos), (0, 0, 255), 2)
        for _, _, x, y, width, height in detections:
            cv2.rectangle(vis, (int(x), int(y)),
                          (int(x + width), int(y + height)), (0, 255, 0), 2)
        for track in tracker.tracks:
            center = (int(track.cx), int(track.cy))
            if track.counted:
                cv2.circle(vis, center, 6, (0, 255, 255), -1)
            cv2.putText(vis, str(track.id), (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis, f"count: {tracker.count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        return vis

    @property
    def count(self):
        return 0 if self.tracker is None else self.tracker.count
