#!/usr/bin/env python3
"""Replay generated videos as non-seekable streams and verify live auto-adapt."""
import argparse
import json
from pathlib import Path

import cv2

import count_cv


class VideoAsLiveSource:
    """FrameSource-compatible adapter that deliberately cannot seek or rewind."""

    video = None
    fps_override = None

    def __init__(self, _source, fps=30.0):
        self.live = True
        self.is_dir = False
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开: {self.video}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.fps_override or self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.n = 0

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield frame

    def release(self):
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, default=Path("validation_scenarios_gray"),
                        nargs="?")
    parser.add_argument("--calibration-frames", type=int, default=48)
    args = parser.parse_args()
    rows = []
    original = count_cv.FrameSource
    try:
        count_cv.FrameSource = VideoAsLiveSource
        for video in sorted(args.directory.glob("*.mp4")):
            meta_path = video.with_name(video.stem + "_meta.json")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            VideoAsLiveSource.video = video
            VideoAsLiveSource.fps_override = float(meta["fps"])
            detected = count_cv.count_source("non-seekable-live", {
                "method": "thresh", "thresh_lo": 90, "thresh_hi": 150,
                "auto_adapt": True,
                "calibration_frames": args.calibration_frames,
                "axis": "x", "flow": "pos",
            })
            gt = int(meta["crossed_center_line"])
            error = detected - gt
            accuracy = max(0.0, 100.0 - 100.0 * abs(error) / gt) if gt else 100.0
            rows.append((video.stem, gt, detected, error, accuracy))
    finally:
        count_cv.FrameSource = original

    print("| 场景 | 真值 | 实时检测 | 误差 | 准确率 |")
    print("|---|---:|---:|---:|---:|")
    for name, gt, detected, error, accuracy in rows:
        print(f"| {name} | {gt} | {detected} | {error:+d} | {accuracy:.2f}% |")


if __name__ == "__main__":
    main()
