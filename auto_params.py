#!/usr/bin/env python3
"""Estimate initial ``count_cv.py`` parameters from an unlabeled video or frames.

This is deliberately an *initialiser*, not a replacement for a labelled
validation set.  It searches for a stable central background, estimates the
foreground intensity and object area from sampled frames, then estimates the
frame-to-frame tracking distance from a short consecutive run.
"""
import argparse
import json
import os

import cv2
import numpy as np

from count_cv import FrameSource


def largest_true_run(mask):
    """Return [start, end) for the largest contiguous True run."""
    best = start = 0
    best_pair = (0, len(mask))
    for i, value in enumerate(np.r_[mask, False]):
        if value and i == 0:
            start = 0
        elif value and not mask[i - 1]:
            start = i
        elif not value:
            if i - start > best:
                best, best_pair = i - start, (start, i)
    return best_pair


def estimate_roi(gray_frames):
    """Discard dark letterbox/border rows while retaining the largest bright band."""
    h, w = gray_frames[0].shape
    background = np.median(np.stack(gray_frames), axis=0)
    row_level = np.median(background, axis=1)
    lo, hi = np.percentile(row_level, [5, 95])
    # Keep rows that are materially closer to the normal scene than to a dark
    # border.  If that is not a useful band, fall back to the full frame.
    cutoff = lo + 0.30 * max(hi - lo, 1.0)
    y0, y1 = largest_true_run(row_level >= cutoff)
    if y1 - y0 < h * 0.35:
        y0, y1 = 0, h
    pad = max(2, int((y1 - y0) * 0.01))
    return (0, max(0, y0 - pad), w, min(h, y1 + pad)), background


def components(gray, threshold, roi, kernel):
    x0, y0, x1, y1 = roi
    mask = (gray[y0:y1, x0:x1] < threshold).astype(np.uint8) * 255
    if kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        out.append((float(cents[i, 0] + x0), float(cents[i, 1] + y0),
                    int(area), int(bw), int(bh)))
    return out


def estimate(source, samples=50, motion_frames=80):
    src = FrameSource(source)
    if src.n < 2:
        raise RuntimeError("至少需要两帧")
    idxs = np.linspace(0, src.n - 1, min(samples, src.n), dtype=int)
    frames = [src.sample(int(i)) for i in idxs]
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames if f is not None]
    if len(frames) < 2:
        raise RuntimeError("无法读取足够的样本帧")
    roi, background = estimate_roi(frames)
    x0, y0, x1, y1 = roi
    roi_pixels = (x1 - x0) * (y1 - y0)

    # Dark-object threshold candidates come from the lower brightness tail.
    values = np.concatenate([f[y0:y1, x0:x1].ravel() for f in frames])
    candidates = sorted({int(np.clip(np.percentile(values, q), 3, 250))
                         for q in (2, 3, 4, 5, 6, 8, 10, 12)})
    scored = []
    for threshold in candidates:
        areas = []
        for frame in frames:
            areas.extend(c[2] for c in components(frame, threshold, roi, 3)
                         if roi_pixels * 0.002 <= c[2] <= roi_pixels * 0.35)
        if len(areas) < max(3, len(frames) // 5):
            continue
        med = float(np.median(areas))
        # Prefer a threshold that yields similarly sized objects on many frames.
        cv = float(np.std(areas) / max(np.mean(areas), 1.0))
        scored.append((len(areas) / (1.0 + cv), threshold, med, areas))
    if not scored:
        raise RuntimeError("未能从画面自动分离出稳定目标；请改用带真值的 tune_params.py")
    # Several thresholds often describe the same objects.  Prefer the largest
    # threshold whose stability score remains close to the best one: this keeps
    # more of a dark object's boundary without admitting the background tail.
    best_score = max(v[0] for v in scored)
    _, threshold, median_area, areas = max(
        (v for v in scored if v[0] >= best_score * 0.75), key=lambda v: v[1])
    min_area = max(20, int(round(median_area * 0.28 / 100.0) * 100))

    # Read a consecutive run to measure apparent motion, avoiding jumps between
    # the evenly spaced calibration samples.
    run = min(motion_frames, src.n)
    start = max(0, (src.n - run) // 2)
    consecutive = []
    for i in range(start, start + run):
        frame = src.sample(i)
        if frame is not None:
            consecutive.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    src.release()
    dets = [[c for c in components(f, threshold, roi, 3) if c[2] >= min_area]
            for f in consecutive]
    dx, dy = [], []
    for previous, current in zip(dets, dets[1:]):
        for cx, cy, area, _, _ in current:
            candidates_prev = [(px, py) for px, py, pa, _, _ in previous
                               if 0.4 <= area / max(pa, 1) <= 2.5]
            if candidates_prev:
                px, py = min(candidates_prev, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                d = np.hypot(cx - px, cy - py)
                if d < max(20, np.sqrt(median_area) * 1.5):
                    dx.append(abs(cx - px)); dy.append(abs(cy - py))
    step = float(np.percentile(np.hypot(dx, dy), 95)) if dx else 8.0
    axis = "x" if np.median(dx or [0]) >= np.median(dy or [0]) else "y"
    return {
        "method": "thresh", "thresh_lo": threshold, "thresh_hi": 255,
        "roi": f"{x0},{y0},{x1},{y1}", "min_area": min_area,
        "morph_kernel": 3, "morph_iter": 1,
        "max_dist": int(max(20, np.ceil(step * 1.4 + 8))), "track_ttl": 6,
        "min_hits": 3, "axis": axis, "flow": "both", "line": 0.5,
        "line_band": 0.04,
    }, {"sampled_frames": len(frames), "object_area_median": round(median_area, 1),
          "motion_p95_px": round(step, 2), "candidate_thresholds": candidates}


def main():
    ap = argparse.ArgumentParser(description="无真值自动估计 count_cv 初始参数")
    ap.add_argument("source", help="视频文件或图像帧目录")
    ap.add_argument("--out", default="auto_params.json", help="输出 JSON 路径")
    ap.add_argument("--samples", type=int, default=50, help="均匀采样帧数")
    ap.add_argument("--motion-frames", type=int, default=80, help="连续帧运动估计长度")
    args = ap.parse_args()
    params, diagnostics = estimate(args.source, args.samples, args.motion_frames)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(params, fh, ensure_ascii=False, indent=2)
    print("自动估参（无真值初始值）:", json.dumps(params, ensure_ascii=False))
    print("诊断:", json.dumps(diagnostics, ensure_ascii=False))
    command = "python count_cv.py " + repr(args.source) + " " + " ".join(
        f"--{key.replace('_', '-')} {value}" for key, value in params.items())
    print("建议命令:", command)
    print("提示：请用 --save 导出的标注视频复核；有真值时请再用 tune_params.py 优化。")


if __name__ == "__main__":
    main()
