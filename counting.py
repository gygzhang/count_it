#!/usr/bin/env python3
"""Detection, tracking, and counting pipeline."""
import glob
import json
import math
import os
import time

import cv2
import numpy as np

from params import merge_params, parse_roi, validate_params
from sources import FrameSource, scaled


class Track:
    def __init__(self, track_id, cx, cy):
        self.id = track_id
        self.cx, self.cy = cx, cy
        self.vx, self.vy = 0.0, 0.0
        self.prev_main = cx
        self.min_main = float("inf")
        self.max_main = float("-inf")
        self.consecutive_hits = 1
        self.missing = 0
        self.counted = False
        self.matched = False


def choose_method_frames(frames):
    """从内存帧列表判定 color/bgsub(高饱和像素占比)。"""
    frac = 0.0
    idxs = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
    for i in idxs:
        sat = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)[:, :, 1]
        frac = max(frac, float((sat > 80).mean()))
    return ("color" if frac > 0.003 else "bgsub"), frac


def make_ref_gray(frames):
    """取若干帧的中位数作为参考背景(灰度 int16)。"""
    idxs = np.linspace(0, len(frames) - 1, min(15, len(frames)), dtype=int)
    stack = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in idxs]
    return np.median(stack, axis=0).astype(np.int16)


def load_or_make_ref(P, sample_frames, w, h):
    """refbg 参考图：优先读 bg_ref 文件(缩放到 w×h)，否则用样例帧中位数。"""
    ref_path = P.get("bg_ref")
    if ref_path and ref_path != "auto" and os.path.exists(ref_path):
        img = cv2.imread(ref_path)
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return make_ref_gray(sample_frames)


def prepare_method(P, sample_frames, w, h):
    """统一的分离方式解析：auto->color/bgsub；refbg 备好参考背景。

    sample_frames: 已按 scale 处理过的若干采样帧(尺寸应为 w×h)。
    返回 (method, ref_gray_or_None)。
    """
    method = P["method"]
    if method == "auto":
        method, _ = choose_method_frames(sample_frames)
    ref = load_or_make_ref(P, sample_frames, w, h) if method == "refbg" else None
    return method, ref


class Detector:
    """一帧 -> 检测点列表 [(cx,cy,x,y,w,h), ...]。bgsub 有状态，须按帧序调用。"""

    def __init__(self, P, method, w, h, ref_gray=None):
        self.P = P
        self.method = method
        self.ref = ref_gray.copy() if ref_gray is not None else None
        self.axis = P["axis"]
        self.roi = parse_roi(P["roi"], w, h)
        # ROI 先裁后算:参考帧同步裁到 ROI，前景/形态学只在小区域跑
        if self.roi is not None and self.ref is not None:
            x0, y0, x1, y1 = self.roi
            self.ref = self.ref[y0:y1, x0:x1].copy()
        # 面积阈值:优先用"占画面比例"(分辨率/缩放无关),否则用绝对像素
        area = w * h
        self.min_area = (P["min_area_frac"] * area
                         if P.get("min_area_frac", 0) > 0 else P["min_area"])
        self.max_area = (P["max_area_frac"] * area
                         if P.get("max_area_frac", 0) > 0 else P["max_area"])
        self.bg = (cv2.createBackgroundSubtractorMOG2(
            history=P["bg_history"], varThreshold=P["bg_var"],
            detectShadows=False) if method == "bgsub" else None)
        k = int(P["morph_kernel"]) | 1
        # 矩形核:OpenCV 可分离优化,比椭圆快约 3 倍,去噪/补洞效果基本一致
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def _foreground(self, frame):
        if self.method == "color":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv[:, :, 1], self.P["sat_thresh"], 255)
        if self.method == "thresh":
            # 强度阈值:物体亮度在背景带 [lo,hi] 之外(过暗或过亮)。
            # 无状态,不建模,天然免疫任意纹理运动(只要物体亮度可区分)。
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lo, hi = self.P["thresh_lo"], self.P["thresh_hi"]
            return ((g < lo) | (g > hi)).astype(np.uint8) * 255
        if self.method == "refbg":
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
            fg = (np.abs(g - self.ref) > self.P["ref_thresh"]).astype(np.uint8) * 255
            a = self.P["ref_alpha"]
            if a > 0:   # 背景像素处慢更新参考帧，抗光照漂移
                bgmask = fg == 0
                self.ref[bgmask] = ((1 - a) * self.ref[bgmask]
                                    + a * g[bgmask]).astype(np.int16)
            return fg
        return self.bg.apply(frame)

    def _merge_close(self, dets):
        """Merge detections connected by pairwise centroid distances below the threshold."""
        md = self.P["merge_dist"]
        if md <= 0 or len(dets) < 2:
            return dets
        parent = list(range(len(dets)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                if math.hypot(dets[i][0] - dets[j][0],
                              dets[i][1] - dets[j][1]) < md:
                    union(i, j)

        groups = {}
        for i in range(len(dets)):
            groups.setdefault(find(i), []).append(i)
        merged = []
        for members in groups.values():
            x0 = min(dets[i][2] for i in members)
            y0 = min(dets[i][3] for i in members)
            x1 = max(dets[i][2] + dets[i][4] for i in members)
            y1 = max(dets[i][3] + dets[i][5] for i in members)
            merged.append(((x0, y0, x1, y1, min(members)),
                           ((x0 + x1) / 2, (y0 + y1) / 2,
                            x0, y0, x1 - x0, y1 - y0)))
        merged.sort(key=lambda item: item[0])
        return [item[1] for item in merged]

    def detect(self, frame):
        ox, oy = 0, 0
        if self.roi is not None:   # 先裁到 ROI,重活只在小区域跑
            ox, oy, x1, y1 = self.roi
            frame = frame[oy:y1, ox:x1]
        fg = self._foreground(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel,
                              iterations=int(self.P["morph_iter"]))

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sa, ua = self.P["split_area"], self.P["unit_area"]
        max_aspect = self.P["max_aspect"]
        dets = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            x += ox   # 偏移回全图坐标
            y += oy
            if max_aspect > 0:   # 滤掉细长噪声条纹
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect > max_aspect:
                    continue
            # 粘连分割：大块按单位面积拆成 n 个，沿运动轴均分
            if sa and ua > 0 and area > sa:
                n = max(1, int(round(area / ua)))
                if n > 1:
                    for j in range(n):
                        if self.axis == "x":
                            dets.append((x + bw * (j + 0.5) / n, y + bh / 2,
                                         x + bw * j / n, y, bw / n, bh))
                        else:
                            dets.append((x + bw / 2, y + bh * (j + 0.5) / n,
                                         x, y + bh * j / n, bw, bh / n))
                    continue
            dets.append((x + bw / 2, y + bh / 2, x, y, bw, bh))
        return self._merge_close(dets)


class Tracker:
    """逐帧接收检测点，速度预测匹配 + 越线计数。"""

    def __init__(self, P, w, h):
        self.P = P
        self.axis = P["axis"]
        dim = w if self.axis == "x" else h
        self.line_pos = int(dim * P["line"])
        self.band = P["line_band"] * dim
        self.tracks = []
        self.count = 0
        self._next_id = 0

    def _new_track(self, cx, cy):
        track = Track(self._next_id, cx, cy)
        self._next_id += 1
        return track

    def update(self, dets, warming=False):
        P, axis = self.P, self.axis
        if axis == "x":
            main_of, vel_of = (lambda t: t.cx), (lambda t: t.vx)
        else:
            main_of, vel_of = (lambda t: t.cy), (lambda t: t.vy)
        for t in self.tracks:
            t.matched = False

        md = P["max_dist"]
        cell = md if md > 0 else 1.0
        grid = {}
        for t in self.tracks:
            px, py = t.cx + t.vx, t.cy + t.vy
            grid.setdefault((int(px // cell), int(py // cell)), []).append((t, px, py))
        pairs = []
        for di, d in enumerate(dets):
            cx, cy = d[0], d[1]
            kx, ky = int(cx // cell), int(cy // cell)
            for ax in (kx - 1, kx, kx + 1):
                for ay in (ky - 1, ky, ky + 1):
                    for t, px, py in grid.get((ax, ay), ()):
                        dist = math.hypot(px - cx, py - cy)
                        if dist < md:
                            pairs.append((dist, di, t))
        pairs.sort(key=lambda z: z[0])
        det_used = [False] * len(dets)
        for _, di, t in pairs:
            if t.matched or det_used[di]:
                continue
            cx, cy = dets[di][0], dets[di][1]
            t.prev_main = main_of(t)
            t.vx, t.vy = cx - t.cx, cy - t.cy
            t.cx, t.cy = cx, cy
            t.matched = True
            t.missing = 0
            t.consecutive_hits = (t.consecutive_hits + 1
                                  if t.consecutive_hits > 0 else 1)
            m = main_of(t)
            t.min_main = min(t.min_main, m)
            t.max_main = max(t.max_main, m)
            det_used[di] = True

        for di, d in enumerate(dets):
            if not det_used[di]:
                t = self._new_track(d[0], d[1])
                t.min_main = t.max_main = main_of(t)
                t.matched = True
                self.tracks.append(t)

        for t in self.tracks:
            if not t.matched:
                t.prev_main = main_of(t)
                t.cx += t.vx
                t.cy += t.vy
                t.missing += 1
                t.consecutive_hits = 0
                m = main_of(t)
                t.min_main = min(t.min_main, m)
                t.max_main = max(t.max_main, m)

        if not warming:
            lp, band = self.line_pos, self.band
            for t in self.tracks:
                if t.counted or t.consecutive_hits < P["min_hits"]:
                    continue
                main = main_of(t)
                vmain = vel_of(t)
                cross_pos = (t.prev_main < lp <= main and t.min_main <= lp - band)
                cross_neg = (t.prev_main > lp >= main and t.max_main >= lp + band)
                if P["min_speed"] > 0:
                    cross_pos = cross_pos and vmain >= P["min_speed"]
                    cross_neg = cross_neg and -vmain >= P["min_speed"]
                crossed = ((P["flow"] == "pos" and cross_pos) or
                           (P["flow"] == "neg" and cross_neg) or
                           (P["flow"] == "both" and (cross_pos or cross_neg)))
                if crossed:
                    t.counted = True
                    self.count += 1

        self.tracks = [t for t in self.tracks if t.missing <= P["track_ttl"]]


# ---- 供调参器复用的分阶段接口 ----


def is_warming(method, idx, warmup):
    """bgsub 前若干帧只建模不计数。"""
    return method == "bgsub" and idx < warmup




def resolve_method(P, frames, w=None, h=None):
    """确定实际分离方式与参考背景。frames 为已按 scale 处理的内存帧列表。"""
    if w is None or h is None:
        h, w = frames[0].shape[:2]
    return prepare_method(P, frames, w, h)


def detect_sequence(frames, P, w, h, method, ref):
    det = Detector(P, method, w, h, ref)
    return [det.detect(f) for f in frames]


def track_sequence(dets_seq, P, w, h, method):
    trk = Tracker(P, w, h)
    for i, dets in enumerate(dets_seq):
        trk.update(dets, is_warming(method, i, P["warmup"]))
    return trk.count


def count_source(source, params=None, fps=30.0, save=None, save_fps=None,
                 save_frames=None, show=False, debug=False, verbose=False,
                 profile=False):
    """对一个视频/图片文件夹计数，返回越线总数(单遍处理，可视化)。

    save_fps: 指定标注视频的播放帧率(不丢帧;低于源帧率=慢放,方便高帧率视频回看)。
    save_frames: 目录路径;把标注帧另存为图片序列(契合图片文件夹工作流)。
    profile: 逐帧打印处理耗时(仅检测+跟踪,不含读帧),结尾给实时性统计。
    """
    P = merge_params(cli_params=params or {})
    src = FrameSource(source, fps)
    writer = None
    try:
        scale = P["scale"]
        w, h = int(src.w * scale), int(src.h * scale)
        validate_params(P, w, h)

        method = P["method"]
        ref = None
        if method in ("auto", "refbg"):
            samp = [src.sample(int(i))
                    for i in np.linspace(0, max(src.n - 1, 0), 15, dtype=int)]
            samp = [scaled(s, scale, w, h) for s in samp if s is not None]
            method, ref = prepare_method(P, samp, w, h)
            if verbose:
                print(f"[method] {method}")

        det = Detector(P, method, w, h, ref)
        trk = Tracker(P, w, h)

        if save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = save_fps if save_fps else src.fps
            writer = cv2.VideoWriter(save, fourcc, out_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"unable to open output video: {save}")
        if save_frames:
            os.makedirs(save_frames, exist_ok=True)

        idx = -1
        proc_ms = [] if profile else None
        for frame in src.frames():
            idx += 1
            frame = scaled(frame, scale, w, h)
            t0 = time.perf_counter() if profile else 0.0
            dets = det.detect(frame)
            warming = is_warming(method, idx, P["warmup"])
            prev = trk.count
            trk.update(dets, warming)
            if profile:
                dt = (time.perf_counter() - t0) * 1000.0
                proc_ms.append(dt)
                print(f"[profile] frame {idx}: {dt:.2f} ms")
            if debug and trk.count > prev:
                print(f"[count {trk.count}] frame={idx}")

            if show or writer or save_frames:
                vis = frame.copy()
                if trk.axis == "x":
                    cv2.line(vis, (trk.line_pos, 0), (trk.line_pos, h),
                             (0, 0, 255), 2)
                else:
                    cv2.line(vis, (0, trk.line_pos), (w, trk.line_pos),
                             (0, 0, 255), 2)
                for d in dets:
                    _, _, x, y, bw, bh = d
                    cv2.rectangle(vis, (int(x), int(y)),
                                  (int(x + bw), int(y + bh)), (0, 255, 0), 2)
                for t in trk.tracks:
                    c = (int(t.cx), int(t.cy))
                    if t.counted:
                        cv2.circle(vis, c, 6, (0, 255, 255), -1)
                    cv2.putText(vis, str(t.id), (c[0] + 6, c[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 0), 2)
                cv2.putText(vis, f"count: {trk.count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                if writer:
                    writer.write(vis)
                if save_frames:
                    cv2.imwrite(os.path.join(save_frames,
                                             f"frame_{idx:06d}.jpg"), vis)
                if show:
                    cv2.imshow("count", vis)
                    if cv2.waitKey(1) == 27:
                        break

        if profile and proc_ms:
            a = np.array(proc_ms)
            print(f"[profile] 帧数={len(a)} 平均={a.mean():.2f}ms "
                  f"中位={np.median(a):.2f}ms "
                  f"p95={np.percentile(a, 95):.2f}ms 最大={a.max():.2f}ms")
            eff = 1000.0 / a.mean() if a.mean() > 0 else float("inf")
            line = f"[profile] 处理吞吐≈{eff:.0f} fps"
            if src.fps:
                budget = 1000.0 / src.fps
                ok = np.percentile(a, 95) <= budget
                line += (f" | 源{src.fps:.0f}fps 每帧预算{budget:.2f}ms -> "
                         f"{'满足实时✅' if ok else '达不到实时(p95超预算)⚠️'}")
            print(line)

        return trk.count
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def find_gt(source, meta_arg):
    meta = meta_arg
    if meta is None:
        if os.path.isdir(source):
            cands = glob.glob(os.path.join(source, "*_meta.json"))
            meta = cands[0] if cands else None
        else:
            cand = os.path.splitext(source)[0] + "_meta.json"
            meta = cand if os.path.exists(cand) else None
    if meta and os.path.exists(meta):
        with open(meta) as fh:
            return json.load(fh).get("crossed_center_line")
    return None


