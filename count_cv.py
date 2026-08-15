#!/usr/bin/env python3
"""
纯 OpenCV 传送带物体计数（不用深度学习）。

输入可以是【视频文件】或【图片文件夹】(逐帧图片，工业现场常见)。
流程分两阶段(便于调参复用):
  检测(Detector): 前景分离 -> 去噪 -> 轮廓 -> (可选粘连分割) -> 每帧检测点
  跟踪(Tracker):  速度预测匹配 -> 轨迹确认/方向门控 -> 越线(可迟滞)计数

前景分离方式:
  - bgsub:  背景减除(MOG2)，灰度工业相机常见
  - color:  HSV 饱和度(彩色物体/灰背景)
  - refbg:  参考帧背景减除(空传送带基准图/自动中位数)，对传送带最稳
  - auto:   按采样帧饱和度在 color/bgsub 间自动选

计数核心(Detector/Tracker)供调参脚本 tune_params.py 分阶段复用以提速。
"""
import argparse
import glob
import json
import math
import os
import re
import time

import cv2
import numpy as np


# 支持的图像格式(OpenCV imread 可读)
IMG_EXTS = (".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dib",
            ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm", ".pnm",
            ".jp2", ".ras", ".sr")


def natural_key(name):
    """自然排序键:让 frame2 < frame10(非零填充命名也能正确排序)。"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


DEFAULT_PARAMS = {
    # 检测
    "method": "auto", "sat_thresh": 60,
    "thresh_lo": 50, "thresh_hi": 205,
    "min_area": 300, "max_area": 0, "max_aspect": 0.0,
    "min_area_frac": 0.0, "max_area_frac": 0.0,
    "morph_kernel": 7, "morph_iter": 2,
    "bg_history": 200, "bg_var": 40.0,
    "ref_thresh": 25, "bg_ref": None, "ref_alpha": 0.0,
    "split_area": 0, "unit_area": 0, "merge_dist": 0.0,
    "roi": None, "scale": 1.0,
    # 跟踪/计数
    "max_dist": 140.0, "track_ttl": 5,
    "min_hits": 1, "min_speed": 0.0,
    "line": 0.5, "line_band": 0.0,
    "axis": "x", "flow": "both", "warmup": 8,
}

# 供调参器区分"检测参数"与"跟踪参数"(检测结果可跨跟踪组合复用)
DET_KEYS = {"method", "sat_thresh", "thresh_lo", "thresh_hi",
            "min_area", "max_area", "max_aspect",
            "min_area_frac", "max_area_frac",
            "morph_kernel", "morph_iter", "bg_history", "bg_var", "ref_thresh",
            "bg_ref", "ref_alpha", "split_area", "unit_area", "merge_dist",
            "roi", "scale"}
TRK_KEYS = {"max_dist", "track_ttl", "min_hits", "min_speed", "line",
            "line_band", "axis", "flow", "warmup"}


class FrameSource:
    """统一封装：视频文件 或 图片文件夹，按顺序产出帧。"""

    def __init__(self, path, fps=30.0):
        self.path = path
        self.is_dir = os.path.isdir(path)
        if self.is_dir:
            self.files = sorted(
                (os.path.join(path, f) for f in os.listdir(path)
                 if f.lower().endswith(IMG_EXTS)),
                key=lambda p: natural_key(os.path.basename(p)))
            if not self.files:
                raise RuntimeError(
                    f"文件夹中没有支持的图片({'/'.join(IMG_EXTS)}): {path}")
            first = cv2.imread(self.files[0])
            if first is None:
                raise RuntimeError(f"无法读取图片: {self.files[0]}")
            self.h, self.w = first.shape[:2]
            self.n = len(self.files)
            self.fps = fps
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开: {path}")
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def sample(self, i):
        if self.is_dir:
            return cv2.imread(self.files[min(i, self.n - 1)])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = self.cap.read()
        return f if ret else None

    def frames(self):
        if self.is_dir:
            for f in self.files:
                img = cv2.imread(f)
                if img is not None:
                    yield img
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, f = self.cap.read()
                if not ret:
                    break
                yield f

    def release(self):
        if not self.is_dir:
            self.cap.release()


class Track:
    _next_id = 0

    def __init__(self, cx, cy):
        self.id = Track._next_id
        Track._next_id += 1
        self.cx, self.cy = cx, cy
        self.vx, self.vy = 0.0, 0.0
        self.prev_main = cx
        self.min_main = float("inf")
        self.max_main = float("-inf")
        self.hits = 0
        self.missing = 0
        self.counted = False
        self.matched = False


def parse_roi(roi, w, h):
    if roi is None:
        return None
    if isinstance(roi, str):
        roi = [int(v) for v in roi.split(",")]
    return (max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3]))


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
        """合并质心过近的检测点(修复单物体碎裂成多块 -> 重复计数)。"""
        md = self.P["merge_dist"]
        if md <= 0 or len(dets) < 2:
            return dets
        used = [False] * len(dets)
        merged = []
        for i in range(len(dets)):
            if used[i]:
                continue
            group = [dets[i]]
            used[i] = True
            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                if any(math.hypot(g[0] - dets[j][0], g[1] - dets[j][1]) < md
                       for g in group):
                    group.append(dets[j])
                    used[j] = True
            x0 = min(g[2] for g in group)
            y0 = min(g[3] for g in group)
            x1 = max(g[2] + g[4] for g in group)
            y1 = max(g[3] + g[5] for g in group)
            merged.append(((x0 + x1) / 2, (y0 + y1) / 2, x0, y0, x1 - x0, y1 - y0))
        return merged

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

    def update(self, dets, warming=False):
        P, axis = self.P, self.axis
        # 沿计数轴取位置/速度(避免到处写 cx if axis=="x" else cy)
        if axis == "x":
            main_of, vel_of = (lambda t: t.cx), (lambda t: t.vx)
        else:
            main_of, vel_of = (lambda t: t.cy), (lambda t: t.vy)
        for t in self.tracks:
            t.matched = False

        # 全局最短距离贪心匹配(顺序无关)。用空间网格分桶把候选对从
        # O(轨迹×检测) 降到 ~O(N):格边长=max_dist,则任意 <max_dist 的
        # 轨迹-检测对必在同格或相邻 8 格,结果与全量枚举完全一致。
        md = P["max_dist"]
        cell = md if md > 0 else 1.0
        grid = {}
        for t in self.tracks:
            px, py = t.cx + t.vx, t.cy + t.vy   # 预测位置
            grid.setdefault((int(px // cell), int(py // cell)), []).append(
                (t, px, py))
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
        for dist, di, t in pairs:
            if t.matched or det_used[di]:
                continue
            cx, cy = dets[di][0], dets[di][1]
            t.prev_main = main_of(t)
            t.vx, t.vy = cx - t.cx, cy - t.cy
            t.cx, t.cy = cx, cy
            t.matched = True
            t.missing = 0
            t.hits += 1
            m = main_of(t)
            t.min_main = min(t.min_main, m)
            t.max_main = max(t.max_main, m)
            det_used[di] = True

        # 未匹配的检测点 -> 新建轨迹
        for di, d in enumerate(dets):
            if not det_used[di]:
                t = Track(d[0], d[1])
                t.hits = 1
                t.min_main = t.max_main = main_of(t)
                t.matched = True
                self.tracks.append(t)

        # 未匹配的轨迹按速度惯性滑行(容忍短暂漏检)
        for t in self.tracks:
            if not t.matched:
                t.prev_main = main_of(t)
                t.cx += t.vx
                t.cy += t.vy
                t.missing += 1
                m = main_of(t)
                t.min_main = min(t.min_main, m)
                t.max_main = max(t.max_main, m)

        if not warming:
            lp, band = self.line_pos, self.band
            for t in self.tracks:
                if t.counted or t.hits < P["min_hits"]:
                    continue
                main = main_of(t)
                vmain = vel_of(t)
                # 越线 + 迟滞(要求确实从线另一侧带 band 走过来)
                cross_pos = (t.prev_main < lp <= main and t.min_main <= lp - band)
                cross_neg = (t.prev_main > lp >= main and t.max_main >= lp + band)
                if P["min_speed"] > 0:   # 速度方向门控
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

def scaled(frame, scale, w, h):
    """按需缩放一帧(scale==1 时原样返回)。"""
    return frame if scale == 1.0 else cv2.resize(frame, (w, h))


def is_warming(method, idx, warmup):
    """bgsub 前若干帧只建模不计数。"""
    return method == "bgsub" and idx < warmup


def decode_all(source, scale=1.0, fps=30.0):
    """解码为内存帧列表(可缩放)。返回 (frames, w, h)。"""
    src = FrameSource(source, fps)
    w, h = int(src.w * scale), int(src.h * scale)
    frames = [scaled(f, scale, w, h) for f in src.frames()]
    src.release()
    return frames, w, h


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
    P = {**DEFAULT_PARAMS, **(params or {})}
    src = FrameSource(source, fps)
    scale = P["scale"]
    w, h = int(src.w * scale), int(src.h * scale)

    # refbg/auto 需要预扫样例帧统一解析
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

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = save_fps if save_fps else src.fps
        writer = cv2.VideoWriter(save, fourcc, out_fps, (w, h))
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
            dt = (time.perf_counter() - t0) * 1000.0   # 仅检测+跟踪耗时
            proc_ms.append(dt)
            print(f"[profile] frame {idx}: {dt:.2f} ms")
        if debug and trk.count > prev:
            print(f"[count {trk.count}] frame={idx}")

        if show or writer or save_frames:
            vis = frame.copy()
            if trk.axis == "x":
                cv2.line(vis, (trk.line_pos, 0), (trk.line_pos, h), (0, 0, 255), 2)
            else:
                cv2.line(vis, (0, trk.line_pos), (w, trk.line_pos), (0, 0, 255), 2)
            for d in dets:
                _, _, x, y, bw, bh = d
                cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)),
                              (0, 255, 0), 2)
            # 轨迹 ID + 已计数高亮(便于人工核对)
            for t in trk.tracks:
                c = (int(t.cx), int(t.cy))
                if t.counted:
                    cv2.circle(vis, c, 6, (0, 255, 255), -1)   # 黄点=已计过
                cv2.putText(vis, str(t.id), (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(vis, f"count: {trk.count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            if writer:
                writer.write(vis)
            if save_frames:
                cv2.imwrite(os.path.join(save_frames, f"frame_{idx:06d}.jpg"), vis)
            if show:
                cv2.imshow("count", vis)
                if cv2.waitKey(1) == 27:
                    break

    src.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if profile and proc_ms:
        a = np.array(proc_ms)
        print(f"[profile] 帧数={len(a)} 平均={a.mean():.2f}ms 中位={np.median(a):.2f}ms "
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


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="视频文件 或 图片文件夹")
    p.add_argument("--fps", type=float, default=30.0, help="图片文件夹帧率(仅影响保存)")
    # 检测
    p.add_argument("--method", choices=["auto", "color", "bgsub", "refbg", "thresh"],
                   default="auto", help="前景分离方式")
    p.add_argument("--sat-thresh", type=int, default=60, help="color模式饱和度阈值")
    p.add_argument("--thresh-lo", type=int, default=50,
                   help="thresh模式:暗于此灰度=前景(背景带下界)")
    p.add_argument("--thresh-hi", type=int, default=205,
                   help="thresh模式:亮于此灰度=前景(背景带上界)")
    p.add_argument("--min-area", type=int, default=300, help="轮廓最小面积(像素)")
    p.add_argument("--max-area", type=int, default=0, help="轮廓最大面积(像素,0=不限)")
    p.add_argument("--min-area-frac", type=float, default=0.0,
                   help="最小面积占画面比例(0~1,优先于像素;分辨率/缩放无关)")
    p.add_argument("--max-area-frac", type=float, default=0.0,
                   help="最大面积占画面比例(0~1,优先于像素;0=不限)")
    p.add_argument("--max-aspect", type=float, default=0.0,
                   help="最大长宽比(0=不限;滤除细长噪声条纹)")
    p.add_argument("--morph-kernel", type=int, default=7, help="形态学核大小(奇数)")
    p.add_argument("--morph-iter", type=int, default=2, help="闭运算迭代次数")
    p.add_argument("--bg-history", type=int, default=200, help="MOG2历史帧数")
    p.add_argument("--bg-var", type=float, default=40, help="MOG2方差阈值")
    p.add_argument("--ref-thresh", type=int, default=25, help="refbg灰度差阈值")
    p.add_argument("--bg-ref", default=None, help="refbg基准图路径(缺省或auto=自动中位数)")
    p.add_argument("--ref-alpha", type=float, default=0.0,
                   help="refbg参考帧慢更新系数(0=静态;>0抗光照漂移)")
    p.add_argument("--split-area", type=int, default=0, help="粘连分割触发面积(0=关)")
    p.add_argument("--unit-area", type=int, default=0, help="单个物体典型面积(配合分割)")
    p.add_argument("--merge-dist", type=float, default=0.0,
                   help="合并质心过近的检测(0=关;修复碎裂重复计数)")
    p.add_argument("--roi", default=None, help="感兴趣区 'x0,y0,x1,y1'(处理分辨率)")
    p.add_argument("--scale", type=float, default=1.0, help="处理前缩放系数")
    # 跟踪/计数
    p.add_argument("--max-dist", type=float, default=140, help="预测-检测最大匹配距离")
    p.add_argument("--track-ttl", type=int, default=5, help="轨迹漏检存活帧数")
    p.add_argument("--min-hits", type=int, default=1, help="轨迹连续确认帧数才计数")
    p.add_argument("--min-speed", type=float, default=0.0, help="计数所需最小轴向速度(px/帧)")
    p.add_argument("--line", type=float, default=0.5, help="计数线位置(画面比例)")
    p.add_argument("--line-band", type=float, default=0.0, help="迟滞带宽(画面比例)")
    p.add_argument("--axis", choices=["x", "y"], default="x", help="运动/计数轴")
    p.add_argument("--flow", choices=["pos", "neg", "both"], default="both",
                   help="计数方向")
    p.add_argument("--warmup", type=int, default=8, help="bgsub预热帧(只建模不计数)")
    # 输出
    p.add_argument("--show", action="store_true", help="实时显示")
    p.add_argument("--save", default=None, help="保存可视化视频")
    p.add_argument("--save-fps", type=float, default=None,
                   help="标注视频播放帧率(不丢帧;低于源帧率=慢放,便于高帧率回看)")
    p.add_argument("--save-frames", default=None,
                   help="把标注帧另存为图片序列到该目录(图片文件夹工作流)")
    p.add_argument("--meta", default=None, help="计数真值json(默认自动查找)")
    p.add_argument("--debug", action="store_true", help="打印计数事件")
    p.add_argument("--profile", action="store_true",
                   help="逐帧打印处理耗时(仅检测+跟踪,不含读帧)+实时性统计")
    return p


def args_to_params(args):
    return {k: getattr(args, k) for k in DEFAULT_PARAMS if hasattr(args, k)}


def main():
    args = build_arg_parser().parse_args()
    params = args_to_params(args)
    count = count_source(args.source, params, fps=args.fps, save=args.save,
                         save_fps=args.save_fps, save_frames=args.save_frames,
                         show=args.show, debug=args.debug, verbose=True,
                         profile=args.profile)
    print(f"CV 计数结果: {count}")
    gt = find_gt(args.source, args.meta)
    if gt is not None:
        print(f"真值(越过中线): {gt}  |  误差: {count - gt}")


if __name__ == "__main__":
    main()
