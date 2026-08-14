#!/usr/bin/env python3
"""
用一批带真值的样例，网格搜索 count_cv 的最佳参数（真值只需"总数"）。

性能:
  1) 每段样例帧缓存复用；超内存预算则自动回退为"逐组合重读"(内存安全)。
  2) 检测/跟踪分阶段：相同检测参数只检测一次，再扫所有跟踪组合(极轻量)。
  3) 多进程并行处理不同样例。

鲁棒:
  - 打分用所有样例的绝对误差之和(SAE)，并报告带符号 BIAS(暴露抵消)。
  - 多个并列最优时，按"平台中心度"选参：优先四周邻居也都最优的参数
    (离失败边界最远、最不敏感)，而非任意/最激进的取值。
  - 可选 --val 验证集：用选出的参数在未参与搜索的样例上评估泛化。

用法:
    python tune_params.py samples.txt --axis x --flow both --scale 0.5 --jobs 4
    python tune_params.py samples.txt --val val.txt --mem-mb 4000
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

from count_cv import (DEFAULT_PARAMS, DET_KEYS, TRK_KEYS, FrameSource,
                      decode_all, resolve_method, detect_sequence,
                      track_sequence, count_source, find_gt, scaled)
import cv2


DEFAULT_GRID = {
    "min_area": [150, 300, 500],
    "bg_var": [25, 40, 60],
    "morph_kernel": [5, 7, 9],
    "max_dist": [100, 150, 200],
}


class FrameCache:
    """内存安全的帧源：帧总量在预算内则驻留内存反复用，否则每次重读磁盘。"""

    def __init__(self, source, scale=1.0, mem_mb=2000):
        src = FrameSource(source)
        self.source = source
        self.scale = scale
        self.w, self.h = int(src.w * scale), int(src.h * scale)
        self.n = src.n
        src.release()
        est_mb = max(self.n, 1) * self.w * self.h * 3 / 1e6
        self.in_ram = est_mb <= mem_mb
        self.frames = decode_all(source, scale)[0] if self.in_ram else None

    def passes(self):
        if self.in_ram:
            return iter(self.frames)
        return self._stream()

    def _stream(self):
        src = FrameSource(self.source)
        for f in src.frames():
            yield scaled(f, self.scale, self.w, self.h)
        src.release()

    def samples(self, k=15):
        import numpy as np
        if self.in_ram:
            idxs = np.linspace(0, len(self.frames) - 1, min(k, len(self.frames)),
                               dtype=int)
            return [self.frames[i] for i in idxs]
        # 非驻留：按索引 seek 抓 k 帧，避免整段重解码
        src = FrameSource(self.source)
        out = [src.sample(int(i))
               for i in np.linspace(0, max(self.n - 1, 0), k, dtype=int)]
        src.release()
        return [scaled(s, self.scale, self.w, self.h) for s in out if s is not None]


def load_samples(manifest):
    samples = []
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            path = parts[0]
            gt = int(parts[1]) if len(parts) > 1 and parts[1] else find_gt(path, None)
            if gt is None:
                raise SystemExit(f"样例缺少真值: {path}")
            samples.append((path, gt))
    if not samples:
        raise SystemExit("清单为空")
    return samples


def eval_sample(task):
    """在一段样例上评估所有参数组合，返回 {combo_tuple: count}。分阶段+内存安全。"""
    path, base, grid, scale, mem_mb = task
    cache = FrameCache(path, scale=scale, mem_mb=mem_mb)
    w, h = cache.w, cache.h
    method, ref = resolve_method(base, cache.samples(), w, h)

    keys = list(grid.keys())
    det_keys = [k for k in keys if k in DET_KEYS]
    trk_keys = [k for k in keys if k in TRK_KEYS]

    results = {}
    for det_vals in itertools.product(*[grid[k] for k in det_keys]):
        P = dict(base)
        P.update(dict(zip(det_keys, det_vals)))
        dets_seq = detect_sequence(cache.passes(), P, w, h, method, ref)
        for trk_vals in itertools.product(*[grid[k] for k in trk_keys]):
            Pt = dict(P)
            Pt.update(dict(zip(trk_keys, trk_vals)))
            count = track_sequence(dets_seq, Pt, w, h, method)
            combo = {**dict(zip(det_keys, det_vals)),
                     **dict(zip(trk_keys, trk_vals))}
            results[tuple(combo[k] for k in keys)] = count
    return results


def score_combos(samples, per_sample, grid):
    combos = list(itertools.product(*grid.values()))
    scored = []
    for combo in combos:
        sae = bias = 0
        per = []
        for (path, gt), res in zip(samples, per_sample):
            err = res[combo] - gt
            sae += abs(err); bias += err
            per.append((res[combo], gt, err))
        scored.append((sae, abs(bias), combo, bias, per))
    scored.sort(key=lambda r: (r[0], r[1]))
    return scored


def pick_robust(scored, grid):
    """在并列最优(SAE 相同)里，选"平台中心度"最高者。

    平台中心度 = 网格中相邻(单参数±1格)也同为最优的邻居数；
    并列再按"离网格中心近"打破(参数取值更居中=更不敏感)。
    """
    keys = list(grid.keys())
    val_lists = [grid[k] for k in keys]
    idx_of = [{v: i for i, v in enumerate(vals)} for vals in val_lists]
    best_sae = scored[0][0]
    sae_map = {s[2]: s[0] for s in scored}
    tied = [s for s in scored if s[0] == best_sae]

    def robustness(combo):
        idx = [idx_of[a][combo[a]] for a in range(len(keys))]
        r = 0
        for a in range(len(keys)):
            for step in (-1, 1):
                j = idx[a] + step
                if 0 <= j < len(val_lists[a]):
                    nb = list(combo); nb[a] = val_lists[a][j]
                    if sae_map.get(tuple(nb)) == best_sae:
                        r += 1
        return r

    def centrality(combo):
        idx = [idx_of[a][combo[a]] for a in range(len(keys))]
        mids = [(len(v) - 1) / 2 for v in val_lists]
        return sum(abs(idx[a] - mids[a]) for a in range(len(keys)))

    tied.sort(key=lambda s: (-robustness(s[2]), centrality(s[2])))
    return tied[0], robustness(tied[0][2]), len(tied)


def main():
    p = argparse.ArgumentParser(description="网格搜索 count_cv 最佳参数(鲁棒+加速)")
    p.add_argument("manifest", help="样例清单文件")
    p.add_argument("--val", default=None, help="验证集清单(不参与搜索,只评泛化)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out", default="best_params.json")
    p.add_argument("--jobs", type=int, default=1, help="并行进程数(按样例)")
    p.add_argument("--mem-mb", type=float, default=2000, help="每段帧缓存内存上限(MB)")
    p.add_argument("--axis", choices=["x", "y"], default=None)
    p.add_argument("--flow", choices=["pos", "neg", "both"], default=None)
    p.add_argument("--method", choices=["auto", "color", "bgsub", "refbg"], default=None)
    p.add_argument("--line", type=float, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--roi", default=None)
    p.add_argument("--grid", default=None, help="自定义网格json")
    args = p.parse_args()

    samples = load_samples(args.manifest)
    print(f"样例 {len(samples)} 段: " +
          ", ".join(f"{os.path.basename(s)}(真值{g})" for s, g in samples))

    base = dict(DEFAULT_PARAMS)
    for key in ("axis", "flow", "method", "line", "scale", "roi"):
        v = getattr(args, key)
        if v is not None:
            base[key] = v
    scale = base["scale"]

    grid = json.loads(args.grid) if args.grid else DEFAULT_GRID
    keys = list(grid.keys())
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"网格: {grid}")
    print(f"共 {n_combos} 组 × {len(samples)} 段 (分阶段+缓存, jobs={args.jobs}, "
          f"mem<={args.mem_mb}MB)\n")

    tasks = [(path, base, grid, scale, args.mem_mb) for path, _ in samples]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            per_sample = list(ex.map(eval_sample, tasks))
    else:
        per_sample = [eval_sample(t) for t in tasks]

    scored = score_combos(samples, per_sample, grid)

    print(f"=== 前 {args.topk} 组(按SAE升序) ===")
    header = "  ".join(f"{k:>11}" for k in keys)
    print(f"{'SAE':>5} {'BIAS':>5}  {header}")
    for sae, _, combo, bias, _ in scored[:args.topk]:
        vals = "  ".join(f"{v:>11}" for v in combo)
        print(f"{sae:>5} {bias:>+5}  {vals}")

    (best_sae, _, best_combo, best_bias, best_per), robust, n_tied = \
        pick_robust(scored, grid)
    best_full = dict(base)
    best_full.update(dict(zip(keys, best_combo)))
    print(f"\n=== 最佳 (SAE={best_sae}, BIAS={best_bias:+d}) ===")
    print(f"并列最优 {n_tied} 组，按平台中心度选出(鲁棒邻居数={robust}):")
    print("  ", dict(zip(keys, best_combo)))
    for (path, _), (c, g, err) in zip(samples, best_per):
        print(f"  {os.path.basename(path):<24} {c}/{g}  误差{err:+d}")

    if args.val:
        print("\n=== 验证集(泛化检查) ===")
        vsae = 0
        for path, gt in load_samples(args.val):
            c = count_source(path, best_full)
            vsae += abs(c - gt)
            print(f"  {os.path.basename(path):<24} {c}/{gt}  误差{c-gt:+d}")
        print(f"  验证集 SAE={vsae}")

    with open(args.out, "w") as fh:
        json.dump(best_full, fh, ensure_ascii=False, indent=2)
    print(f"\n最佳参数已写入: {args.out}")
    print("套用: python count_cv.py <source> " +
          " ".join(f"--{k.replace('_','-')} {v}"
                   for k, v in zip(keys, best_combo)))


if __name__ == "__main__":
    main()
