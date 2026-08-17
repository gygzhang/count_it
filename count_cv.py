#!/usr/bin/env python3
"""Command-line entry point for the conveyor counter."""
import argparse

from params import DEFAULT_PARAMS
from counting import count_source, find_gt

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
