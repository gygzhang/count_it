#!/usr/bin/env python3
"""Command-line entry point for the conveyor counter."""
import argparse

from params import DEFAULT_PARAMS, load_params, merge_params
from counting import count_source, find_gt


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="视频文件 或 图片文件夹")
    p.add_argument("--params", default=None, help="JSON参数文件")
    p.add_argument("--fps", type=float, default=30.0, help="图片文件夹帧率(仅影响保存)")
    # 检测
    p.add_argument("--method", choices=["auto", "color", "bgsub", "refbg", "thresh"],
                   default=argparse.SUPPRESS, help="前景分离方式")
    p.add_argument("--sat-thresh", type=int, default=argparse.SUPPRESS, help="color模式饱和度阈值")
    p.add_argument("--thresh-lo", type=int, default=argparse.SUPPRESS,
                   help="thresh模式:暗于此灰度=前景(背景带下界)")
    p.add_argument("--thresh-hi", type=int, default=argparse.SUPPRESS,
                   help="thresh模式:亮于此灰度=前景(背景带上界)")
    p.add_argument("--min-area", type=int, default=argparse.SUPPRESS, help="轮廓最小面积(像素)")
    p.add_argument("--max-area", type=int, default=argparse.SUPPRESS, help="轮廓最大面积(像素,0=不限)")
    p.add_argument("--min-area-frac", type=float, default=argparse.SUPPRESS,
                   help="最小面积占画面比例(0~1,优先于像素;分辨率/缩放无关)")
    p.add_argument("--max-area-frac", type=float, default=argparse.SUPPRESS,
                   help="最大面积占画面比例(0~1,优先于像素;0=不限)")
    p.add_argument("--max-aspect", type=float, default=argparse.SUPPRESS,
                   help="最大长宽比(0=不限;滤除细长噪声条纹)")
    p.add_argument("--morph-kernel", type=int, default=argparse.SUPPRESS, help="形态学核大小(奇数)")
    p.add_argument("--morph-iter", type=int, default=argparse.SUPPRESS, help="闭运算迭代次数")
    p.add_argument("--bg-history", type=int, default=argparse.SUPPRESS, help="MOG2历史帧数")
    p.add_argument("--bg-var", type=float, default=argparse.SUPPRESS, help="MOG2方差阈值")
    p.add_argument("--ref-thresh", type=int, default=argparse.SUPPRESS, help="refbg灰度差阈值")
    p.add_argument("--bg-ref", default=argparse.SUPPRESS, help="refbg基准图路径(缺省或auto=自动中位数)")
    p.add_argument("--ref-alpha", type=float, default=argparse.SUPPRESS,
                   help="refbg参考帧慢更新系数(0=静态;>0抗光照漂移)")
    p.add_argument("--split-area", type=int, default=argparse.SUPPRESS, help="粘连分割触发面积(0=关)")
    p.add_argument("--unit-area", type=int, default=argparse.SUPPRESS, help="单个物体典型面积(配合分割)")
    p.add_argument("--merge-dist", type=float, default=argparse.SUPPRESS,
                   help="合并质心过近的检测(0=关;修复碎裂重复计数)")
    p.add_argument("--roi", default=argparse.SUPPRESS, help="感兴趣区 'x0,y0,x1,y1'(处理分辨率)")
    p.add_argument("--scale", type=float, default=argparse.SUPPRESS, help="处理前缩放系数")
    # 跟踪/计数
    p.add_argument("--max-dist", type=float, default=argparse.SUPPRESS, help="预测-检测最大匹配距离")
    p.add_argument("--track-ttl", type=int, default=argparse.SUPPRESS, help="轨迹漏检存活帧数")
    p.add_argument("--min-hits", type=int, default=argparse.SUPPRESS, help="轨迹连续确认帧数才计数")
    p.add_argument("--min-speed", type=float, default=argparse.SUPPRESS, help="计数所需最小轴向速度(px/帧)")
    p.add_argument("--line", type=float, default=argparse.SUPPRESS, help="计数线位置(画面比例)")
    p.add_argument("--line-band", type=float, default=argparse.SUPPRESS, help="迟滞带宽(画面比例)")
    p.add_argument("--axis", choices=["x", "y"], default=argparse.SUPPRESS, help="运动/计数轴")
    p.add_argument("--flow", choices=["pos", "neg", "both"], default=argparse.SUPPRESS,
                   help="计数方向")
    p.add_argument("--warmup", type=int, default=argparse.SUPPRESS, help="bgsub预热帧(只建模不计数)")
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


def resolve_cli_params(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    file_params = {}
    if args.params:
        try:
            file_params = load_params(args.params)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    cli_params = {key: getattr(args, key) for key in DEFAULT_PARAMS if hasattr(args, key)}
    try:
        params = merge_params(file_params, cli_params)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return args.source, params, args


def main():
    source, params, args = resolve_cli_params()
    count = count_source(source, params, fps=args.fps, save=args.save,
                         save_fps=args.save_fps, save_frames=args.save_frames,
                         show=args.show, debug=args.debug, verbose=True,
                         profile=args.profile)
    print(f"CV 计数结果: {count}")
    gt = find_gt(source, args.meta)
    if gt is not None:
        print(f"真值(越过中线): {gt}  |  误差: {count - gt}")


if __name__ == "__main__":
    main()
