# 传送带物体计数(纯 OpenCV)

传送带/流水线上连续经过物体的**检测与计数**系统,纯 OpenCV 实现,**不依赖深度学习、不需要训练、不需要 GPU**。适用于:物体不遮挡、单列经过、背景相对干净的计数场景。

配套一套合成数据生成 + 自动调参工具,可在没有真实数据时先跑通全链路。

## 依赖

```
python3, opencv-python, numpy
```

## 四个脚本

| 脚本 | 作用 |
|------|------|
| `gen_shapes_video.py` | 生成合成传送带视频(不规则形状,可控速度/密度/灰度/噪声/模糊)+ 计数真值 + YOLO 标注 |
| `video_to_frames.py` | 视频 → 图片文件夹(工业现场逐帧落盘的常见形态) |
| `count_cv.py` | **核心**:视频或图片文件夹 → 计数(检测 + 跟踪 + 越线计数) |
| `tune_params.py` | 用带真值样例网格搜索 `count_cv` 的最佳参数 |

## 快速开始

```bash
# 1) 生成一段灰度视频(或换成你的真实相机视频)
python3 gen_shapes_video.py -o belt.mp4 --gray --fps 30 --speed 300 --count 6 --duration 10

# 2) 抽帧成图片文件夹
python3 video_to_frames.py belt.mp4 frames_belt

# 3) 计数(自动识别灰度/彩色,自动查找真值打印误差)
python3 count_cv.py frames_belt --save out.mp4
```

## 处理流程

```
[相机采集 / gen_shapes_video]  →  .mp4 (+ _meta.json 真值)
        → [video_to_frames]  →  frames/  (逐帧图片)
        → [count_cv]         →  计数结果 (+ 可视化视频)
                ↑ 参数
        [tune_params] → best_params.json  (用带真值样例搜出)
```

`count_cv` 内部两阶段:

```
每帧 → 检测(Detector): 前景分离 → 去噪 → 轮廓 → 面积/形状过滤 → 粘连分割/碎裂合并
     → 跟踪(Tracker):  全局最短距离匹配(速度预测) → 轨迹确认/方向门控 → 越线(可迟滞)计数
```

## count_cv 参数

### 前景分离
| 参数 | 默认 | 说明 |
|------|------|------|
| `--method` | auto | `bgsub`(MOG2背景减除)/`color`(HSV饱和度)/`refbg`(参考帧)/`auto`(按饱和度自动选) |
| `--sat-thresh` | 60 | color 模式饱和度阈值 |
| `--bg-history` `--bg-var` | 200 / 40 | MOG2 建模历史帧数 / 前景灵敏度(噪声大调大) |
| `--ref-thresh` | 25 | refbg 灰度差阈值 |
| `--bg-ref` | auto | refbg 基准图路径(缺省=帧中位数自动生成) |
| `--ref-alpha` | 0 | refbg 参考帧慢更新系数(>0 抗光照缓慢漂移) |

### 检测过滤
| 参数 | 默认 | 说明 |
|------|------|------|
| `--min-area` `--max-area` | 300 / 0 | 轮廓面积下/上限(滤噪声/超大块;0=不限) |
| `--max-aspect` | 0 | 最大长宽比(滤细长噪声条纹;0=不限) |
| `--morph-kernel` `--morph-iter` | 7 / 2 | 形态学核大小 / 闭运算迭代(补洞连碎块) |
| `--split-area` `--unit-area` | 0 / 0 | 粘连分割:面积超阈值的大块按单体面积拆成 N 个 |
| `--merge-dist` | 0 | 合并质心过近的检测(修复单物体碎裂→重复计数) |
| `--roi` | 无 | 只处理感兴趣区 `x0,y0,x1,y1`,忽略无关区域 |
| `--scale` | 1.0 | 处理前缩放(<1 加速大图;面积/ROI 按缩放后像素) |

### 跟踪 / 计数
| 参数 | 默认 | 说明 |
|------|------|------|
| `--max-dist` | 140 | 预测位置与检测的最大匹配距离(高速需调大到 ≥ 每帧位移) |
| `--track-ttl` | 5 | 轨迹漏检后存活帧数(短暂遮挡/闪断调大) |
| `--min-hits` | 1 | 轨迹连续确认 N 帧才允许计数(抑制噪声闪现误计) |
| `--min-speed` | 0 | 计数所需最小轴向速度(排除静止噪声;0=关) |
| `--line` | 0.5 | 计数线位置(画面比例) |
| `--line-band` | 0 | 迟滞带宽(画面比例),防抖动重复计 |
| `--axis` | x | 运动/计数轴:x=横向传送带,y=竖向 |
| `--flow` | both | 计数方向:pos(坐标增大)/neg(减小)/both |
| `--warmup` | 8 | bgsub 前若干帧只建模不计数 |

### 输出
`--save out.mp4`(可视化)、`--show`(实时窗口)、`--meta path`(真值json)、`--debug`(打印计数事件)

## 自动调参

真值**只需每段总数**即可调参。

```bash
# 1) 写清单 samples.txt,每行: 路径[,真值]  (留空则从同名/同目录 *_meta.json 自动取)
#    frames_a,120
#    clip_b.mp4,86
# 2) 固定场景参数(不搜),搜检测/跟踪参数;多进程加速
python3 tune_params.py samples.txt --axis x --flow both --scale 0.5 --jobs 4

# 3) 可选:自定义网格 + 验证集泛化检查
python3 tune_params.py samples.txt --val val.txt \
    --grid '{"min_area":[60,120,200],"bg_var":[15,30,50],"max_dist":[120,180]}'
```

输出 `best_params.json`,并打印可直接套用的命令行。

**调参器要点**
- 打分 = 所有样例绝对误差之和(SAE),同时报带符号 BIAS 暴露"漏检/误检互相抵消"。
- 多组并列最优时,按**平台中心度**选参(四周邻居也都最优 = 离失败边界最远,最鲁棒)。
- 帧缓存内存安全:超 `--mem-mb` 预算自动回退逐帧重读,长视频不爆内存。
- 检测/跟踪分阶段:相同检测参数只检测一次,跟踪组合几乎免费。

## 数据生成参数(gen_shapes_video)

`--gray`(灰度模式)、`--speed`(px/s)、`--count`(同屏物体数)、`--fps`、`--duration`、`--width/--height`、`--direction`(l2r/r2l/t2b/b2t)、`--motion-blur`、`--wobble`(形状抖动)、`--noise`(传感器噪声)、`--deform`(形状变形)、`--labels DIR`(导出 YOLO 标注)。

## 已知限制(物理层面,非软件可解)

- **采样混叠**:每帧位移应 < 物体尺寸且 < 物体间距,否则跟踪断裂。高速需高帧率相机。
- **运动模糊**:高速下需短曝光/频闪冻结运动,否则物体拉成条纹。
- **粘连/乱序/多列**:超出"单列不遮挡"假设时精度下降。
- 只用总数调参是**弱信号**:样例需覆盖真实速度/光照/密度范围,避免过拟合。

极高速纯计数场景,光电传感器往往比视觉更实际。
