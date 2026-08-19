# 海康高速相机实时计数应用

## 针对相机

目标型号为 `MV-CS004-10UM`（USB3、黑白、720×540）。官方产品页给出的 V5 最大帧率约为 525.5 fps。最大帧率对应约 1.90 ms 的帧周期，因此曝光时间必须明显短于该周期；默认配置为 500 μs，现场需配合足够亮且稳定的光源。

设备枚举按配置严格过滤为型号 `MV-CS004-10UM`、序列号 `DA8557576`、传输类型 `USB3`，网络上可发现的其他 GigE 相机不会再出现在图像源列表，也不会被诊断或测试按钮误选。

## 模块边界

```text
realtime/hik_camera.py   海康 MVS：枚举、打开、配置、取流、释放缓存
realtime/camera_base.py  相机抽象接口
realtime/video_camera.py 视频模拟相机
realtime/counter.py      单帧检测、轨迹和越线计数
realtime/recorder.py     独立录像线程
realtime/service.py      采集队列、处理线程、统计和生命周期
realtime/ui.py           PySide6 UI，只调用 MeasurementService
run_realtime.py          程序入口
```

UI、SDK 和计数代码之间没有相互导入：UI 不接触海康句柄，计数器不认识相机，海康适配器不认识 UI 或检测算法。

## 安装

1. 安装海康机器人 MVS，勾选 USB3 驱动以及 Development / Python Samples。
2. 确认相机在 MVS 客户端中能以 Mono8 连续取流。
3. 安装 Python 依赖：

   ```powershell
   pip install -r requirements.txt
   ```

4. 检查 SDK 和设备：

   ```powershell
   python diagnose_camera.py
   ```

加载器会自动查找 `MVCAM_COMMON_RUNENV\Samples\Python\MvImport` 和常见 MVS 安装目录。自定义安装时可设置：

```powershell
$env:MVS_PYTHON_SDK="D:\MVS\Development\Samples\Python"
```

## 启动

真实相机：

```powershell
python run_realtime.py
```

没有相机时，用视频模拟完整流程：

```powershell
python run_realtime.py --video "C:\path\sample.avi"
```

UI 中点击“开始测量”后开始最大速率连续取流、计数与约 30 fps 的抽样预览；点击“停止”后停止采集，处理完短队列并关闭可选录像。可在开始前设置“满料数量”：达到目标时计数锁定、相机数字输出置为有效，并显示红色满料提示；采集和录像继续运行。处理完满料后点击“开始下一批”，程序会撤销输出、丢弃切批前的旧帧并从 0 重新计数。

“Line1 接线测试”是独立的接线检查工具，只在未开始测量时使用。它会单独打开当前选中的相机并切换 Line1 输出，不启动计数测量、不运行检测、不读取或修改计数，也不参与满料判断。测试输出开启时按钮显示为红色；关闭测试或开始测量时会先撤销测试输出并释放测试相机句柄。

不同相机的 Line1 节点能力不同：支持 `UserOutput0` 的型号使用软件电平输出；现场的 `MV-CS004-10UM / DA8557576` 只有 Strobe 输出，因此关闭短脉冲 Strobe，并通过 `LineInverter` 保持光耦输出晶体管导通或截止。独立接线测试不会启动采集。

## 缓存与丢帧

- 海康 SDK 使用 `MV_CC_SetImageNodeNum(64)` 增大输出缓存，并使用 `MV_GrabStrategy_OneByOne` 保序取帧。采集线程在复制 Mono8 数据后立即 `MV_CC_FreeImageBuffer`，避免长期占用 SDK 节点。
- 应用处理队列默认 512 帧，可吸收调度、UI 和瞬时负载抖动。720×540 Mono8 每帧约 0.37 MiB，512 帧约占 190 MiB。
- UI 只显示约 30 fps，但计数线程仍处理每个进入队列的帧；UI 抽样不属于丢帧。
- `相机/SDK缺帧` 根据海康 `nFrameNum` 跳变统计；`处理队列丢帧` 是应用处理持续跟不上；`录像队列丢帧` 只影响录像，不阻塞计数。
- 缓存只能吸收短时抖动，无法修复持续吞吐不足。如果处理队列持续增长或丢帧，应缩小 ROI、使用 Mono8、降低 `scale`，或限制相机帧率。不要仅仅无限增大缓存。

## 录像说明

录像为独立的 MJPG/AVI 线程，默认按 525.5 fps 写入，并生成同名 `.avi.json`，记录实际写入帧数、录像队列丢帧数和相机帧号范围。720×540 Mono8 在 525.5 fps 下未压缩数据约 195 MiB/s；启用录像会显著增加 CPU 和磁盘负载，正式部署需实测 SSD 持续写入及编码吞吐。

## 配置

默认配置见 `config/realtime.json`。实时模式使用 `otsu`：目标只需明显暗于背景，不要求接近纯黑；每帧会根据灰度分布自动适应曝光和增益变化。ROI 限定到 `0,0,720,440`，排除画面底部暗带。曝光、增益和 SDK 缓存节点也可在 UI 开始前修改。运行中点击“计数清零”会清除累计值和现有轨迹，从下一帧重新统计；满料锁存后该按钮无效，必须点击“开始下一批”。

满料 IO 默认配置为 `Line1`、`UserOutput0`、低电平（NPN 下拉）有效。海康相机输出通常不能直接驱动电机或大功率负载，应按相机电气手册通过隔离继电器或 PLC 输入连接。正式接线前应先断开执行机构验证输出极性和失效状态。
