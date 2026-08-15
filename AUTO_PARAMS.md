# 无真值自动估参

`auto_params.py` 可从视频或图像帧目录生成 `count_cv.py` 的初始参数，无需先手工测量灰度、面积和帧间位移。

```powershell
python auto_params.py sample_10pct_frames --out auto_params.json
```

它会：均匀抽样画面、识别中心有效区域、从暗目标灰度尾部选择阈值、估计典型连通域面积，并用连续帧估计追踪距离和主运动轴。参数会写到 JSON，终端也会打印可复制的计数命令。

随后用打印出的参数运行 `count_cv.py`，并加上 `--save checked.mp4 --save-fps 25` 查看绿色检测框和黄色计数点。

该功能给出的是无真值的初始值；它不能从视频本身证明计数绝对准确。若已有一小段人工确认的总数，请再用 `tune_params.py` 对初始参数做带真值搜索。
