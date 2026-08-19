import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self, service, config, backend_name="海康 MVS"):
        super().__init__()
        self.service = service
        self.config = config
        self.backend_name = backend_name
        self.devices = []
        self.last_preview_no = None
        self.last_error = ""
        self.stop_thread = None
        self.full_bin_announced = False
        self.setWindowTitle("高速相机实时计数")
        self.resize(1180, 820)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start(33)
        QTimer.singleShot(0, self.refresh_devices)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QGroupBox("测量控制")
        grid = QGridLayout(controls)
        self.device_combo = QComboBox()
        self.refresh_button = QPushButton("刷新相机")
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button = QPushButton("开始测量")
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_measurement)
        self.line1_test_button = QPushButton("Line1 接线测试：输出关闭")
        self.line1_test_button.setEnabled(False)
        self.line1_test_button.clicked.connect(self.toggle_line1_test)

        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(1.0, 1_000_000.0)
        self.exposure.setDecimals(1)
        self.exposure.setSuffix(" μs")
        self.exposure.setValue(self.config.camera.exposure_us)
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.0, 24.0)
        self.gain.setDecimals(1)
        self.gain.setSuffix(" dB")
        self.gain.setValue(self.config.camera.gain_db)
        self.buffer_nodes = QSpinBox()
        self.buffer_nodes.setRange(1, 256)
        self.buffer_nodes.setValue(self.config.camera.buffer_nodes)
        self.full_bin_enabled = QCheckBox("达到满料数量时输出相机 IO 并暂停计数")
        self.full_bin_enabled.setChecked(self.config.full_bin.enabled)
        self.full_bin_count = QSpinBox()
        self.full_bin_count.setRange(1, 1_000_000)
        self.full_bin_count.setValue(self.config.full_bin.target_count)

        self.record_check = QCheckBox("同时录像（MJPG AVI）")
        self.output_dir = QLineEdit(self.config.recording.output_dir)
        browse = QPushButton("选择目录")
        browse.clicked.connect(self.choose_output_dir)

        grid.addWidget(QLabel(f"图像源：{self.backend_name}"), 0, 0)
        grid.addWidget(self.device_combo, 0, 1, 1, 3)
        grid.addWidget(self.refresh_button, 0, 4)
        grid.addWidget(QLabel("曝光"), 1, 0)
        grid.addWidget(self.exposure, 1, 1)
        grid.addWidget(QLabel("增益"), 1, 2)
        grid.addWidget(self.gain, 1, 3)
        grid.addWidget(QLabel("SDK缓存节点"), 1, 4)
        grid.addWidget(self.buffer_nodes, 1, 5)
        grid.addWidget(self.record_check, 2, 0, 1, 2)
        grid.addWidget(self.output_dir, 2, 2, 1, 2)
        grid.addWidget(browse, 2, 4)
        grid.addWidget(self.full_bin_enabled, 3, 0, 1, 3)
        grid.addWidget(QLabel("满料数量"), 3, 3)
        grid.addWidget(self.full_bin_count, 3, 4)
        grid.addWidget(self.line1_test_button, 4, 0, 1, 2)
        grid.addWidget(self.start_button, 4, 3)
        grid.addWidget(self.stop_button, 4, 4)
        layout.addWidget(controls)

        self.full_bin_banner = QLabel()
        self.full_bin_banner.setAlignment(Qt.AlignCenter)
        self.full_bin_banner.setWordWrap(True)
        self.full_bin_banner.setStyleSheet(
            "background:#c62828;color:white;font-size:24px;font-weight:bold;"
            "padding:16px;border:3px solid #ffeb3b"
        )
        self.full_bin_banner.hide()
        layout.addWidget(self.full_bin_banner)

        body = QHBoxLayout()
        self.image_label = QLabel("等待图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(820, 600)
        self.image_label.setStyleSheet("background:#181818;color:#aaa;border:1px solid #444")
        body.addWidget(self.image_label, 1)

        side = QVBoxLayout()
        self.count_label = QLabel("0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet(
            "font-size:72px;font-weight:bold;color:#00b86b;padding:12px"
        )
        side.addWidget(QLabel("累计计数"))
        side.addWidget(self.count_label)
        self.reset_count_button = QPushButton("计数清零")
        self.reset_count_button.clicked.connect(self.reset_count)
        side.addWidget(self.reset_count_button)
        stats_box = QGroupBox("运行状态")
        form = QFormLayout(stats_box)
        self.stat_labels = {}
        for key, title in (
            ("acq_fps", "采集帧率"), ("proc_fps", "处理帧率"),
            ("process_ms", "处理耗时"), ("acquired", "已采集"),
            ("processed", "已处理"), ("camera_gaps", "相机/SDK缺帧"),
            ("queue_drops", "处理队列丢帧"),
            ("record_drops", "录像队列丢帧"), ("queue_depth", "处理队列深度"),
        ):
            label = QLabel("0")
            self.stat_labels[key] = label
            form.addRow(title, label)
        side.addWidget(stats_box)
        self.message = QLabel("就绪")
        self.message.setWordWrap(True)
        side.addWidget(self.message)
        side.addStretch(1)
        body.addLayout(side)
        layout.addLayout(body, 1)
        self.setCentralWidget(root)

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择录像保存目录", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def refresh_devices(self):
        if self.service.snapshot().running:
            return
        self.device_combo.clear()
        try:
            self.devices = self.service.list_devices()
            for device in self.devices:
                self.device_combo.addItem(device.label, device.id)
            self.line1_test_button.setEnabled(bool(self.devices))
            self.message.setText(f"找到 {len(self.devices)} 台图像源")
        except Exception as exc:
            self.message.setText(str(exc))
            QMessageBox.warning(self, "相机枚举失败", str(exc))

    def start_measurement(self):
        stats = self.service.snapshot()
        if stats.running and stats.full_bin:
            self.config.full_bin.target_count = self.full_bin_count.value()
            try:
                self.service.start_next_batch(self.config.full_bin.target_count)
            except Exception as exc:
                QMessageBox.critical(self, "下一批启动失败", str(exc))
                self.message.setText(str(exc))
                return
            self.full_bin_banner.hide()
            self.full_bin_announced = False
            self.start_button.setText("开始测量")
            self.start_button.setEnabled(False)
            self.full_bin_count.setEnabled(False)
            self.reset_count_button.setEnabled(True)
            self.message.setText("下一批已开始：计数从 0 重新开始，满料 IO 已撤销")
            return
        device_id = self.device_combo.currentData()
        if not device_id:
            QMessageBox.warning(self, "无法开始", "请先连接并刷新相机")
            return
        self.config.camera.exposure_us = self.exposure.value()
        self.config.camera.gain_db = self.gain.value()
        self.config.camera.buffer_nodes = self.buffer_nodes.value()
        self.config.full_bin.enabled = self.full_bin_enabled.isChecked()
        self.config.full_bin.target_count = self.full_bin_count.value()
        self.config.recording.output_dir = self.output_dir.text().strip() or "recordings"
        record_path = self.service.default_record_path() if self.record_check.isChecked() else None
        try:
            self.service.start(device_id, record_path)
        except Exception as exc:
            QMessageBox.critical(self, "启动失败", str(exc))
            self.message.setText(str(exc))
            return
        self.last_preview_no = None
        self.last_error = ""
        self.full_bin_announced = False
        self.full_bin_banner.hide()
        self.start_button.setText("开始测量")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.refresh_button.setEnabled(False)
        self.full_bin_count.setEnabled(False)
        self.full_bin_enabled.setEnabled(False)
        self.line1_test_button.setEnabled(False)
        self.message.setText("测量中" + (f"，录像: {record_path}" if record_path else ""))

    def stop_measurement(self):
        if self.stop_thread is not None and self.stop_thread.is_alive():
            return
        self.stop_button.setEnabled(False)
        self.message.setText("正在停止并写完缓存…")
        self.stop_thread = threading.Thread(target=self.service.stop, daemon=True)
        self.stop_thread.start()

    def reset_count(self):
        if not self.service.reset_count():
            self.message.setText("已满料，计数已锁定；请点击“开始下一批”")
            return
        self.count_label.setText("0")
        self.message.setText("累计计数已清零")

    def toggle_line1_test(self):
        device_id = self.device_combo.currentData()
        if not device_id:
            QMessageBox.warning(self, "无法测试", "请先连接并刷新相机")
            return
        desired = not self.service.line1_test_active()
        self.line1_test_button.setEnabled(False)
        try:
            active = self.service.set_line1_test_output(device_id, desired)
        except Exception as exc:
            QMessageBox.critical(self, "Line1 接线测试失败", str(exc))
            self.message.setText(str(exc))
            self.line1_test_button.setEnabled(True)
            return
        self.device_combo.setEnabled(not active)
        self.refresh_button.setEnabled(not active)
        state = "开启" if active else "关闭"
        self.message.setText(f"Line1 接线测试输出已{state}（与计数和满料逻辑无关）")

    def _refresh_status(self):
        stats = self.service.snapshot()
        self.count_label.setText(str(stats.count))
        test_active = self.service.line1_test_active()
        self.line1_test_button.setText(
            "Line1 接线测试：输出开启" if test_active else "Line1 接线测试：输出关闭"
        )
        self.line1_test_button.setStyleSheet(
            "background:#c62828;color:white;font-weight:bold"
            if test_active else ""
        )
        self.line1_test_button.setEnabled(
            not stats.running and bool(self.devices) and self.stop_thread is None
        )
        values = {
            "acq_fps": f"{stats.acquisition_fps:.1f} fps",
            "proc_fps": f"{stats.processing_fps:.1f} fps",
            "process_ms": f"{stats.process_ms:.3f} ms",
            "acquired": str(stats.acquired), "processed": str(stats.processed),
            "camera_gaps": str(stats.camera_frame_gaps),
            "queue_drops": str(stats.processing_queue_drops),
            "record_drops": str(stats.recording_queue_drops),
            "queue_depth": str(stats.processing_queue_depth),
        }
        for key, value in values.items():
            self.stat_labels[key].setText(value)
        latest = self.service.latest_result()
        if latest is not None and latest.frame is not None and latest.frame_no != self.last_preview_no:
            self.last_preview_no = latest.frame_no
            h, w = latest.frame.shape[:2]
            image = QImage(latest.frame.data, w, h, latest.frame.strides[0],
                           QImage.Format_BGR888).copy()
            pixmap = QPixmap.fromImage(image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        if stats.error and stats.error != self.last_error:
            self.last_error = stats.error
            self.message.setText(stats.error)
            QMessageBox.critical(self, "运行错误", stats.error)
        if stats.full_bin:
            io_text = (
                "相机 IO 已输出，外部停止信号已发送"
                if stats.io_output_active else
                f"警告：{stats.io_error or '相机 IO 未输出'}"
            )
            self.full_bin_banner.setText(
                f"已达到满料数量：{stats.count}\n{io_text}\n"
                "计数已暂停，请处理满料后点击“开始下一批”"
            )
            self.full_bin_banner.show()
            self.start_button.setText("开始下一批")
            self.start_button.setEnabled(True)
            self.full_bin_count.setEnabled(True)
            self.reset_count_button.setEnabled(False)
            if not self.full_bin_announced:
                self.full_bin_announced = True
                QApplication.beep()
                self.message.setText(f"满料：计数锁定为 {stats.count}。{io_text}")
        if self.stop_thread is not None and not self.stop_thread.is_alive():
            self.stop_thread = None
            self.start_button.setEnabled(True)
            self.start_button.setText("开始测量")
            self.refresh_button.setEnabled(True)
            self.device_combo.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.full_bin_count.setEnabled(True)
            self.full_bin_enabled.setEnabled(True)
            self.reset_count_button.setEnabled(True)
            self.line1_test_button.setEnabled(bool(self.devices))
            self.full_bin_banner.hide()
            self.full_bin_announced = False
            self.message.setText(
                f"已停止：计数 {stats.count}，采集 {stats.acquired}，处理 {stats.processed}"
            )

    def closeEvent(self, event):
        self.service.stop()
        event.accept()
