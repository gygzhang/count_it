"""Measurement orchestration, independent from camera SDK and UI toolkit."""
import copy
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from .counter import StreamingCounter
from .recorder import AsyncRecorder
from .types import RuntimeStats


class MeasurementService:
    def __init__(self, camera_factory, config):
        self.camera_factory = camera_factory
        self.config = config
        self.camera = None
        self.io_test_camera = None
        self._io_test_active = False
        self.counter = None
        self.recorder = None
        self.processing_queue = None
        self.stop_event = threading.Event()
        self.acquisition_done = threading.Event()
        self.reset_count_event = threading.Event()
        self.counting_paused = threading.Event()
        self._resume_after_ns = 0
        self.acquire_thread = None
        self.process_thread = None
        self._lock = threading.Lock()
        self._stats = RuntimeStats()
        self._latest = None
        self._acq_times = deque()
        self._proc_times = deque()

    def list_devices(self):
        camera = self.camera_factory()
        try:
            return camera.enumerate_devices()
        finally:
            camera.close()

    def start(self, device_id, record_path=None):
        self.close_line1_test()
        with self._lock:
            if self._stats.running:
                raise RuntimeError("测量已经在运行")
        self.stop_event.clear()
        self.acquisition_done.clear()
        self.reset_count_event.clear()
        self.counting_paused.clear()
        self.processing_queue = queue.Queue(
            maxsize=int(self.config.camera.processing_queue_size)
        )
        self.counter = StreamingCounter(self.config.counting)
        self.camera = self.camera_factory()
        self.recorder = None
        self._latest = None
        self._acq_times.clear()
        self._proc_times.clear()
        with self._lock:
            self._stats = RuntimeStats(
                running=False,
                started_ns=time.time_ns(),
                full_bin_target=(
                    int(self.config.full_bin.target_count)
                    if self.config.full_bin.enabled else 0
                ),
            )
        try:
            self.camera.open(device_id)
            if self.config.full_bin.enabled:
                supported = self.camera.set_digital_output(False)
                if not supported:
                    with self._lock:
                        self._stats.extra["io_warning"] = (
                            "当前图像源不支持相机数字输出；满料时仍会暂停计数"
                        )
            self.camera.start()
            if record_path:
                self.recorder = AsyncRecorder(
                    record_path,
                    fps=self.config.recording.nominal_fps,
                    codec=self.config.recording.codec,
                    queue_size=self.config.recording.queue_size,
                )
                self.recorder.start()
            with self._lock:
                self._stats.running = True
                warnings = getattr(self.camera, "warnings", [])
                if warnings:
                    self._stats.extra["camera_warnings"] = list(warnings)
            self.acquire_thread = threading.Thread(
                target=self._acquire_loop, name="camera-acquisition", daemon=True
            )
            self.process_thread = threading.Thread(
                target=self._process_loop, name="frame-processing", daemon=True
            )
            self.acquire_thread.start()
            self.process_thread.start()
        except Exception:
            try:
                self.camera.close()
            finally:
                self.camera = None
            raise

    def default_record_path(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path(self.config.recording.output_dir) / f"measurement_{stamp}.avi")

    @staticmethod
    def _rate(times, now):
        times.append(now)
        edge = now - 1_000_000_000
        while times and times[0] < edge:
            times.popleft()
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) * 1e9 / (times[-1] - times[0])

    def _record_frame_gap(self, frame_no):
        previous = self._stats.last_frame_no
        if previous is not None:
            if frame_no > previous:
                self._stats.camera_frame_gaps += max(0, frame_no - previous - 1)
            elif previous - frame_no < 0x7FFFFFFF:
                # A decreasing number generally means a source restart, not wraparound.
                self._stats.extra["frame_number_resets"] = (
                    self._stats.extra.get("frame_number_resets", 0) + 1
                )
        self._stats.last_frame_no = frame_no

    def _acquire_loop(self):
        try:
            while not self.stop_event.is_set():
                packet = self.camera.read(self.config.camera.read_timeout_ms)
                if packet is None:
                    continue
                now = time.perf_counter_ns()
                with self._lock:
                    self._stats.acquired += 1
                    self._record_frame_gap(packet.frame_no)
                    self._stats.acquisition_fps = self._rate(self._acq_times, now)
                if self.recorder is not None:
                    self.recorder.submit(packet)
                try:
                    self.processing_queue.put_nowait(packet)
                except queue.Full:
                    # Keep latency bounded. The dropped-frame counter makes this visible;
                    # sustained drops mean ROI/scale or camera rate must be reduced.
                    try:
                        self.processing_queue.get_nowait()
                    except queue.Empty:
                        pass
                    with self._lock:
                        self._stats.processing_queue_drops += 1
                    self.processing_queue.put_nowait(packet)
        except Exception as exc:
            self._set_error(f"采集线程: {exc}")
            self.stop_event.set()
        finally:
            self.acquisition_done.set()

    def _process_loop(self):
        preview_period_ns = int(1e9 / max(self.config.ui.preview_fps, 1.0))
        next_preview_ns = 0
        try:
            while not self.acquisition_done.is_set() or not self.processing_queue.empty():
                try:
                    packet = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.reset_count_event.is_set():
                    self.counter.clear_count()
                    self.reset_count_event.clear()
                    self.counting_paused.clear()
                    with self._lock:
                        self._stats.count = 0
                        self._stats.full_bin = False
                        self._stats.counting_paused = False
                        self._stats.io_output_active = False
                        self._stats.io_error = ""
                        self._latest = None
                    if packet.captured_ns < self._resume_after_ns:
                        continue
                if self.counting_paused.is_set():
                    now = time.perf_counter_ns()
                    with self._lock:
                        self._stats.processed += 1
                        self._stats.processing_fps = self._rate(self._proc_times, now)
                        self._stats.processing_queue_depth = self.processing_queue.qsize()
                    continue
                if packet.captured_ns < self._resume_after_ns:
                    continue
                now = time.perf_counter_ns()
                annotate = now >= next_preview_ns
                result = self.counter.process(packet.image, packet.frame_no, annotate)
                if annotate:
                    next_preview_ns = now + preview_period_ns
                with self._lock:
                    self._stats.processed += 1
                    self._stats.count = result.count
                    self._stats.process_ms = result.process_ms
                    self._stats.processing_fps = self._rate(self._proc_times, now)
                    self._stats.processing_queue_depth = self.processing_queue.qsize()
                    if annotate:
                        self._latest = result
                if (
                    self.config.full_bin.enabled
                    and result.count >= int(self.config.full_bin.target_count)
                ):
                    self._latch_full_bin(result.count)
        except Exception as exc:
            self._set_error(f"计数线程: {exc}")
            self.stop_event.set()

    def _set_error(self, message):
        with self._lock:
            self._stats.error = message

    def snapshot(self):
        with self._lock:
            stats = copy.deepcopy(self._stats)
        if self.recorder is not None:
            stats.recording_queue_drops = self.recorder.dropped
            stats.extra["recorded_frames"] = self.recorder.written
            if self.recorder.error and not stats.error:
                stats.error = f"录像线程: {self.recorder.error}"
        return stats

    def latest_result(self):
        with self._lock:
            return self._latest

    def reset_count(self):
        """Thread-safe total reset, applied before processing the next frame."""
        if self.counting_paused.is_set():
            return False
        with self._lock:
            self._stats.count = 0
            self._latest = None
        if self.counter is not None:
            self.reset_count_event.set()
        return True

    def set_line1_test_output(self, device_id, active):
        """Standalone wiring test; deliberately independent from measurement state."""
        with self._lock:
            if self._stats.running:
                raise RuntimeError("测量运行期间不能使用 Line1 接线测试")
        if active:
            if self.io_test_camera is None:
                camera = self.camera_factory()
                try:
                    camera.open(device_id)
                    supported = camera.set_digital_output(True)
                    if not supported:
                        raise RuntimeError("当前图像源不支持相机数字输出")
                except Exception:
                    camera.close()
                    raise
                self.io_test_camera = camera
            else:
                supported = self.io_test_camera.set_digital_output(True)
                if not supported:
                    raise RuntimeError("当前图像源不支持相机数字输出")
            self._io_test_active = True
            return True
        self.close_line1_test()
        return False

    def line1_test_active(self):
        return self._io_test_active

    def close_line1_test(self):
        camera = self.io_test_camera
        self.io_test_camera = None
        self._io_test_active = False
        if camera is None:
            return
        try:
            supported = camera.set_digital_output(False)
            if not supported:
                raise RuntimeError("当前图像源不支持相机数字输出")
        finally:
            camera.close()

    def _latch_full_bin(self, count):
        """Latch once: freeze counting and assert the configured camera output."""
        if self.counting_paused.is_set():
            return
        self.counting_paused.set()
        io_active = False
        io_error = ""
        try:
            io_active = bool(self.camera.set_digital_output(True))
            if not io_active:
                io_error = "当前图像源不支持数字输出，未能向外部硬件发送满料信号"
        except Exception as exc:
            io_error = f"满料 IO 输出失败：{exc}"
        with self._lock:
            self._stats.count = count
            self._stats.full_bin = True
            self._stats.counting_paused = True
            self._stats.io_output_active = io_active
            self._stats.io_error = io_error

    def start_next_batch(self, target_count=None):
        """Release the output and resume as a new zero-based batch."""
        with self._lock:
            if not self._stats.running or not self._stats.full_bin:
                raise RuntimeError("当前没有等待复位的满料批次")
        if target_count is not None:
            self.config.full_bin.target_count = max(1, int(target_count))
        try:
            supported = self.camera.set_digital_output(False)
            if not supported:
                with self._lock:
                    self._stats.extra["io_warning"] = "当前图像源不支持相机数字输出"
        except Exception as exc:
            raise RuntimeError(f"撤销满料 IO 输出失败，不能开始下一批：{exc}") from exc
        self._resume_after_ns = time.perf_counter_ns()
        with self._lock:
            self._stats.full_bin_target = int(self.config.full_bin.target_count)
            self._stats.count = 0
            self._stats.full_bin = False
            self._stats.io_output_active = False
            self._stats.io_error = ""
            self._latest = None
        self.reset_count_event.set()

    def stop(self):
        self.stop_event.set()
        if self.acquire_thread is not None:
            self.acquire_thread.join(timeout=2.0)
        if self.camera is not None:
            self.camera.stop()
        if self.acquire_thread is not None and self.acquire_thread.is_alive():
            self.acquire_thread.join(timeout=2.0)
        if self.process_thread is not None:
            self.process_thread.join(timeout=5.0)
        if self.recorder is not None:
            self.recorder.stop()
        if self.camera is not None:
            self.camera.close()
        with self._lock:
            self._stats.running = False
            self._stats.processing_queue_depth = (
                self.processing_queue.qsize() if self.processing_queue else 0
            )
        self.camera = None
        self.close_line1_test()
