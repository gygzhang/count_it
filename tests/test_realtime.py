import time
import unittest

import cv2
import numpy as np

from realtime.camera_base import CameraSource
from realtime.config import AppConfig
from realtime.counter import StreamingCounter
from realtime.service import MeasurementService
from realtime.types import CameraDevice, FramePacket


def moving_frames(background=50, object_level=0):
    frames = []
    for x in range(-160, 760, 10):
        frame = np.full((540, 720), background, dtype=np.uint8)
        cv2.rectangle(frame, (x, 170), (x + 140, 310), object_level, -1)
        frames.append(frame)
    return frames


class FakeCamera(CameraSource):
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.running = False
        self.output_history = []

    def enumerate_devices(self):
        return [CameraDevice("fake", "Fake", "1", "MEM")]

    def open(self, device_id):
        self.index = 0

    def start(self):
        self.running = True

    def read(self, timeout_ms):
        if not self.running or self.index >= len(self.frames):
            time.sleep(0.001)
            return None
        packet = FramePacket(self.frames[self.index], self.index, time.perf_counter_ns())
        self.index += 1
        return packet

    def stop(self):
        self.running = False

    def close(self):
        self.running = False

    def set_digital_output(self, active):
        self.output_history.append(bool(active))
        return True


class RealtimeTests(unittest.TestCase):
    def test_streaming_counter_counts_one_crossing(self):
        counter = StreamingCounter(AppConfig().counting)
        for index, frame in enumerate(moving_frames()):
            counter.process(frame, index)
        self.assertEqual(counter.count, 1)

    def test_otsu_adapts_to_bright_background_and_gray_object(self):
        counter = StreamingCounter(AppConfig().counting)
        for index, frame in enumerate(moving_frames(245, 105)):
            counter.process(frame, index)
        self.assertEqual(counter.count, 1)

    def test_counter_can_be_cleared(self):
        counter = StreamingCounter(AppConfig().counting)
        for index, frame in enumerate(moving_frames()):
            counter.process(frame, index)
        self.assertEqual(counter.count, 1)
        counter.clear_count()
        self.assertEqual(counter.count, 0)

    def test_measurement_service_pipeline(self):
        frames = moving_frames()
        config = AppConfig()
        config.camera.processing_queue_size = 128
        service = MeasurementService(lambda: FakeCamera(frames), config)
        service.start("fake")
        deadline = time.time() + 5
        while service.snapshot().processed < len(frames) and time.time() < deadline:
            time.sleep(0.01)
        service.stop()
        stats = service.snapshot()
        self.assertEqual(stats.acquired, len(frames))
        self.assertEqual(stats.processed, len(frames))
        self.assertEqual(stats.processing_queue_drops, 0)
        self.assertEqual(stats.count, 1)

    def test_full_bin_latches_output_and_next_batch_releases_it(self):
        frames = moving_frames()
        config = AppConfig()
        config.full_bin.enabled = True
        config.full_bin.target_count = 1
        cameras = []

        def factory():
            camera = FakeCamera(frames)
            cameras.append(camera)
            return camera

        service = MeasurementService(factory, config)
        service.start("fake")
        deadline = time.time() + 5
        while not service.snapshot().full_bin and time.time() < deadline:
            time.sleep(0.01)
        stats = service.snapshot()
        self.assertTrue(stats.full_bin)
        self.assertTrue(stats.counting_paused)
        self.assertTrue(stats.io_output_active)
        self.assertEqual(stats.count, 1)
        self.assertFalse(service.reset_count())
        self.assertEqual(cameras[-1].output_history, [False, True])

        service.start_next_batch(2)
        stats = service.snapshot()
        self.assertFalse(stats.full_bin)
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.full_bin_target, 2)
        self.assertEqual(cameras[-1].output_history, [False, True, False])
        service.stop()

    def test_standalone_line1_wiring_test(self):
        config = AppConfig()
        config.full_bin.enabled = False
        cameras = []

        def factory():
            camera = FakeCamera([])
            cameras.append(camera)
            return camera

        service = MeasurementService(factory, config)
        self.assertTrue(service.set_line1_test_output("fake", True))
        self.assertTrue(service.line1_test_active())
        self.assertFalse(service.snapshot().running)
        self.assertEqual(service.snapshot().count, 0)
        self.assertFalse(service.set_line1_test_output("fake", False))
        self.assertFalse(service.line1_test_active())
        self.assertEqual(cameras[-1].output_history, [True, False])
        self.assertFalse(cameras[-1].running)


if __name__ == "__main__":
    unittest.main()
