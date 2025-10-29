#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캡처 및 전처리 모듈 테스트
- 카메라 소스 테스트
- 전처리 테스트
- 파이프라인 통합 테스트
- 구현 완료
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import time
import unittest

from capture.camera_source import (
    DummySource,
    VideoFileSource,
    create_camera_source
)
from capture.preprocessor import FramePreprocessor
from capture.frame_capture import FrameCaptureSystem

from common.logger import get_logger

logger = get_logger(__name__)


class TestCameraSources(unittest.TestCase):
    """카메라 소스 테스트"""
    
    def test_dummy_source(self):
        """더미 소스 테스트"""
        source = DummySource()
        
        # 열기
        self.assertTrue(source.open())
        self.assertTrue(source.is_opened())
        
        # 프레임 읽기
        for i in range(10):
            ret, frame = source.read()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape[2], 3)  # RGB
            logger.info(f"Frame {i+1}: {frame.shape}")
        
        # 닫기
        source.release()
        self.assertFalse(source.is_opened())
    
    def test_camera_factory(self):
        """카메라 팩토리 테스트"""
        # 더미 소스 생성
        source = create_camera_source("dummy")
        self.assertIsInstance(source, DummySource)
        
        # 잘못된 타입
        with self.assertRaises(ValueError):
            create_camera_source("invalid_type")


class TestPreprocessor(unittest.TestCase):
    """전처리 모듈 테스트"""
    
    def setUp(self):
        self.preprocessor = FramePreprocessor()
    
    def test_resize(self):
        """리사이즈 스켈레톤 테스트"""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        resized = self.preprocessor.resize(frame)
        self.assertTrue(np.array_equal(resized, frame))
    
    def test_normalize(self):
        """정규화 스켈레톤 테스트"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        normalized = self.preprocessor.normalize(frame, method="standard")
        self.assertTrue(np.array_equal(normalized, frame))
    
    def test_denoise(self):
        """노이즈 제거 스켈레톤 테스트"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        denoised = self.preprocessor.denoise(frame, method="gaussian")
        self.assertTrue(np.array_equal(denoised, frame))
    
    def test_enhance_contrast(self):
        """대비 향상 스켈레톤 테스트"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        enhanced = self.preprocessor.enhance_contrast(frame, method="clahe")
        self.assertTrue(np.array_equal(enhanced, frame))
    
    def test_preprocessing_pipeline(self):
        """전처리 파이프라인 스켈레톤 테스트"""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        start_time = time.time()
        processed = self.preprocessor.process(
            frame,
            resize=True,
            normalize=True,
            denoise=False,
            enhance=False
        )
        process_time = time.time() - start_time
        
        logger.info(f"Preprocessing time: {process_time*1000:.2f}ms")
        
        self.assertTrue(np.array_equal(processed, frame))


class TestFrameCaptureSystem(unittest.TestCase):
    """프레임 캡처 시스템 통합 테스트"""
    
    def test_system_initialization(self):
        """시스템 초기화 테스트"""
        system = FrameCaptureSystem(source_type="dummy")
        system.setup()
        
        self.assertIsNotNone(system.camera)
        self.assertTrue(system.camera.is_opened())
        
        system.teardown()
    
    def test_single_thread_capture(self):
        """단일 스레드 캡처 테스트"""
        system = FrameCaptureSystem(source_type="dummy")
        system.setup()
        
        # 5초간 실행
        logger.info("Running single-thread capture for 5 seconds...")
        system.run_single_thread(duration=5.0)
        
        system.teardown()
    
    def test_performance_monitoring(self):
        """성능 모니터링 테스트"""
        from common.frame_buffer import PerformanceMonitor
        
        monitor = PerformanceMonitor(window_size=30)
        
        # 더미 처리 시간 기록
        for i in range(50):
            process_time = 0.033 + np.random.randn() * 0.005  # ~30 FPS
            monitor.update(process_time)
            
            if i % 10 == 0:
                fps = monitor.get_fps()
                latency = monitor.get_avg_latency()
                logger.info(f"Iteration {i}: FPS={fps:.1f}, Latency={latency:.1f}ms")
        
        final_fps = monitor.get_fps()
        self.assertGreater(final_fps, 20.0)  # 최소 20 FPS


class TestSharedBuffer(unittest.TestCase):
    """공유 버퍼 테스트"""
    
    def test_buffer_operations(self):
        """버퍼 기본 동작 테스트"""
        from common.frame_buffer import SharedFrameBuffer
        
        buffer = SharedFrameBuffer(max_size=5)
        
        # 프레임 추가
        for i in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            success = buffer.put_frame(frame, timeout=1.0)
            self.assertTrue(success)
        
        # 프레임 가져오기
        for i in range(3):
            frame = buffer.get_frame(timeout=1.0)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
        
        # 빈 버퍼에서 가져오기 (타임아웃)
        frame = buffer.get_frame(timeout=0.1)
        self.assertIsNone(frame)
        
        buffer.stop()
    
    def test_buffer_overflow(self):
        """버퍼 오버플로우 테스트"""
        from common.frame_buffer import SharedFrameBuffer
        
        buffer = SharedFrameBuffer(max_size=3)
        
        # 버퍼 가득 채우기
        for i in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            success = buffer.put_frame(frame, timeout=0.1)
            self.assertTrue(success)
        
        # 하나 더 추가 시도 (타임아웃)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        success = buffer.put_frame(frame, timeout=0.1)
        self.assertFalse(success)
        
        buffer.stop()


def run_all_tests():
    """모든 테스트 실행"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestCameraSources))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameCaptureSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSharedBuffer))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("캡처 및 전처리 모듈 테스트")
    print("=" * 60)
    
    result = run_all_tests()
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    sys.exit(0 if result.wasSuccessful() else 1)
