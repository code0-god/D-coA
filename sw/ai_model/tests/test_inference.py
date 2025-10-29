#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 추론 모듈 테스트
- 각 검증 모듈 단위 테스트
- 통합 추론 테스트
- 성능 벤치마크
- 구현 완료
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import time
import unittest

from ai_model.inference import MediaIntegrityAnalyzer, setup, analyze, teardown
from ai_model.object_detection import ObjectDetector
from ai_model.liveness import LivenessDetector
from ai_model.deepfake import DeepfakeDetector

from common.logger import get_logger

logger = get_logger(__name__)


class TestObjectDetection(unittest.TestCase):
    """Object Detection 모듈 테스트"""
    
    def setUp(self):
        self.detector = ObjectDetector()
    
    def test_detect_dummy_frame(self):
        """더미 프레임 검출 테스트"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detected, result = self.detector.detect(frame)
        
        self.assertIsInstance(detected, bool)
        self.assertIn("detected", result)
        self.assertIn("detections", result)
        self.assertIn("inference_time", result)
    
    def test_multiple_frames(self):
        """연속 프레임 검출 테스트"""
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detected, result = self.detector.detect(frame)
            logger.info(f"Frame {i+1}: detected={detected}")


class TestLivenessDetection(unittest.TestCase):
    """Liveness 검증 모듈 테스트"""
    
    def setUp(self):
        self.detector = LivenessDetector()
    
    def test_verify_dummy_frame(self):
        """더미 프레임 라이브니스 검증"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        is_live, result = self.detector.verify(frame)
        
        self.assertIsInstance(is_live, bool)
        self.assertIn("is_live", result)
        self.assertIn("face_detected", result)
        self.assertIn("movement_score", result)
    
    def test_history_tracking(self):
        """히스토리 추적 테스트"""
        for i in range(15):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            is_live, result = self.detector.verify(frame)
            logger.info(
                f"Frame {i+1}: live={is_live}, "
                f"movement={result.get('movement_score', 0):.3f}"
            )
        
        # 히스토리가 10개로 제한되는지 확인
        self.assertLessEqual(len(self.detector.landmark_history), 10)


class TestDeepfakeDetection(unittest.TestCase):
    """Deepfake 탐지 모듈 테스트"""
    
    def setUp(self):
        self.detector = DeepfakeDetector()
    
    def test_analyze_dummy_frame(self):
        """더미 프레임 딥페이크 분석"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        is_real, result = self.detector.analyze(frame)
        
        self.assertIsInstance(is_real, bool)
        self.assertIn("is_real", result)
        self.assertIn("overall_confidence", result)
        self.assertIn("pixel_analysis", result)
        self.assertIn("compression_check", result)
        self.assertIn("temporal_check", result)
    
    def test_temporal_consistency(self):
        """시간적 일관성 검사 테스트"""
        # 첫 프레임은 히스토리가 없어서 기본 점수
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        is_real1, result1 = self.detector.analyze(frame1)
        
        # 두 번째 프레임부터 시간적 일관성 검사 수행
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        is_real2, result2 = self.detector.analyze(frame2)
        
        self.assertGreater(len(self.detector.frame_history), 0)


class TestIntegratedInference(unittest.TestCase):
    """통합 추론 테스트"""
    
    def setUp(self):
        self.analyzer = setup()
    
    def tearDown(self):
        teardown()
    
    def test_analyze_single_frame(self):
        """단일 프레임 분석 테스트"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pass_flag, result = analyze(frame)
        
        self.assertIsInstance(pass_flag, bool)
        self.assertIn("frame_id", result)
        self.assertIn("pass_flag", result)
        self.assertIn("object_detection", result)
        self.assertIn("liveness", result)
        self.assertIn("deepfake", result)
        self.assertIn("inference_time", result)
        
        logger.info(f"Pass flag: {pass_flag}")
        logger.info(f"Inference time: {result['inference_time']*1000:.1f}ms")
    
    def test_analyze_multiple_frames(self):
        """다중 프레임 분석 테스트"""
        num_frames = 30
        pass_count = 0
        total_time = 0
        
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pass_flag, result = analyze(frame)
            
            if pass_flag:
                pass_count += 1
            
            total_time += result["inference_time"]
        
        # 통계 출력
        from ai_model.inference import get_statistics
        stats = get_statistics()
        
        logger.info(f"Total frames: {stats['total_frames']}")
        logger.info(f"Passed frames: {stats['passed_frames']}")
        logger.info(f"Pass rate: {stats['pass_rate']*100:.1f}%")
        logger.info(f"Avg inference time: {total_time/num_frames*1000:.1f}ms")
        
        self.assertEqual(stats['total_frames'], num_frames)


class TestPerformance(unittest.TestCase):
    """성능 벤치마크 테스트"""
    
    def test_fps_benchmark(self):
        """FPS 벤치마크"""
        analyzer = setup()
        
        num_frames = 100
        start_time = time.time()
        
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pass_flag, result = analyze(frame)
        
        elapsed_time = time.time() - start_time
        fps = num_frames / elapsed_time
        
        logger.info(f"Processed {num_frames} frames in {elapsed_time:.2f}s")
        logger.info(f"Average FPS: {fps:.2f}")
        
        teardown()
        
        # 최소 FPS 요구사항 확인 (목표: 15 fps)
        self.assertGreater(fps, 5.0, "FPS too low")


def run_all_tests():
    """모든 테스트 실행"""
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestObjectDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestLivenessDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDeepfakeDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedInference))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("AI 모델 추론 테스트")
    print("=" * 60)
    
    result = run_all_tests()
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    sys.exit(0 if result.wasSuccessful() else 1)