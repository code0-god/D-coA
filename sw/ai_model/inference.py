#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 추론 메인 모듈
- Object Detection, Liveness, Deepfake 검증 통합
- analyze(frame) 함수로 단일 pass flag 반환
- 임시 구현 완료
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import time
from typing import Dict, Tuple

from common.config import MODEL_CONFIG
from common.logger import get_logger

# 각 검증 모듈 임포트
from .object_detection import ObjectDetector
from .liveness import LivenessDetector
from .deepfake import DeepfakeDetector

logger = get_logger(__name__)


class MediaIntegrityAnalyzer:
    """미디어 무결성 분석기 - 3가지 검증 통합"""
    
    def __init__(self):
        """초기화: 각 검증 모듈 로드"""
        logger.info("Initializing MediaIntegrityAnalyzer...")
        
        self.object_detector = ObjectDetector()
        self.liveness_detector = LivenessDetector()
        self.deepfake_detector = DeepfakeDetector()
        
        self.frame_count = 0
        self.pass_count = 0
        
        logger.info("MediaIntegrityAnalyzer initialized successfully")
    
    def analyze(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        프레임 분석 - 3가지 검증 수행
        
        Args:
            frame: 입력 프레임 (H, W, 3) RGB format
            
        Returns:
            (pass_flag, analysis_result)
            - pass_flag: 모든 검증 통과 여부 (bool)
            - analysis_result: 각 검증 상세 결과 (dict)
        """
        self.frame_count += 1
        start_time = time.time()
        
        result = {
            "frame_id": self.frame_count,
            "timestamp": start_time,
            "object_detection": {},
            "liveness": {},
            "deepfake": {},
            "pass_flag": False,
            "inference_time": 0.0
        }
        
        try:
            # 1. Object Detection (사람/사물 검출)
            obj_detected, obj_result = self.object_detector.detect(frame)
            result["object_detection"] = obj_result
            
            if not obj_detected:
                logger.debug(f"Frame {self.frame_count}: Object detection failed")
                return False, result
            
            # 2. Liveness 검증 (얼굴 라이브니스)
            is_live, liveness_result = self.liveness_detector.verify(frame)
            result["liveness"] = liveness_result
            
            if not is_live:
                logger.debug(f"Frame {self.frame_count}: Liveness verification failed")
                return False, result
            
            # 3. Deepfake 탐지 (위변조 검사)
            is_real, deepfake_result = self.deepfake_detector.analyze(frame)
            result["deepfake"] = deepfake_result
            
            if not is_real:
                logger.debug(f"Frame {self.frame_count}: Deepfake detected")
                return False, result
            
            # 모든 검증 통과
            result["pass_flag"] = True
            self.pass_count += 1
            
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            
            logger.debug(
                f"Frame {self.frame_count} PASSED "
                f"(inference: {inference_time*1000:.1f}ms)"
            )
            
            return True, result
            
        except Exception as e:
            logger.error(f"Analysis error on frame {self.frame_count}: {e}")
            result["error"] = str(e)
            return False, result
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        return {
            "total_frames": self.frame_count,
            "passed_frames": self.pass_count,
            "pass_rate": self.pass_count / self.frame_count if self.frame_count > 0 else 0.0
        }
    
    def reset(self):
        """카운터 초기화"""
        self.frame_count = 0
        self.pass_count = 0
        logger.info("Statistics reset")


# 편의 함수
_analyzer = None

def setup():
    """전역 분석기 초기화"""
    global _analyzer
    if _analyzer is None:
        _analyzer = MediaIntegrityAnalyzer()
    return _analyzer

def analyze(frame: np.ndarray) -> Tuple[bool, Dict]:
    """
    프레임 분석 - 외부 인터페이스
    
    Args:
        frame: 입력 프레임 (H, W, 3)
        
    Returns:
        (pass_flag, result_dict)
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = setup()
    return _analyzer.analyze(frame)

def get_statistics() -> Dict:
    """통계 정보 반환"""
    global _analyzer
    if _analyzer is None:
        return {"error": "Analyzer not initialized"}
    return _analyzer.get_statistics()

def teardown():
    """리소스 해제"""
    global _analyzer
    if _analyzer is not None:
        logger.info("Analyzer teardown")
        _analyzer = None


if __name__ == "__main__":
    # 간단한 테스트
    print("Testing MediaIntegrityAnalyzer...")
    
    # 더미 프레임으로 테스트
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    analyzer = setup()
    pass_flag, result = analyze(dummy_frame)
    
    print(f"Pass flag: {pass_flag}")
    print(f"Result: {result}")
    print(f"Statistics: {get_statistics()}")
    
    teardown()