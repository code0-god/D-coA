#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Detection 모듈
- YOLOv5-nano TFLite 기반 객체 탐지
- COCO 데이터셋 활용 (특히 사람 탐지)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from typing import Tuple, Dict, List
import time

from common.config import MODEL_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)


class ObjectDetector:
    """YOLOv5-nano 기반 객체 탐지기"""
    
    def __init__(self):
        """초기화"""
        self.config = MODEL_CONFIG["yolo"]
        self.model = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()
        
    def _load_model(self):
        """
        TFLite 모델 로드
        """
        model_path = self.config["model_path"]
        
        # TODO(ObjectDetector._load_model, L42-L60): TFLite 인터프리터 로딩
        #  - Interpreter 생성 및 allocate_tensors 호출 흐름 작성
        #  - input/output details 캐싱 및 예외 처리 포함
        logger.warning(
            "ObjectDetector._load_model: TFLite 인터프리터가 아직 연결되지 않았습니다. "
            "더미 경로로 진행합니다."
        )
        
        logger.info(f"ObjectDetector model loading from {model_path} (placeholder)")
        
        # 임시: 모델 로딩 시뮬레이션
        self.model_loaded = False
        if model_path.exists():
            self.model_loaded = True
            logger.info("Model file exists, ready for implementation")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        입력 전처리
        
        Args:
            frame: 원본 프레임 (H, W, 3)
            
        Returns:
            전처리된 텐서
        """
        # TODO(ObjectDetector._preprocess, L66-L84): YOLO 입력 전처리 규격 준수
        #  - target size/letterbox 전략 반영
        #  - 정규화/채널 순서 조정
        #  - 배치 차원 확장
        logger.warning(
            "ObjectDetector._preprocess: 기본 리사이즈/정규화만 수행합니다. "
            "YOLO 규격에 맞는 개선이 필요합니다."
        )
        
        # 임시 구현 (더미)
        input_size = 640
        resized = cv2.resize(frame, (input_size, input_size))
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        
        return input_tensor
    
    def _postprocess(self, outputs: np.ndarray, conf_threshold: float) -> List[Dict]:
        """
        출력 후처리
        
        Args:
            outputs: 모델 출력
            conf_threshold: 신뢰도 임계값
            
        Returns:
            검출된 객체 리스트 [{"class": int, "confidence": float, "bbox": [x, y, w, h]}, ...]
        """
        # TODO(ObjectDetector._postprocess, L90-L107): YOLO 출력 디코딩/NMS 구현
        #  - bounding box 복원 및 confidence 필터
        #  - NMS 후 COCO 클래스 매핑
        logger.warning(
            "ObjectDetector._postprocess: 실제 YOLO 출력 파싱이 구현되지 않았습니다. "
            "빈 결과를 반환합니다."
        )
        
        # 임시 구현 (더미)
        detections = []
        
        return detections
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        객체 탐지 수행 - 최소 구현 완료
        
        Args:
            frame: 입력 프레임 (H, W, 3) RGB
            
        Returns:
            (detection_success, result_dict)
            - detection_success: 타겟 객체 검출 여부
            - result_dict: 검출 결과 상세 정보
        """
        start_time = time.time()
        
        result = {
            "detected": False,
            "detections": [],
            "inference_time": 0.0
        }
        
        try:
            input_tensor = self._preprocess(frame)

            if (
                self.model is None
                or not self.input_details
                or not self.output_details
            ):
                logger.error("ObjectDetector.detect: 모델이 로드되지 않았습니다.")
                result["error"] = "Model not loaded"
                return False, result

            # 2. 추론
            self.model.set_tensor(self.input_details[0]["index"], input_tensor)
            self.model.invoke()
            outputs = self.model.get_tensor(self.output_details[0]["index"])

            # 3. 후처리
            detections = self._postprocess(
                outputs,
                self.config["confidence_threshold"],
            )

            # 4. 타겟 클래스 필터링 (사람 = 0)
            target_classes = self.config["target_classes"]
            filtered = [
                det for det in detections if det.get("class") in target_classes
            ]

            result["detections"] = filtered
            result["detected"] = len(filtered) > 0
            
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            
            if result["detected"]:
                logger.debug(f"Objects detected: {len(result['detections'])} ({inference_time*1000:.1f}ms)")
            
            return result["detected"], result
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            result["error"] = str(e)
            return False, result


if __name__ == "__main__":
    # 테스트 코드
    print("Testing ObjectDetector...")
    
    detector = ObjectDetector()
    
    # 더미 프레임
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    detected, result = detector.detect(frame)
    print(f"Detected: {detected}")
    print(f"Result: {result}")
