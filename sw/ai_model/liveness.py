#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Liveness 검증 모듈
- 얼굴 랜드마크 기반 라이브니스 판별
- Mediapipe 또는 Dlib 활용
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
import time
from collections import deque

import mediapipe as mp
# import dlib
import os

from common.config import MODEL_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)


class LivenessDetector:
    """얼굴 랜드마크 기반 라이브니스 검증기"""
    
    def __init__(self):
        """초기화"""
        self.config = MODEL_CONFIG["liveness"]
        self.model_type = self.config["model"]  # "mediapipe" or "dlib"
        self.detector = None
        
        # 움직임 추적을 위한 히스토리
        self.landmark_history = deque(maxlen=10)
        
        self._load_detector()
    
    def _load_detector(self):
        """
        얼굴 랜드마크 검출기 로드
        """
        if self.model_type == "mediapipe":
            # TODO(LivenessDetector._load_detector, L43-L49): Mediapipe FaceMesh 초기화
            #  - FaceMesh 파이프라인 생성 및 파라미터 설정 
            #  - detector lifecycle 관리 코드 추가#!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            # FaceMesh 파이프라인 생성 및 파라미터 설정 
            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, # 영상
                max_num_faces=3, # 감지할 얼굴 최대 개수
                refine_landmarks=True, # 세밀 탐지
                min_detection_confidence=0.5, # 감지 신뢰도 임계값
                min_tracking_confidence=0.5 # 추적 신뢰도 임계값
            )

            # detector lifecycle 관리 코드 추가
            if not hasattr(self, "_closeables"):
                self._closeables = []
            self._closeables.append(self.detector)



            # logger.warning(
            #     "LivenessDetector: Mediapipe FaceMesh 초기화가 구현되지 않았습니다. "
            #     "더미 랜드마크를 사용합니다."
            # )
            
            # logger.info("LivenessDetector using Mediapipe (placeholder)")
            
        elif self.model_type == "dlib":
            # TODO(LivenessDetector._load_detector, L51-L56): Dlib 검출기/랜드마크 모델 로딩
            #  - frontal face detector 및 shape predictor 초기화
            #  - 모델 파일 경로 설정 및 예외 처리

            # 모델 파일 경로 설정 및 예외 처리
            model_path = "sw/ai_model/models/shape_predictor_68_face_landmarks.dat"

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Dlib 모델 파일이 없습니다: {model_path}\n"
                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 에서 다운로드 후 저장하세요."
                )
            
            # frontal face detector 및 shape predictor 초기화
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(model_path)
            
            # logger.warning(
            #     "LivenessDetector: Dlib 검출기 로딩이 구현되지 않았습니다. "
            #     "더미 랜드마크를 사용합니다."
            # )
            
            # logger.info("LivenessDetector using Dlib (placeholder)")
        
        else:
            logger.warning(f"Unknown model type: {self.model_type}")
        
        self.detector_loaded = True # !!
    
    def _detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        얼굴 랜드마크 검출
        
        Args:
            frame: 입력 프레임 (H, W, 3) RGB
            
        Returns:
            landmarks: (N, 2) array of (x, y) coordinates, None if not detected
        """
        # TODO(LivenessDetector._detect_landmarks, L72-L88): 실제 랜드마크 검출 구현
        #  - detector/process 호출 후 (N,2) 좌표 배열 반환
        #  - 얼굴 미검출 시 None 반환 처리


        # logger.warning(
        #     "LivenessDetector._detect_landmarks: 실제 랜드마크 검출 대신 더미 값을 반환합니다."
        # )
        
        if self.model_type == "mediapipe":
            try:
                # detector/process 호출
                results = self.detector.process(frame)

                # 미검출 시 None 반환
                if not results.multi_face_landmarks:
                    return None


                height, width, _ = frame.shape
                all_landmarks = []

                for face_landmarks in results.multi_face_landmarks:
                    # 각 얼굴의 (N, 2) 좌표 생성
                    landmarks = np.array(
                        [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark],
                        dtype=np.float32
                    )
                    all_landmarks.append(landmarks)

                # 여러 얼굴의 좌표를 하나로 합친 (M, N, 2) 배열 반환
                return np.array(all_landmarks, dtype=np.float32)


            # 오류 처리
            except Exception as e:
                logger.error(f"Mediapipe landmarks 검출 실패: {e}")
                return None
        
        elif self.model_type == "dlib":
            try:
                # detector/process 호출
                faces = self.detector(frame, 1)
                if len(faces) == 0:
                    return None

                all_landmarks = []

                for face in faces:
                    # 각 얼굴 내 픽셀 좌표 추출 및 저장
                    shape = self.predictor(frame, face)
                    landmarks = np.array(
                        [(p.x, p.y) for p in shape.parts()],
                        dtype=np.float32
                    )
                    all_landmarks.append(landmarks)

                return np.array(all_landmarks, dtype=np.float32)

            except Exception as e:
                logger.error(f"Dlib 랜드마크 검출 실패: {e}")
                return None
        
        # 임시: 더미 랜드마크 (50% 확률로 검출)
        if np.random.rand() > 0.5:
            # 68개 랜드마크 더미 생성
            landmarks = np.random.rand(68, 2) * [frame.shape[1], frame.shape[0]]
            return landmarks
        
        return None
    
    def _calculate_movement(self, current_landmarks: np.ndarray) -> float:
        """
        이전 프레임 대비 움직임 계산
        
        Args:
            current_landmarks: 현재 랜드마크 (N, 2)
            
        Returns:
            movement_score: 움직임 점수 (0~1)
        """
        # TODO(LivenessDetector._calculate_movement, L100-L109): 랜드마크 히스토리 기반 움직임 산출
        #  - 이전 프레임 대비 이동 거리 정규화
        #  - 스푸핑 감지 기준선 정의
        logger.warning(
            "LivenessDetector._calculate_movement: 움직임 점수를 랜덤으로 생성합니다."
        )
        
        if len(self.landmark_history) < 2:
            return 0.5  # 초기에는 중립
        
        # 임시: 랜덤 움직임 점수
        movement = np.random.rand() * 0.1
        
        return movement
    
    def _check_3d_consistency(self, landmarks: np.ndarray) -> bool:
        """
        3D 일관성 검사 (평면 얼굴 vs 입체 얼굴)
        
        Args:
            landmarks: 얼굴 랜드마크 (N, 2)
            
        Returns:
            3D 일관성 통과 여부
        """
        # TODO(LivenessDetector._check_3d_consistency, L122-L126): 3D 얼굴 구조 일관성 평가
        #  - z 좌표 활용 또는 3D 재구성 기반 판별
        logger.warning(
            "LivenessDetector._check_3d_consistency: 3D 일관성 검증이 구현되지 않아 "
            "랜덤 통과 여부를 반환합니다."
        )
        
        # 임시: 랜덤 (80% 통과)
        return np.random.rand() > 0.2
    
    def verify(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        라이브니스 검증 수행
        
        Args:
            frame: 입력 프레임 (H, W, 3) RGB
            
        Returns:
            (is_live, result_dict)
            - is_live: 실제 사람인지 여부
            - result_dict: 검증 결과 상세 정보
        """
        start_time = time.time()
        
        result = {
            "is_live": False,
            "face_detected": False,
            "movement_score": 0.0,
            "3d_consistent": False,
            "inference_time": 0.0
        }
        
        try:
            # 1. 얼굴 랜드마크 검출
            landmarks = self._detect_landmarks(frame)
            
            if landmarks is None:
                logger.debug("No face detected for liveness check")
                return False, result
            
            result["face_detected"] = True
            
            # 2. 움직임 분석
            movement = self._calculate_movement(landmarks)
            result["movement_score"] = float(movement)
            
            # 랜드마크 히스토리 업데이트
            self.landmark_history.append(landmarks.copy())
            
            # 3. 3D 일관성 검사
            is_3d_consistent = self._check_3d_consistency(landmarks)
            result["3d_consistent"] = is_3d_consistent
            
            # 4. 라이브니스 판정
            movement_threshold = self.config["movement_threshold"]
            
            # TODO(LivenessDetector.verify, L174-L176): 판정 로직 고도화
            #  - 눈 깜빡임, 표정 변화, 움직임 임계값 조합 규칙 정의
            logger.warning(
                "LivenessDetector.verify: 간단한 임계값 기반 규칙만 적용됩니다."
            )
            
            
            # 임시: 간단한 규칙
            is_live = (
                result["face_detected"] and
                movement >= movement_threshold and
                is_3d_consistent
            )
            
            result["is_live"] = is_live
            
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            
            if is_live:
                logger.debug(f"Liveness verified (movement: {movement:.3f}, {inference_time*1000:.1f}ms)")
            else:
                logger.debug(f"Liveness failed (movement: {movement:.3f})")
            
            return is_live, result
            
        except Exception as e:
            logger.error(f"Liveness verification error: {e}")
            result["error"] = str(e)
            return False, result
    
    def reset(self):
        """히스토리 초기화"""
        self.landmark_history.clear()
        logger.debug("Landmark history cleared")
    
    def close(self):
        """detector 리소스 해제"""
        if hasattr(self, "_closeables"):
            for c in self._closeables:
                if hasattr(c, "close"):
                    c.close()
        self._closeables.clear()



if __name__ == "__main__":
    # 테스트 코드
    print("Testing LivenessDetector...")
    
    detector = LivenessDetector()
    
    # 더미 프레임
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for i in range(5):
        is_live, result = detector.verify(frame)
        print(f"Frame {i+1} - Live: {is_live}, Result: {result}")
