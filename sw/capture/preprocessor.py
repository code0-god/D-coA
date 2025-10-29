#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 기반 전처리 모듈
- 프레임 리사이즈, 색상 변환, 필터링 등
- Python 우선, 추후 C++ 최적화 가능
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from typing import Optional, Tuple

from common.config import PROCESSING_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)


class FramePreprocessor:
    """프레임 전처리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = PROCESSING_CONFIG["preprocessing"]
        self.target_size = self.config["target_size"]
        logger.info(f"FramePreprocessor initialized (target_size: {self.target_size})")
    
    def resize(self, frame: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        프레임 크기 조정
        
        Args:
            frame: 입력 프레임 (H, W, C)
            size: 타겟 크기 (width, height), None이면 config 사용
            
        Returns:
            리사이즈된 프레임
        """
        # TODO(resize, L43-L52): 고급 리사이즈 옵션 구현
        #  - interpolation 파라미터로 보간법 선택 지원
        #  - keep_aspect/padding 옵션을 추가해 종횡비 유지 처리
        
        
        if size is None:
            size = self.target_size
        
        try:
            resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
            return resized
        except Exception as e:
            logger.error(f"Resize error: {e}")
            return frame
    
    def normalize(self, frame: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        프레임 정규화
        
        Args:
            frame: 입력 프레임 (H, W, C)
            method: 정규화 방법
                - "standard": [0, 255] -> [0, 1]
                - "centered": [0, 255] -> [-1, 1]
                - "zscore": mean=0, std=1
        
        Returns:
            정규화된 프레임
        """
        # TODO(normalize, L72-L100): 정규화 모드 확장
        #  - ImageNet 평균/표준편차 기반 모드 추가
        #  - 채널별 개별 정규화 파라미터 지원
        
        
        try:
            if method == "standard":
                # [0, 255] -> [0, 1]
                normalized = frame.astype(np.float32) / 255.0
            
            elif method == "centered":
                # [0, 255] -> [-1, 1]
                normalized = (frame.astype(np.float32) / 127.5) - 1.0
            
            elif method == "zscore":
                # Z-score normalization
                mean = np.mean(frame, axis=(0, 1), keepdims=True)
                std = np.std(frame, axis=(0, 1), keepdims=True)
                normalized = (frame.astype(np.float32) - mean) / (std + 1e-7)
            
            else:
                logger.warning(f"Unknown normalization method: {method}, using standard")
                normalized = frame.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            return frame.astype(np.float32) / 255.0
    
    def denoise(self, frame: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        노이즈 제거
        
        Args:
            frame: 입력 프레임
            method: 노이즈 제거 방법
                - "gaussian": 가우시안 블러
                - "bilateral": 양방향 필터 (엣지 보존)
                - "nlm": Non-Local Means
        
        Returns:
            노이즈 제거된 프레임
        """
        # TODO(denoise, L116-L142): 적응형 노이즈 제거 구현
        #  - 입력 프레임에서 노이즈 레벨 추정 로직 작성
        #  - method별 파라미터를 자동 조정하도록 개선
        
        
        try:
            if method == "gaussian":
                denoised = cv2.GaussianBlur(frame, (5, 5), 0)
            
            elif method == "bilateral":
                denoised = cv2.bilateralFilter(frame, 9, 75, 75)
            
            elif method == "nlm":
                if len(frame.shape) == 3:
                    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
                else:
                    denoised = cv2.fastNlMeansDenoising(frame, None, 10, 7, 21)
            
            else:
                logger.warning(f"Unknown denoise method: {method}, skipping")
                denoised = frame
            
            return denoised
            
        except Exception as e:
            logger.error(f"Denoise error: {e}")
            return frame
    
    def enhance_contrast(self, frame: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        대비 향상
        
        Args:
            frame: 입력 프레임
            method: 대비 향상 방법
                - "clahe": Contrast Limited Adaptive Histogram Equalization
                - "histogram": 히스토그램 평활화
        
        Returns:
            대비 향상된 프레임
        """
        # TODO(enhance_contrast, L157-L197): 대비 향상 고도화
        #  - 채널별 처리 및 로컬 대비 조절 파라미터 추가
        
        
        try:
            if method == "clahe":
                # CLAHE는 grayscale에 적용
                if len(frame.shape) == 3:
                    # RGB to LAB
                    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # L 채널에 CLAHE 적용
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    
                    # 다시 합치기
                    lab = cv2.merge([l, a, b])
                    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(frame)
            
            elif method == "histogram":
                if len(frame.shape) == 3:
                    # 채널별 히스토그램 평활화
                    channels = cv2.split(frame)
                    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
                    enhanced = cv2.merge(eq_channels)
                else:
                    enhanced = cv2.equalizeHist(frame)
            
            else:
                logger.warning(f"Unknown enhance method: {method}, skipping")
                enhanced = frame
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Contrast enhancement error: {e}")
            return frame
    
    def color_convert(self, frame: np.ndarray, conversion: str) -> np.ndarray:
        """
        색상 공간 변환
        
        Args:
            frame: 입력 프레임
            conversion: 변환 타입 (예: "RGB2GRAY", "RGB2HSV")
        
        Returns:
            변환된 프레임
        """
        try:
            conversion_code = getattr(cv2, f"COLOR_{conversion}", None)
            if conversion_code is None:
                logger.warning(f"Unknown conversion: {conversion}")
                return frame
            
            converted = cv2.cvtColor(frame, conversion_code)
            return converted
            
        except Exception as e:
            logger.error(f"Color conversion error: {e}")
            return frame
    
    def process(
        self,
        frame: np.ndarray,
        resize: bool = True,
        normalize: bool = False,
        denoise: bool = False,
        enhance: bool = False
    ) -> np.ndarray:
        """
        전처리 파이프라인
        
        Args:
            frame: 입력 프레임
            resize: 리사이즈 수행 여부
            normalize: 정규화 수행 여부
            denoise: 노이즈 제거 수행 여부
            enhance: 대비 향상 수행 여부
        
        Returns:
            전처리된 프레임
        """
        processed = frame.copy()
        
        # TODO(process, L246-L260): 전처리 파이프라인 재구성
        #  - CONFIG의 단계별 on/off 제어와 의존성 정의
        # - 각 단계의 필요성 및 순서 검토
        # - 파라미터 튜닝
        
        try:
            if resize and self.config["resize"]:
                processed = self.resize(processed)
            
            if denoise:
                processed = self.denoise(processed, method="gaussian")
            
            if enhance:
                processed = self.enhance_contrast(processed, method="clahe")
            
            if normalize and self.config["normalize"]:
                processed = self.normalize(processed, method="standard")
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline error: {e}")
            return frame


if __name__ == "__main__":
    # 테스트 코드
    print("Testing FramePreprocessor...")
    
    preprocessor = FramePreprocessor()
    
    # 더미 프레임
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"Original frame: {frame.shape}")
    
    # 전처리
    processed = preprocessor.process(
        frame,
        resize=True,
        normalize=True,
        denoise=False,
        enhance=False
    )
    print(f"Processed frame: {processed.shape}, dtype: {processed.dtype}")
    
    # 개별 테스트
    resized = preprocessor.resize(frame, (640, 480))
    print(f"Resized: {resized.shape}")
    
    normalized = preprocessor.normalize(frame, method="standard")
    print(f"Normalized: min={normalized.min():.3f}, max={normalized.max():.3f}")
