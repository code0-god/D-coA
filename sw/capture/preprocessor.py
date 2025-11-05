#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
전처리 스켈레톤 모듈
- 임시로 작성된 전처리 알고리즘들
- 필요에 따라 적용 및 구현해야 함
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from common.config import PROCESSING_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)


class FramePreprocessor:
    """프레임 전처리 스켈레톤"""

    def __init__(self):
        self.config = PROCESSING_CONFIG["preprocessing"]
        logger.info("FramePreprocessor initialized (skeleton mode)")

    def resize(self, frame: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        프레임 크기 조정 스켈레톤
        TODO(FramePreprocessor.resize, L25-L31): CONFIG 기반 리사이즈 옵션(종횡비 유지, 패딩, 보간법)을 구현한다.
        """
        # 1. 목표 크기 결정 (CONFIG 또는 기본값 640x640 사용)
        target_size = size if size else (self.config.get("target_width", 640), self.config.get("target_height", 640))
        target_w, target_h = target_size
        
        h, w = frame.shape[:2]
        
        # 2. 종횡비 계산 및 비율 결정
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 3. 이미지 리사이징 (종횡비 유지된 채 리사이징)
        try:
            # INTER_AREA 보간법 사용
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            logger.error(f"Resize failed: {e}")
            return frame
            
        # 4. 패딩 적용 (남는 공간을 채워서 최종 크기를 맞춥니다.)
        # 패딩할 크기 계산
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        # 패딩을 위/아래, 왼쪽/오른쪽에 균등하게 분배
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        # 패딩을 적용하여 최종 프레임 생성 (cv2.BORDER_CONSTANT로 검은색(0) 채우기)
        padded_frame = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
        # 종횡비 유지와 패딩이 적용된 최종 결과 반환
        return padded_frame

    def normalize(self, frame: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        프레임 정규화 스켈레톤
        TODO(FramePreprocessor.normalize, L35-L41): 정규화 모드별(float 변환, 채널별 파라미터) 로직을 적용한다.
        """
        # 1. 정규화를 위해 데이터 타입을 float로 변경
        normalized_frame = frame.astype(np.float32)
        
        # 2. 'standard' 모드 (0-1 정규화) 적용
        if method == "standard":
            # 픽셀 값을 255.0으로 나누어 범위를 0.0 ~ 1.0으로 만듭니다.
            normalized_frame /= 255.0
            
        # 3. '채널별 파라미터 정규화' 모드 적용 
        elif method == "mean_std":
            # 설정 파일에서 채널별 평균과 표준편차 파라미터를 가져옵니다.
            # (예: ImageNet 데이터셋 기준)
            
            # 파라미터가 CONFIG에 정의되어 있다고 가정
            mean = self.config.get("norm_mean", [0.485, 0.456, 0.406])
            std = self.config.get("norm_std", [0.229, 0.224, 0.225])
            
            # BGR 또는 RGB 순서에 맞게 [H, W, C] 형태의 프레임에 적용합니다.
            
            # 3-1. 0-1 정규화 선행 (0-255 -> 0-1)
            normalized_frame /= 255.0
            
            # 3-2. 채널별 평균 빼고 표준편차로 나누기 (Z-score normalization)
            try:
                mean = np.array(mean, dtype=np.float32)
                std = np.array(std, dtype=np.float32)
                
                # 프레임에서 평균을 빼고 표준편차로 나눕니다.
                normalized_frame = (normalized_frame - mean) / std
            except Exception as e:
                logger.error(f"Mean/Std normalization failed: {e}. Falling back to standard.")
                normalized_frame /= 255.0 # 오류 시 기본 정규화로 대체
        
        return normalized_frame
    
    def denoise(self, frame: np.ndarray, method: str = "gaussian") -> np.ndarray: # 임수인 수정
        """
        노이즈 제거 스켈레톤
        TODO(FramePreprocessor.denoise, L43-L49): 노이즈 추정 및 필터별 파라미터 조정 로직을 구현한다.
        """

        try:
            # 노이즈를 제거한다. 기본값 (5, 5), 필요시 조정 (홀수값으로 설정해야 한다.)
            # 수정 필요. 기본값이 아닌 받는 영상마다 노이즈를 추정하여 피라미터값을 제어할 것.
            denoise_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # cv2 관련 에러가 발생했을 때
        except cv2.error as e:
            logger.error(f"Denoise (GaussianBlur) failed: {e}") # logger: 기록을 남깁니다.
            return frame 
        
        return denoise_frame


    def enhance_contrast(self, frame: np.ndarray, method: str = "clahe") -> np.ndarray: # 임수인 수정
        """
        대비 향상 스켈레톤
        TODO(FramePreprocessor.enhance_contrast, L50-L56): CLAHE/히스토그램 평활화 등 대비 향상 전략을 구현한다.
        """
        try:
            # 'clahe' 메소드일 때만 실행
            if method == "clahe":

                # 'CLAHE 필터' 생성 (clipLimit=2.0, tileGridSize=(8, 8)로 세팅, 필요시 변경 또는 자체 피라미터 변경으로 코드를 수정할것.)
                clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                # 컬러 프레임이 들어왔을 때
                if frame.ndim != 3 or frame.shape[2] == 3:
                    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # BGR(컬러) -> YCrCb (밝기+색정보)로 '분리'
                    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_frame) # 채널을 3개로 쪼갭니다: (밝기, 색1, 색2)
                    enhanced_y = clahe_filter.apply(y_channel) # 밝기(y_channel) 채널에 CLAHE 필터 적용
                    merged_ycrcb = cv2.merge([enhanced_y, cr_channel, cb_channel]) # '필터가 적용된 밝기' + '원본 색깔'로 다시 합체
                    enhanced_frame = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR) # BGR(컬러)로 최종 번역
                
                # 흑백 프레임이 들어왔을 때
                elif frame.ndim == 2:
                    enhanced_frame = clahe_filter.apply(frame) # CLAHE 필터 적용

                # 예외처리 (컬러, 흑백이 둘 다 아닐때)
                else:
                    logger.warning(f"CLAHE input has unusual shape {frame.shape}. Skipping.")
                    enhanced_frame = frame 
                
            else:
                # 매소드가 'clahe'가 아니면 일단 원본 그대로 반환
                enhanced_frame = frame
        
        # cv2 관련 에러가 발생했을 때
        except cv2.error as e:
            logger.error(f"Enhance Contrast (CLAHE) failed: {e}") # logger: 기록을 남깁니다.
            return frame 
        
        return enhanced_frame

    def process(
        self,
        frame: np.ndarray,
        resize: bool = True,
        normalize: bool = False,
        denoise: bool = False,
        enhance: bool = False,
    ) -> np.ndarray:
        """
        전처리 파이프라인 스켈레톤
        TODO(FramePreprocessor.process, L56-L67): 단계별 on/off 제어, 의존성, 파라미터 튜닝을 실제 적용한다.
        """
        processed_frame = frame
        
        # 1. 크기 조정 (가장 먼저 실행)
        if resize:
            # self.resize() 함수를 호출하여 크기를 조정합니다.
            processed_frame = self.resize(processed_frame)

        # 2. 정규화, 노이즈 제거, 대비 향상 (TODO가 해결되면 여기에 추가)
        if denoise:
            processed_frame = self.denoise(processed_frame) # L43-L49 TODO
        if enhance:
            processed_frame = self.enhance_contrast(processed_frame) # L50-L56 TODO
        if normalize:
            processed_frame = self.normalize(processed_frame) # L35-L41 TODO

        # 3. 필수: BGR -> RGB 색상 변환 (AI 모델에게 전달하기 전에 반드시 필요)
        # OpenCV는 기본 BGR이지만, AI 모델은 RGB를 원합니다.
        # 프레임에 3채널(색상) 정보가 있을 때만 변환합니다.
        if processed_frame.ndim == 3 and processed_frame.shape[2] == 3:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        return processed_frame
