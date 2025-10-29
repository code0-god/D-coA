#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전처리 스켈레톤 모듈
- 실제 전처리 로직은 TODO 형태로 남겨 두고, 기본 pass-through만 제공한다.
"""

import numpy as np
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
        return frame

    def normalize(self, frame: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        프레임 정규화 스켈레톤
        TODO(FramePreprocessor.normalize, L35-L41): 정규화 모드별(float 변환, 채널별 파라미터) 로직을 적용한다.
        """
        return frame

    def denoise(self, frame: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        노이즈 제거 스켈레톤
        TODO(FramePreprocessor.denoise, L43-L49): 노이즈 추정 및 필터별 파라미터 조정 로직을 구현한다.
        """
        return frame

    def enhance_contrast(self, frame: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        대비 향상 스켈레톤
        TODO(FramePreprocessor.enhance_contrast, L50-L56): CLAHE/히스토그램 평활화 등 대비 향상 전략을 구현한다.
        """
        return frame

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
        return frame
