#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 임시 구현 완료
"""
카메라 소스 추상화
- 실제 카메라, 더미 파일, 웹 스트리밍 등 다양한 소스 지원
- 개발 환경(aarch64 서버)과 배포 환경(라즈베리파이) 모두 지원
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import time

from common.config import CAMERA_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)


class CameraSource(ABC):
    """카메라 소스 추상 클래스"""
    
    @abstractmethod
    def open(self) -> bool:
        """소스 열기"""
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 읽기"""
        pass
    
    @abstractmethod
    def release(self):
        """소스 해제"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """소스 열림 상태 확인"""
        pass
    
    def get_fps(self) -> float:
        """FPS 반환 (기본 구현)"""
        return CAMERA_CONFIG["fps"]


class PiCameraSource(CameraSource):
    """Picamera2 기반 실제 카메라 (라즈베리파이)"""
    
    def __init__(self):
        self.camera = None
        self.config = CAMERA_CONFIG
        logger.info("PiCameraSource initialized")
    
    def open(self) -> bool:
        """
        Picamera2 초기화
        """
        try:
            # TODO(PiCameraSource.open, L66-L72): Picamera2 장치 초기화/설정/시작 구현
            #  - Picamera2 객체 생성 후 해상도, 포맷, FPS 적용
            #  - start() 호출 전 예외 처리와 로그 추가
            
            logger.info("PiCamera opened placeholder (Picamera2 not implemented)")
            return False  # 실제 구현 전까지 False
            
        except Exception as e:
            logger.error(f"Failed to open PiCamera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        프레임 캡처
        """
        if self.camera is None:
            return False, None
        
        try:
            # TODO(PiCameraSource.read, L85-L90): Picamera2에서 프레임 캡처 후 RGB 변환
            #  - capture_array 호출 결과 검증 및 예외 처리
            #  - 필요 시 컬러 포맷 변환 적용
            
            return False, None
            
        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
            return False, None
    
    def release(self):
        """카메라 해제"""
        if self.camera is not None:
            # TODO(PiCameraSource.release, L98-L104): Picamera2 stop/close 호출 추가
            #  - start 상태 여부 확인 후 stop() 실행
            #  - 리소스 해제 실패 시 경고 로그 남기기
            
            self.camera = None
            logger.info("PiCamera released")
    
    def is_opened(self) -> bool:
        return self.camera is not None


class VideoFileSource(CameraSource):
    """비디오 파일 소스 (테스트용 더미)"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.fps = CAMERA_CONFIG["fps"]
        logger.info(f"VideoFileSource initialized: {video_path}")
    
    def open(self) -> bool:
        """비디오 파일 열기"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"Video file opened: {self.video_path} (FPS: {self.fps})")
                return True
            else:
                logger.error(f"Failed to open video file: {self.video_path}")
                return False
        except Exception as e:
            logger.error(f"Error opening video file: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 읽기"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame
        else:
            # 비디오 끝나면 처음으로 루프
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.debug("Video loop: restarting from beginning")
            return self.read()
    
    def release(self):
        """비디오 파일 해제"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video file released")
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        return self.fps


class DummySource(CameraSource):
    """더미 소스 (랜덤 노이즈 프레임 생성)"""
    
    def __init__(self):
        self.is_open = False
        self.frame_count = 0
        self.config = CAMERA_CONFIG
        logger.info("DummySource initialized")
    
    def open(self) -> bool:
        """더미 소스 열기"""
        self.is_open = True
        logger.info("Dummy source opened")
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """랜덤 프레임 생성"""
        if not self.is_open:
            return False, None
        
        self.frame_count += 1
        
        # 랜덤 노이즈 프레임 생성
        frame = np.random.randint(
            0, 255,
            (self.config["height"], self.config["width"], 3),
            dtype=np.uint8
        )
        
        # 프레임 번호 텍스트 추가
        cv2.putText(
            frame,
            f"Frame {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return True, frame
    
    def release(self):
        """더미 소스 해제"""
        self.is_open = False
        logger.info(f"Dummy source released (total frames: {self.frame_count})")
    
    def is_opened(self) -> bool:
        return self.is_open


class WebStreamSource(CameraSource):
    """웹 스트리밍 소스 (RTMP, RTSP 등)"""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap = None
        self.fps = CAMERA_CONFIG["fps"]
        logger.info(f"WebStreamSource initialized: {stream_url}")
    
    def open(self) -> bool:
        """
        웹 스트림 열기
        """
        try:
            # TODO(WebStreamSource.open, L229-L235): RTMP/RTSP 연결 보강
            #  - 재시도/타임아웃/인증 파라미터 지원
            #  - OpenCV 실패 시 ffmpeg 파이프 대안 적용
            
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                logger.info(f"Web stream opened: {self.stream_url}")
                return True
            else:
                logger.error(f"Failed to open web stream: {self.stream_url}")
                return False
        except Exception as e:
            logger.error(f"Error opening web stream: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """스트림에서 프레임 읽기"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame
        else:
            logger.warning("Failed to read from web stream")
            return False, None
    
    def release(self):
        """스트림 해제"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Web stream released")
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        return self.fps


def create_camera_source(source_type: str = "dummy", **kwargs) -> CameraSource:
    """
    카메라 소스 팩토리 함수
    
    Args:
        source_type: "picamera", "video", "dummy", "stream"
        **kwargs: 소스별 추가 인자
            - video: video_path
            - stream: stream_url
    
    Returns:
        CameraSource 인스턴스
    """
    if source_type == "picamera":
        return PiCameraSource()
    
    elif source_type == "video":
        video_path = kwargs.get("video_path", "")
        if not video_path:
            raise ValueError("video_path required for video source")
        return VideoFileSource(video_path)
    
    elif source_type == "stream":
        stream_url = kwargs.get("stream_url", "")
        if not stream_url:
            raise ValueError("stream_url required for stream source")
        return WebStreamSource(stream_url)
    
    elif source_type == "dummy":
        return DummySource()
    
    else:
        raise ValueError(f"Unknown source type: {source_type}")


if __name__ == "__main__":
    # 테스트 코드
    print("Testing camera sources...")
    
    # 더미 소스 테스트
    source = create_camera_source("dummy")
    
    if source.open():
        for i in range(5):
            ret, frame = source.read()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
            time.sleep(0.1)
        
        source.release()
