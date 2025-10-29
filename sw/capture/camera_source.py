#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 임시 구현 완료
"""
카메라 소스 추상화
- 실제 카메라, 더미 파일, 웹 스트리밍 등 다양한 소스 지원
- 개발 환경(aarch64 서버)과 배포 환경(라즈베리파이) 모두 지원
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from typing import Any, Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from fractions import Fraction
from contextlib import contextmanager
import time
from threading import Lock
from urllib.parse import urlparse, urlunparse

from common.config import CAMERA_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)

try:
    from picamera2 import Picamera2  # type: ignore
except ImportError:  # pragma: no cover - Picamera2가 없는 환경
    Picamera2 = None


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
        self._lock = Lock()
        self._cleanup_callbacks: List[Tuple[str, Callable[[], None]]] = []
        self._picamera_config = dict(self.config.get("picamera", {}))
        self._capture_stream = self._picamera_config.get("stream_name", "main")
        self._use_capture_array = bool(self._picamera_config.get("use_capture_array", True))
        logger.info("PiCameraSource initialized")

    def register_cleanup(self, name: str, callback: Callable[[], None]) -> None:
        """외부 리소스 해제 콜백 등록"""
        self._cleanup_callbacks.append((name, callback))

    def _normalize_controls(self, controls: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in controls.items():
            if not key:
                continue
            normalized_key = "FrameRate" if key.lower() == "framerate" else key
            control_value = value
            if normalized_key == "FrameRate" and not isinstance(value, Fraction):
                try:
                    if isinstance(value, (int, float)):
                        control_value = Fraction(value).limit_denominator(120)
                    else:
                        control_value = Fraction(str(value))
                except (ValueError, ZeroDivisionError, TypeError):
                    control_value = value
            normalized[normalized_key] = control_value
        return normalized

    def _build_video_configuration_kwargs(self) -> Dict[str, Any]:
        main_cfg = dict(self._picamera_config.get("main", {}))
        main_cfg.setdefault(
            "size",
            (int(self.config.get("width", 640)), int(self.config.get("height", 480))),
        )
        main_cfg.setdefault("format", self.config.get("format", "RGB888"))

        video_kwargs: Dict[str, Any] = {"main": main_cfg}

        for optional_key in ("lores", "raw"):
            if optional_key in self._picamera_config:
                video_kwargs[optional_key] = self._picamera_config[optional_key]

        for optional_key in ("controls", "buffer_count", "transform", "colour_space"):
            if optional_key in self._picamera_config:
                value = self._picamera_config[optional_key]
                if optional_key == "controls" and isinstance(value, dict):
                    video_kwargs[optional_key] = self._normalize_controls(value)
                elif value is not None:
                    video_kwargs[optional_key] = value
        return video_kwargs

    def _safe_close_camera(self):
        if self.camera is None:
            return
        try:
            self.camera.close()
        except Exception as exc:  # pragma: no cover - closing failures on hardware
            logger.warning(f"PiCamera close error: {exc}")
        self.camera = None

    def _resolve_channel_count(self) -> int:
        fmt = str(self.config.get("format", "RGB888")).upper()
        if fmt in {"RGB888", "BGR888"}:
            return 3
        if fmt in {"RGBA8888", "BGRA8888", "ARGB8888", "ABGR8888", "XRGB8888", "XBGR8888"}:
            return 4
        # 기본값: RGB 3채널로 처리
        return 3

    def _capture_from_buffer(self, stream_name: str) -> Optional[np.ndarray]:
        if self.camera is None:
            return None

        buffer = self.camera.capture_buffer(stream_name)
        if buffer is None:
            return None

        if isinstance(buffer, np.ndarray):
            array = buffer
        elif hasattr(buffer, "array"):
            array = np.asarray(buffer.array)
        else:
            array = np.frombuffer(buffer, dtype=np.uint8)

        channels = self._resolve_channel_count()
        width = int(self.config.get("width", 640))
        height = int(self.config.get("height", 480))
        expected_size = width * height * channels

        if array.size < expected_size:
            logger.warning(
                "Picamera buffer smaller than expected (%s < %s)",
                array.size,
                expected_size,
            )
            return None

        reshaped = np.ascontiguousarray(array[:expected_size]).reshape(
            height, width, channels
        )

        fmt = str(self.config.get("format", "RGB888")).upper()
        if fmt == "BGR888":
            reshaped = cv2.cvtColor(reshaped, cv2.COLOR_BGR2RGB)
        elif fmt == "BGRA8888":
            reshaped = cv2.cvtColor(reshaped, cv2.COLOR_BGRA2RGB)
        elif fmt in {"RGBA8888", "ARGB8888", "XRGB8888"}:
            reshaped = cv2.cvtColor(reshaped, cv2.COLOR_RGBA2RGB)

        return reshaped
    
    def open(self) -> bool:
        """
        Picamera2 초기화
        """
        if Picamera2 is None:
            logger.error(
                "Picamera2 라이브러리를 찾을 수 없다. `sudo apt install python3-picamera2`로 설치한다."
            )
            return False

        self._cleanup_callbacks.clear()

        try:
            self.camera = Picamera2()
        except Exception as exc:  # pragma: no cover - 하드웨어 의존
            logger.error(f"Picamera2 인스턴스를 생성하지 못했다: {exc}", exc_info=True)
            self.camera = None
            return False

        try:
            config_kwargs = self._build_video_configuration_kwargs()
            video_config = self.camera.create_video_configuration(**config_kwargs)
        except Exception as exc:
            logger.error(f"Picamera2 설정 생성 실패: {exc}", exc_info=True)
            self._safe_close_camera()
            return False

        try:
            self.camera.configure(video_config)
        except Exception as exc:
            logger.error(f"Picamera2 구성 실패: {exc}", exc_info=True)
            self._safe_close_camera()
            return False

        try:
            self.camera.start()
        except Exception as exc:
            logger.error(f"Picamera2 시작 실패: {exc}", exc_info=True)
            self._safe_close_camera()
            return False

        post_start_controls = self._picamera_config.get("post_start_controls")
        if isinstance(post_start_controls, dict) and post_start_controls:
            try:
                self.camera.set_controls(self._normalize_controls(post_start_controls))
            except Exception as exc:  # pragma: no cover - 환경 의존
                logger.warning(f"시작 후 제어값 적용 실패: {exc}")

        # 구성 정보 로그
        main_cfg = config_kwargs.get("main", {})
        size = main_cfg.get("size", ("?", "?"))
        fmt = main_cfg.get("format", self.config.get("format", "RGB888"))
        fps_config = self._picamera_config.get("controls", {}).get(
            "FrameRate", self.config.get("fps")
        )
        if isinstance(fps_config, Fraction):
            fps_display = float(fps_config)
        else:
            try:
                fps_display = float(fps_config)
            except (TypeError, ValueError):
                fps_display = fps_config
        logger.info("PiCamera2 started (%sx%s %s @ %s FPS)", size[0], size[1], fmt, fps_display)
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        프레임 캡처
        """
        if self.camera is None:
            return False, None
        
        try:
            with self._lock:
                if self._use_capture_array:
                    array = self.camera.capture_array(self._capture_stream)
                    if array is None:
                        return False, None
                    frame = np.ascontiguousarray(array)
                else:
                    frame = self._capture_from_buffer(self._capture_stream)

            if frame is None:
                return False, None

            return True, frame

        except RuntimeError as exc:
            if self._use_capture_array:
                logger.warning(
                    "capture_array 실패로 capture_buffer로 폴백합니다: %s", exc
                )
                with self._lock:
                    frame = self._capture_from_buffer(self._capture_stream)
                if frame is not None:
                    return True, frame
            logger.error(f"Failed to read frame: {exc}", exc_info=True)
            return False, None
        except Exception as e:
            logger.error(f"Failed to read frame: {e}", exc_info=True)
            return False, None
    
    def release(self):
        """카메라 해제"""
        callbacks: List[Tuple[str, Callable[[], None]]] = []
        released = False
        with self._lock:
            if self.camera is not None:
                try:
                    self.camera.stop()
                except Exception as exc:  # pragma: no cover - 하드웨어 의존
                    logger.warning(f"PiCamera stop error: {exc}")
                self._safe_close_camera()
                released = True
            if self._cleanup_callbacks:
                callbacks = list(self._cleanup_callbacks)
                self._cleanup_callbacks.clear()

        for name, callback in callbacks:
            try:
                callback()
            except Exception as exc:
                logger.warning(f"Cleanup `{name}` failed: {exc}")

        if callbacks or released:
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

    def __init__(
        self,
        stream_url: str,
        *,
        retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        auth: Optional[Tuple[str, str]] = None,
        backend: Optional[str] = None,
        ffmpeg_options: Optional[Dict[str, str]] = None,
        open_timeout_ms: Optional[int] = None,
        read_timeout_ms: Optional[int] = None,
    ):
        self.stream_url = stream_url
        self.cap: Optional[cv2.VideoCapture] = None
        stream_config = CAMERA_CONFIG.get("stream", {})
        self.retries = max(1, int(retries if retries is not None else stream_config.get("retries", 3)))
        self.retry_delay = max(0.0, float(retry_delay if retry_delay is not None else stream_config.get("retry_delay", 1.5)))
        self._backend_name = backend or stream_config.get("backend", "auto")
        self._backend_flag = self._resolve_backend_flag(self._backend_name)
        self.ffmpeg_options = dict(ffmpeg_options if ffmpeg_options is not None else stream_config.get("ffmpeg_options", {}))
        self.auth = auth if auth is not None else self._extract_auth(stream_config.get("auth", {}))
        self.open_timeout_ms = open_timeout_ms if open_timeout_ms is not None else stream_config.get("open_timeout_ms")
        self.read_timeout_ms = read_timeout_ms if read_timeout_ms is not None else stream_config.get("read_timeout_ms")
        self.fps = CAMERA_CONFIG["fps"]
        logger.info(
            "WebStreamSource initialized: %s (backend=%s, retries=%s)",
            stream_url,
            self._backend_name,
            self.retries,
        )

    def _resolve_backend_flag(self, backend_name: Optional[str]) -> Optional[int]:
        if backend_name is None:
            return None
        name = str(backend_name).lower()
        if name in ("auto", "", "none"):
            return None
        if name == "ffmpeg":
            return getattr(cv2, "CAP_FFMPEG", None)
        if name in {"gstreamer", "gst"}:
            return getattr(cv2, "CAP_GSTREAMER", None)
        if name in {"any", "default"}:
            return getattr(cv2, "CAP_ANY", None)
        logger.warning("알 수 없는 스트림 백엔드 `%s`, 기본값을 사용합니다.", backend_name)
        return None

    def _extract_auth(self, auth_config: Any) -> Optional[Tuple[str, str]]:
        if not isinstance(auth_config, dict):
            return None
        username = auth_config.get("username", "")
        password = auth_config.get("password", "")
        if username:
            return username, password or ""
        return None

    def _build_authenticated_url(self) -> str:
        if not self.auth:
            return self.stream_url
        parsed = urlparse(self.stream_url)
        if parsed.username:
            return self.stream_url
        username, password = self.auth
        credentials = username if not password else f"{username}:{password}"
        netloc = f"{credentials}@{parsed.netloc}"
        return urlunparse(parsed._replace(netloc=netloc))

    def _compose_ffmpeg_option_string(self) -> Optional[str]:
        if not self.ffmpeg_options:
            return None
        parts = []
        for key, value in self.ffmpeg_options.items():
            if value in (None, ""):
                parts.append(str(key))
            else:
                parts.append(f"{key};{value}")
        return "|".join(parts) if parts else None

    @contextmanager
    def _ffmpeg_options_context(self):
        option_string = self._compose_ffmpeg_option_string()
        if self._backend_flag != getattr(cv2, "CAP_FFMPEG", None) or not option_string:
            yield
            return

        previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = option_string
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous

    def _create_capture(self, url: str) -> cv2.VideoCapture:
        if self._backend_flag is None:
            return cv2.VideoCapture(url)

        cap = cv2.VideoCapture(url, self._backend_flag)
        if not cap or not cap.isOpened():
            logger.warning(
                "요청한 백엔드(%s)를 사용해 스트림을 열지 못해 기본 백엔드를 시도합니다.",
                self._backend_name,
            )
            if cap:
                cap.release()
            cap = cv2.VideoCapture(url)
        return cap

    def _release_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                logger.debug(f"VideoCapture release error: {exc}")
            self.cap = None

    def open(self) -> bool:
        """
        웹 스트림 열기
        """
        authenticated_url = self._build_authenticated_url()
        last_error: Optional[Exception] = None

        for attempt in range(self.retries):
            try:
                with self._ffmpeg_options_context():
                    self.cap = self._create_capture(authenticated_url)
                if self.cap is None or not self.cap.isOpened():
                    raise RuntimeError("VideoCapture.open() 실패")

                # 버퍼 크기 최소화 및 타임아웃 설정
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if self.open_timeout_ms:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(self.open_timeout_ms))
                if self.read_timeout_ms:
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(self.read_timeout_ms))

                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = fps if fps and fps > 1e-6 else CAMERA_CONFIG["fps"]
                logger.info(
                    "Web stream opened: %s (attempt %d/%d)",
                    self.stream_url,
                    attempt + 1,
                    self.retries,
                )
                return True
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to open web stream (%d/%d): %s",
                    attempt + 1,
                    self.retries,
                    exc,
                )
                self._release_capture()
                if attempt < self.retries - 1 and self.retry_delay > 0:
                    time.sleep(self.retry_delay)

        error_msg = str(last_error) if last_error else "unknown error"
        logger.error(f"Error opening web stream: {error_msg}")
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
            self._release_capture()
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
