#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 MJPEG 프레임 스트리머
- SSH 환경에서 GUI 출력 대신 브라우저로 프레임 확인용
- FrameCaptureSystem에서 전처리된 프레임을 전달해 사용
- 구현 완료
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

import cv2
import numpy as np

from common.logger import get_logger

logger = get_logger(__name__)


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """JPEG 인코딩을 위해 입력 프레임을 BGR uint8로 변환"""
    if frame is None:
        return frame

    working = frame
    if not isinstance(working, np.ndarray):
        working = np.asarray(working)

    if working.dtype != np.uint8:
        if np.issubdtype(working.dtype, np.floating):
            working = np.clip(working, 0.0, 1.0)
            working = (working * 255).astype(np.uint8)
        else:
            working = working.astype(np.uint8)

    if working.ndim == 2:
        working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
    elif working.ndim == 3 and working.shape[2] == 3:
        # 대부분의 파이프라인이 RGB를 사용하므로 변환
        working = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)

    return working


class _StreamingHTTPServer(HTTPServer):
    """프레임 공급자를 참조하는 HTTPServer"""

    def __init__(self, server_address: Tuple[str, int], handler_class, frame_supplier):
        self.frame_supplier = frame_supplier
        super().__init__(server_address, handler_class)

    def get_frame(self) -> Optional[np.ndarray]:
        return self.frame_supplier()


def _build_handler():
    """프레임 스트림을 제공하는 핸들러 클래스 생성"""

    class StreamRequestHandler(BaseHTTPRequestHandler):
        server: _StreamingHTTPServer  # 타입 힌트

        def do_GET(self):
            if self.path not in ("/", "/stream", "/stream.mjpg"):
                self.send_error(404, "Stream not found")
                return

            self.send_response(200)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    frame = self.server.get_frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue

                    ret, encoded = cv2.imencode(".jpg", frame)
                    if not ret:
                        logger.warning("Failed to encode frame for streaming")
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(encoded.tobytes())
                    self.wfile.write(b"\r\n")
            except BrokenPipeError:
                # 클라이언트 연결 종료
                logger.debug("Frame stream client disconnected")
            except Exception as exc:
                logger.error(f"Frame streaming error: {exc}", exc_info=True)

        def log_message(self, format, *args):
            # 기본 로그 (stdout) 억제
            return

    return StreamRequestHandler


class FrameStreamer:
    """HTTP MJPEG 스트리머"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._server: Optional[_StreamingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """스트리머 시작"""
        if self._running:
            return

        handler_cls = _build_handler()
        try:
            self._server = _StreamingHTTPServer(
                (self.host, self.port), handler_cls, self._get_frame
            )
        except OSError as exc:
            logger.error(f"Failed to start FrameStreamer on {self.port}: {exc}")
            self._server = None
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._serve_forever, name="FrameStreamer", daemon=True
        )
        self._thread.start()
        logger.info(
            "Frame streamer started at http://%s:%d/stream.mjpg",
            self.host,
            self.port,
        )

    def _serve_forever(self):
        if not self._server:
            return
        try:
            self._server.serve_forever()
        except Exception as exc:
            logger.error(f"FrameStreamer server error: {exc}", exc_info=True)
        finally:
            self._running = False

    def stop(self):
        """스트리머 중지"""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._running = False
        self._server = None
        self._thread = None
        logger.info("Frame streamer stopped")

    def push_frame(self, frame: np.ndarray):
        """최신 프레임 갱신"""
        if not self._running:
            return
        bgr_frame = _ensure_bgr(frame)
        with self._frame_lock:
            self._latest_frame = bgr_frame.copy()

    def _get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()
