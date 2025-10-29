#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프레임 버퍼 및 성능 모니터링 유틸리티
- 멀티프로세싱 공유 메모리 대신 스레드 안전 큐 기반 구현
- OpenCV 기반 파이프라인과 함께 사용
- 구현 완료
"""

import time
import threading
import queue
from collections import deque
from typing import Any, Deque, Optional


class FrameBuffer:
    """스레드 안전 큐 기반 프레임 버퍼"""

    def __init__(self, max_size: int = 10):
        self._frames: queue.Queue[Any] = queue.Queue(maxsize=max_size)
        self._results: Deque[Any] = deque(maxlen=100)
        self._running = True
        self._state_lock = threading.Lock()
        self._fps_lock = threading.Lock()
        self._fps: float = 0.0
        self._latest_result: Optional[Any] = None

    def start(self):
        """버퍼 실행 상태 전환"""
        self.clear()
        with self._state_lock:
            self._running = True

    def stop(self):
        """버퍼 중지"""
        with self._state_lock:
            self._running = False
        self._drain_queue(self._frames)

    def is_running(self) -> bool:
        """실행 여부 확인"""
        with self._state_lock:
            return self._running

    def has_frames(self) -> bool:
        """버퍼 내 프레임 존재 여부"""
        return not self._frames.empty()

    def put_frame(self, frame: Any, timeout: Optional[float] = None) -> bool:
        """프레임 추가"""
        if not self.is_running():
            return False

        try:
            if timeout is None:
                self._frames.put(frame, block=False)
            else:
                self._frames.put(frame, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get_frame(self, timeout: Optional[float] = None) -> Optional[Any]:
        """프레임 가져오기"""
        if not self.is_running() and self._frames.empty():
            return None

        try:
            if timeout is None:
                frame = self._frames.get(block=False)
            else:
                frame = self._frames.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None

    def put_result(self, result: Any):
        """추론 결과 저장"""
        with self._state_lock:
            self._latest_result = result
        self._results.append(result)

    def get_latest_result(self) -> Optional[Any]:
        """가장 최근 결과 반환"""
        with self._state_lock:
            return self._latest_result

    def update_fps(self, fps: float):
        """FPS 업데이트"""
        with self._fps_lock:
            self._fps = fps

    def get_fps(self) -> float:
        """현재 FPS 조회"""
        with self._fps_lock:
            return self._fps

    def clear(self):
        """버퍼 비우기"""
        self._drain_queue(self._frames)
        self._results.clear()
        with self._state_lock:
            self._latest_result = None
        with self._fps_lock:
            self._fps = 0.0

    @staticmethod
    def _drain_queue(q: queue.Queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break


class PerformanceMonitor:
    """성능 모니터링 도우미"""

    def __init__(self, window_size: int = 60):
        self._window_size = window_size
        self._process_times: Deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._last_log_time = 0.0

    def update(self, process_time: float):
        """프레임 처리 시간 기록"""
        with self._lock:
            self._process_times.append(process_time)

    def get_fps(self) -> float:
        """평균 FPS 계산"""
        with self._lock:
            if not self._process_times:
                return 0.0
            avg_time = sum(self._process_times) / len(self._process_times)
        if avg_time <= 0:
            return 0.0
        return 1.0 / avg_time

    def get_avg_latency(self) -> float:
        """평균 지연(ms) 반환"""
        with self._lock:
            if not self._process_times:
                return 0.0
            avg_time = sum(self._process_times) / len(self._process_times)
        return avg_time * 1000.0

    def should_log(self, interval: float) -> bool:
        """로그 출력 여부 결정"""
        current_time = time.time()
        with self._lock:
            if current_time - self._last_log_time >= interval:
                self._last_log_time = current_time
                return True
        return False

    def reset(self):
        """통계 초기화"""
        with self._lock:
            self._process_times.clear()
            self._last_log_time = 0.0


# 하위 호환성: 기존 SharedFrameBuffer 명칭 유지
SharedFrameBuffer = FrameBuffer
