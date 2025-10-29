#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프레임 캡처 및 파이프라인 메인 모듈
- 실시간 영상 캡처
- 전처리 수행
- AI 추론 모듈 호출
- Producer-Consumer 패턴 구현
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import argparse
from threading import Event, Thread
from typing import Optional

from common.config import CAMERA_CONFIG, PERFORMANCE_CONFIG
from common.logger import get_logger
from common.frame_buffer import FrameBuffer, PerformanceMonitor

from capture.camera_source import create_camera_source, CameraSource
from capture.preprocessor import FramePreprocessor
from capture.frame_streamer import FrameStreamer

# AI 추론 모듈
sys.path.append(str(Path(__file__).parent.parent / "ai_model"))
from ai_model import inference

logger = get_logger(__name__)


class FrameCaptureSystem:
    """프레임 캡처 및 처리 시스템"""
    
    def __init__(
        self,
        source_type: str = "dummy",
        stream_host: str = "0.0.0.0",
        stream_port: Optional[int] = None,
        **source_kwargs,
    ):
        """
        초기화
        
        Args:
            source_type: 카메라 소스 타입
            stream_host: MJPEG 스트림 바인딩 호스트
            stream_port: 스트림 포트 (None이면 비활성)
            **source_kwargs: 소스별 추가 인자
        """
        self.source_type = source_type
        self.source_kwargs = source_kwargs
        self.stream_host = stream_host
        self.stream_port = stream_port
        
        # 컴포넌트
        self.camera: Optional[CameraSource] = None
        self.preprocessor = FramePreprocessor()
        self.frame_buffer = FrameBuffer(max_size=10)
        
        # 성능 모니터링
        self.perf_monitor = PerformanceMonitor()
        
        # 실행 제어
        self.capture_thread: Optional[Thread] = None
        self.inference_thread: Optional[Thread] = None
        self._stop_event = Event()
        self.streamer: Optional[FrameStreamer] = None
        
        logger.info(f"FrameCaptureSystem initialized (source: {source_type})")
    
    def setup(self):
        """시스템 초기화"""
        # 카메라 소스 생성
        self.camera = create_camera_source(self.source_type, **self.source_kwargs)
        
        if not self.camera.open():
            raise RuntimeError("Failed to open camera source")
        
        self._start_streamer()
        
        logger.info("FrameCaptureSystem setup complete")
    
    def _start_streamer(self):
        """MJPEG 스트리머 시작"""
        if self.stream_port is None:
            return
        if self.streamer is None:
            self.streamer = FrameStreamer(host=self.stream_host, port=self.stream_port)
        self.streamer.start()
    
    def _stop_streamer(self):
        """스트리머 중지"""
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
    
    def capture_loop(self):
        """
        캡처 루프 (Producer)
        - 카메라에서 프레임 읽기
        - 전처리 수행
        - 공유 버퍼에 전달
        """
        logger.info("Capture loop started")
        
        frame_count = 0
        fps_timer = time.time()
        fps_counter = 0
        
        try:
            while not self._stop_event.is_set() and self.frame_buffer.is_running():
                start_time = time.time()
                
                # TODO(FrameCaptureSystem.capture_loop, L87-L143): 실제 캡처 파이프라인 구성
                #  - CameraSource.read 반환값 검증 및 재시도 정책 적용
                #  - FramePreprocessor 옵션을 CONFIG와 연동
                logger.warning(
                    "FrameCaptureSystem.capture_loop: 재시도/전처리 구성 로직이 미구현 상태입니다."
                )
                
                
                # 1. 카메라에서 프레임 읽기
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(0.01)
                    continue
                
                # 2. 전처리
                processed_frame = self.preprocessor.process(
                    frame,
                    resize=True,
                    normalize=False,  # AI 모듈에서 처리
                    denoise=False,
                    enhance=False
                )
                
                # 3. 프레임 버퍼에 전달
                if self.frame_buffer.put_frame(processed_frame, timeout=0.1):
                    frame_count += 1
                    fps_counter += 1
                    if self.streamer:
                        self.streamer.push_frame(processed_frame)
                else:
                    logger.warning("Buffer full, frame dropped")
                
                # FPS 계산 및 업데이트
                current_time = time.time()
                if current_time - fps_timer >= 1.0:
                    fps = fps_counter / (current_time - fps_timer)
                    self.frame_buffer.update_fps(fps)
                    fps_timer = current_time
                    fps_counter = 0
                
                # 성능 모니터링
                process_time = time.time() - start_time
                self.perf_monitor.update(process_time)
                
                if self.perf_monitor.should_log(interval=5.0):
                    logger.info(
                        f"Capture: {frame_count} frames, "
                        f"FPS: {self.perf_monitor.get_fps():.1f}, "
                        f"Latency: {self.perf_monitor.get_avg_latency():.1f}ms"
                    )
                
                # FPS 제한 (필요시)
                target_fps = self.camera.get_fps()
                if target_fps > 0:
                    sleep_time = (1.0 / target_fps) - process_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Capture loop interrupted by user")
        
        except Exception as e:
            logger.error(f"Capture loop error: {e}", exc_info=True)
        
        finally:
            self.frame_buffer.stop()
            logger.info(f"Capture loop finished (total frames: {frame_count})")
    
    def inference_loop(self):
        """
        추론 루프 (Consumer)
        - 공유 버퍼에서 프레임 가져오기
        - AI 추론 수행
        - 결과 저장
        """
        logger.info("Inference loop started")
        
        # AI 모듈 초기화
        inference.setup()
        
        frame_count = 0
        pass_count = 0
        
        try:
            while True:
                if self._stop_event.is_set() and not self.frame_buffer.has_frames():
                    break
                if not self.frame_buffer.is_running() and not self.frame_buffer.has_frames():
                    break

                # TODO(FrameCaptureSystem.inference_loop, L170-L214): 추론 파이프라인 구현
                #  - ai_model.inference.analyze 대신 실제 배치/큐 로직 구성
                #  - 결과 저장 구조(common.frame_buffer) 확장
                logger.warning(
                    "FrameCaptureSystem.inference_loop: 실제 추론 파이프라인이 연결되어 있지 않습니다."
                )
                
                
                # 1. 프레임 버퍼에서 프레임 가져오기
                frame = self.frame_buffer.get_frame(timeout=0.1)
                
                if frame is None:
                    continue
                
                # 2. AI 추론 수행
                start_time = time.time()
                pass_flag, result = inference.analyze(frame)
                inference_time = time.time() - start_time
                
                frame_count += 1
                if pass_flag:
                    pass_count += 1
                
                # 3. 결과 저장
                result_data = {
                    "frame_id": frame_count,
                    "pass_flag": pass_flag,
                    "inference_time": inference_time,
                    "result": result
                }
                self.frame_buffer.put_result(result_data)
                
                # 주기적 로깅
                if frame_count % 30 == 0:
                    pass_rate = pass_count / frame_count * 100
                    logger.info(
                        f"Inference: {frame_count} frames, "
                        f"Pass rate: {pass_rate:.1f}%, "
                        f"Avg time: {inference_time*1000:.1f}ms"
                    )
        
        except KeyboardInterrupt:
            logger.info("Inference loop interrupted by user")
        
        except Exception as e:
            logger.error(f"Inference loop error: {e}", exc_info=True)
        
        finally:
            # 통계 출력
            stats = inference.get_statistics()
            logger.info(f"Inference loop finished: {stats}")
            inference.teardown()
    
    def run_single_thread(self, duration: Optional[float] = None):
        """
        단일 스레드 실행 (디버깅용)
        
        Args:
            duration: 실행 시간 (초), None이면 무한
        """
        logger.info("Running in single-thread mode")
        
        inference.setup()
        self._stop_event.clear()
        self._start_streamer()
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while not self._stop_event.is_set():
                # 시간 제한 체크
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # 프레임 캡처
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # 전처리
                processed = self.preprocessor.process(frame, resize=True)
                if self.streamer:
                    self.streamer.push_frame(processed)
                
                # 추론
                pass_flag, result = inference.analyze(processed)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            logger.info(f"Single-thread run finished: {frame_count} frames")
            inference.teardown()
    
    def run_multiprocess(self, duration: Optional[float] = None):
        """
        멀티프로세스 실행 (Producer-Consumer)
        
        Args:
            duration: 실행 시간 (초), None이면 무한
        """
        logger.info("Running in multi-thread mode (Producer-Consumer)")
        
        # TODO(FrameCaptureSystem.run_multiprocess, L279-L314): 멀티프로세스 요구사항 재검토
        #  - 프로세스 분리 시 IPC/큐 설계 업데이트
        #  - 리소스 종료 및 예외 처리 시나리오 명세
        
        
        # 실행 준비
        self._stop_event.clear()
        self.frame_buffer.start()
        self._start_streamer()
        
        # 스레드 생성
        self.capture_thread = Thread(target=self.capture_loop, name="CaptureThread", daemon=True)
        self.inference_thread = Thread(target=self.inference_loop, name="InferenceThread", daemon=True)
        
        # 스레드 시작
        self.capture_thread.start()
        self.inference_thread.start()
        
        try:
            # 실행 시간 제한
            if duration:
                time.sleep(duration)
                self.stop()
            else:
                # 무한 실행
                while not self._stop_event.is_set():
                    active_threads = [
                        thread for thread in (self.capture_thread, self.inference_thread)
                        if thread and thread.is_alive()
                    ]
                    if not active_threads:
                        break
                    time.sleep(0.2)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.stop()
    
    def stop(self):
        """시스템 중지"""
        logger.info("Stopping FrameCaptureSystem...")
        
        # 실행 중지
        self._stop_event.set()
        self.frame_buffer.stop()
        
        # 스레드 종료 대기
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=5.0)
        
        self.capture_thread = None
        self.inference_thread = None
        self._stop_streamer()
        
        logger.info("FrameCaptureSystem stopped")
    
    def teardown(self):
        """리소스 해제"""
        if self.camera:
            self.camera.release()
        
        self.frame_buffer.clear()
        self._stop_streamer()
        
        logger.info("FrameCaptureSystem teardown complete")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Frame Capture System")
    parser.add_argument(
        "--source",
        type=str,
        default="dummy",
        choices=["dummy", "picamera", "video", "stream"],
        help="Camera source type"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="Video file path (for video source)"
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default="",
        help="Stream URL (for stream source)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Execution mode: single-thread or multi-process"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds (None for infinite)"
    )
    parser.add_argument(
        "--stream-host",
        type=str,
        default="0.0.0.0",
        help="MJPEG stream binding host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=5000,
        help="MJPEG stream port (set 0 to disable streaming, default: 5000)"
    )
    
    args = parser.parse_args()
    
    # 소스별 인자 준비
    source_kwargs = {}
    if args.source == "video":
        source_kwargs["video_path"] = args.video_path
    elif args.source == "stream":
        source_kwargs["stream_url"] = args.stream_url
    
    # 시스템 생성
    stream_port = args.stream_port if args.stream_port and args.stream_port > 0 else None
    system = FrameCaptureSystem(
        args.source,
        stream_host=args.stream_host,
        stream_port=stream_port,
        **source_kwargs,
    )
    
    try:
        # 초기화
        system.setup()
        
        # 실행
        if args.mode == "single":
            system.run_single_thread(duration=args.duration)
        else:
            system.run_multiprocess(duration=args.duration)
    
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    
    finally:
        system.teardown()


if __name__ == "__main__":
    main()
