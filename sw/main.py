#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-블록체인 기반 미디어 무결성 모듈 메인 실행 파일
- 전체 시스템 통합 실행
- 구현 완료
"""

import sys
import argparse
import signal

from capture.frame_capture import FrameCaptureSystem
from common.logger import get_logger
from common.config import get_config

logger = get_logger(__name__)

# 전역 시스템 인스턴스
capture_system = None


def signal_handler(signum, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    logger.info(f"Received signal {signum}, shutting down...")
    
    if capture_system:
        capture_system.stop()
        capture_system.teardown()
    
    sys.exit(0)


def print_banner():
    """배너 출력"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   AI-블록체인 기반 미디어 무결성 모듈                          ║
║   Google-아주대학교 AI 융합 캡스톤 디자인                      ║
║                                                               ║
║   팀원: 유성현, 최지예, 신지웅, 김서연, 임수인, 정영신                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_system_info():
    """시스템 정보 출력"""
    config = get_config()
    
    print("\n[시스템 설정]")
    print(f"  카메라: {config['camera']['width']}x{config['camera']['height']} @ {config['camera']['fps']} FPS")
    print(f"  전처리: {config['processing']['preprocessing']['target_size']}")
    print(f"  AI 모델: YOLOv5-nano, Mediapipe, Deepfake Detector")
    print()


def main():
    """메인 함수"""
    global capture_system
    
    # 인자 파싱
    parser = argparse.ArgumentParser(
        description="AI-블록체인 기반 미디어 무결성 모듈",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="dummy",
        choices=["dummy", "picamera", "video", "stream"],
        help="카메라 소스 타입 (기본: dummy)"
    )
    
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="비디오 파일 경로 (source=video 시 필수)"
    )
    
    parser.add_argument(
        "--stream-url",
        type=str,
        default="",
        help="스트림 URL (source=stream 시 필수)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help="실행 모드: single-thread 또는 multi-process (기본: multi)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="실행 시간 (초, None=무한)"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="배너 출력 생략"
    )
    
    args = parser.parse_args()
    
    # 배너 출력
    if not args.no_banner:
        print_banner()
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 로그 시작
    logger.info("=" * 60)
    logger.info("시스템 시작")
    logger.info("=" * 60)
    
    # 시스템 정보 출력
    print_system_info()
    
    # 소스별 인자 준비
    source_kwargs = {}
    if args.source == "video":
        if not args.video_path:
            logger.error("video source requires --video-path")
            sys.exit(1)
        source_kwargs["video_path"] = args.video_path
    
    elif args.source == "stream":
        if not args.stream_url:
            logger.error("stream source requires --stream-url")
            sys.exit(1)
        source_kwargs["stream_url"] = args.stream_url
    
    # 시스템 생성
    logger.info(f"Creating capture system (source: {args.source}, mode: {args.mode})")
    capture_system = FrameCaptureSystem(args.source, **source_kwargs)
    
    try:
        # 초기화
        logger.info("Initializing system...")
        capture_system.setup()
        logger.info("System initialized successfully")
        
        # 실행
        print(f"\n[실행 시작] 모드: {args.mode}, 소스: {args.source}")
        print("Ctrl+C를 눌러 종료하세요.\n")
        
        if args.mode == "single":
            logger.info("Running in single-thread mode")
            capture_system.run_single_thread(duration=args.duration)
        
        else:
            logger.info("Running in multi-process mode")
            capture_system.run_multiprocess(duration=args.duration)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # 종료
        logger.info("Shutting down...")
        if capture_system:
            capture_system.stop()
            capture_system.teardown()
        
        logger.info("=" * 60)
        logger.info("시스템 종료")
        logger.info("=" * 60)
        
        print("\n프로그램이 정상적으로 종료되었습니다.")


if __name__ == "__main__":
    main()