#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공통 설정 관리
- 모든 모듈에서 사용하는 설정값 중앙 관리
- 임시 구현 완료
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# 디렉토리 경로
MODEL_DIR = PROJECT_ROOT / "ai_model" / "models"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# 디렉토리 생성
for dir_path in [MODEL_DIR, LOG_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 카메라 설정
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30,
    "format": "RGB888"
}

# AI 모델 설정
MODEL_CONFIG = {
    "yolo": {
        "model_path": MODEL_DIR / "yolov5n.tflite",
        "confidence_threshold": 0.5,
        "iou_threshold": 0.4,
        "target_classes": [0]  # COCO: 0 = person
    },
    "liveness": {
        "model": "mediapipe",  # or "dlib"
        "min_detection_confidence": 0.5,
        "movement_threshold": 0.02
    },
    "deepfake": {
        "pixel_analysis": True,
        "compression_check": True,
        "temporal_consistency": True,
        "threshold": 0.7
    }
}

# 처리 설정
PROCESSING_CONFIG = {
    "batch_size": 1,
    "max_queue_size": 10,
    "preprocessing": {
        "resize": True,
        "normalize": True,
        "target_size": (640, 480)
    }
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file": LOG_DIR / "app.log"
}

# 성능 모니터링
PERFORMANCE_CONFIG = {
    "enable_profiling": True,
    "fps_update_interval": 1.0,  # seconds
    "log_interval": 10.0  # seconds
}

def get_config(section: str = None):
    """설정값 반환"""
    configs = {
        "camera": CAMERA_CONFIG,
        "model": MODEL_CONFIG,
        "processing": PROCESSING_CONFIG,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_CONFIG
    }
    
    if section:
        return configs.get(section, {})
    return configs