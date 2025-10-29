#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로깅 유틸리티
- 프로젝트 전체에서 사용하는 통합 로거
- 구현 완료
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import LOGGING_CONFIG


class Logger:
    """프로젝트 전용 로거"""
    
    _instances = {}
    
    def __new__(cls, name: str):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name: str):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
        
        # 중복 핸들러 방지
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """로그 핸들러 설정"""
        formatter = logging.Formatter(LOGGING_CONFIG["format"])
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러
        log_file = LOGGING_CONFIG["file"]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


def get_logger(name: str) -> Logger:
    """로거 인스턴스 반환"""
    return Logger(name)