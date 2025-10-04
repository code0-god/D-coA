#!/bin/bash

echo "========================================"
echo "  웹 스트리밍 시스템 시작"
echo "========================================"
echo ""

# 1. 기존 프로세스 정리
echo "기존 프로세스 정리 중..."

# 포트 5000 정리
PORT_PID=$(sudo lsof -ti:5000 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "  - 포트 5000 프로세스 종료 (PID: $PORT_PID)"
    sudo kill -9 $PORT_PID
fi

# Python 프로세스 정리
PYTHON_PIDS=$(pgrep -f "camera_web_stream\|camera_capture" 2>/dev/null)
if [ ! -z "$PYTHON_PIDS" ]; then
    echo "  - Python 프로세스 종료"
    echo "$PYTHON_PIDS" | xargs kill -9 2>/dev/null
fi

# C++ 프로세스 정리
pkill -9 camera_test 2>/dev/null

# 공유 메모리 정리
rm -f /dev/shm/frame_buffer 2>/dev/null

sleep 1
echo "정리 완료"
echo ""

# 2. Python 스크립트 실행
echo "웹 스트리밍 서버 시작..."
echo ""
python3 ./scripts/camera_capture.py

# 3. 종료 시 정리
echo ""
echo "종료 후 정리 중..."
pkill -9 camera_test 2>/dev/null
rm -f /dev/shm/frame_buffer 2>/dev/null
echo "완료"