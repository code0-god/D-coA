#!/usr/bin/env python3
"""
Picamera2 웹 스트리밍 통합 시스템
- 카메라 초기화 및 공유 메모리 설정
- 웹 스트리밍으로 팀원과 영상 공유
- C++ 백그라운드 프로세스 자동 실행
"""

from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from multiprocessing import shared_memory
from picamera2 import Picamera2
import time
import threading
import subprocess
import os
import atexit
import signal
import sys

app = Flask(__name__)

# 설정
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
SHM_NAME = "frame_buffer"
CPP_EXECUTABLE = "./camera_test"

# 전역 변수
camera = None
shm = None
cpp_process = None
latest_frame = None
frame_lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Picamera2 Live Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        h1 { color: #4CAF50; margin-bottom: 10px; }
        .status-bar {
            margin: 20px 0;
        }
        .status {
            display: inline-block;
            padding: 8px 20px;
            background: #4CAF50;
            border-radius: 20px;
            margin: 5px;
            font-size: 14px;
        }
        .cpp-status { background: #2196F3; }
        img {
            max-width: 90%;
            max-height: 70vh;
            border: 3px solid #4CAF50;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .info {
            margin-top: 20px;
            font-size: 14px;
            color: #aaa;
            line-height: 1.6;
        }
    </style>
    <script>
        setInterval(() => {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const elem = document.getElementById('cpp-status');
                    elem.textContent = data.cpp_running ? '● C++ Processing' : '○ C++ Stopped';
                    elem.style.background = data.cpp_running ? '#2196F3' : '#666';
                });
        }, 2000);
    </script>
</head>
<body>
    <h1>Picamera2 실시간 스트리밍</h1>
    <div class="status-bar">
        <div class="status">● Web Streaming</div>
        <div class="status cpp-status" id="cpp-status">● C++ Processing</div>
    </div>
    <img src="{{ url_for('video_feed') }}" alt="Live Stream">
    <div class="info">
        <p><strong>해상도:</strong> 640×640 | <strong>포맷:</strong> MJPEG</p>
        <p><strong>공유 메모리:</strong> Active (Python ↔ C++)</p>
        <p><strong>접속 주소:</strong> http://192.168.0.34:5000</p>
        <hr style="border-color: #333; margin: 15px 0;">
        <p>C++ 프로세스가 백그라운드에서 프레임을 처리합니다</p>
    </div>
</body>
</html>
"""

def cleanup():
    """종료 시 리소스 정리"""
    global camera, shm, cpp_process
    
    print("\n=== 시스템 종료 중 ===")
    
    # C++ 프로세스 종료
    if cpp_process and cpp_process.poll() is None:
        print("C++ 프로세스 종료...")
        cpp_process.terminate()
        try:
            cpp_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            cpp_process.kill()
    
    # 카메라 종료
    if camera:
        try:
            camera.stop()
            camera.close()
            print("카메라 정지")
        except:
            pass
    
    # 공유 메모리 정리
    if shm:
        try:
            shm.close()
            shm.unlink()
            print("공유 메모리 정리")
        except:
            pass
    
    print("=== 종료 완료 ===\n")

def signal_handler(sig, frame):
    """Ctrl+C 처리"""
    cleanup()
    sys.exit(0)

def start_cpp_process():
    """C++ 백그라운드 프로세스 시작"""
    global cpp_process
    
    if not os.path.exists(CPP_EXECUTABLE):
        print(f"\n경고: C++ 실행 파일 없음 ({CPP_EXECUTABLE})")
        print("     빌드: cd build && make\n")
        return False
    
    print(f"C++ 프로세스 시작: {CPP_EXECUTABLE}")
    try:
        cpp_process = subprocess.Popen(
            [CPP_EXECUTABLE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"C++ 프로세스 실행 (PID: {cpp_process.pid})")
        return True
    except Exception as e:
        print(f"C++ 프로세스 시작 실패: {e}")
        return False

def capture_frames():
    """백그라운드 스레드에서 프레임 캡처"""
    global latest_frame, camera, shm
    
    print("\n" + "="*50)
    print("  Picamera2 통합 시스템 초기화")
    print("="*50 + "\n")
    
    print("카메라 초기화 중...")
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(2)
    print(f"카메라 시작: {FRAME_WIDTH}×{FRAME_HEIGHT}\n")
    
    # 공유 메모리 생성
    frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=frame_size)
        print(f"공유 메모리 생성: {SHM_NAME} ({frame_size:,} bytes)")
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        print(f"공유 메모리 재사용: {SHM_NAME}")
    
    np_frame = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
    
    # C++ 프로세스 시작
    time.sleep(1)
    cpp_started = start_cpp_process()
    
    print("\n프레임 캡처 시작\n")
    
    frame_count = 0
    start_time = time.time()
    cpp_error_logged = False
    
    while True:
        frame = camera.capture_array()
        np.copyto(np_frame, frame)
        
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
            cv2.putText(bgr_frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.putText(bgr_frame, "Shared Memory: Active", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        with frame_lock:
            latest_frame = bgr_frame.copy()
        
        # C++ 프로세스 모니터링
        if cpp_started and cpp_process and cpp_process.poll() is not None:
            if not cpp_error_logged:
                print("\n경고: C++ 프로세스 종료됨")
                print("웹 스트리밍은 계속 작동합니다\n")
                cpp_error_logged = True
                cpp_started = False
        
        time.sleep(0.01)

def generate_frames():
    """MJPEG 스트림 생성"""
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {
        'cpp_running': cpp_process is not None and cpp_process.poll() is None,
        'camera_active': camera is not None,
        'shared_memory_active': shm is not None
    }

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)
    
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    time.sleep(4)
    
    print("\n" + "="*50)
    print("  웹 스트리밍 서버 시작")
    print("="*50)
    print("\n접속 방법:")
    print("  로컬:  http://localhost:5000")
    print("  팀원:  http://192.168.0.34:5000")
    print("\n종료: Ctrl+C\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)