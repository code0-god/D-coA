## 1. 프로젝트 개요

AI·블록체인 기반 미디어 무결성 프로젝트의 소프트웨어(`sw/`)는 실시간 프레임 캡처, 전처리, AI 추론, 그리고 보조 유틸리티 모듈로 구성되어 있다.  
현 시점에서는 전체 파이프라인이 **더미 구현**과 **TODO** 중심으로 구성되어 있으며, 실행 시 경고 로그를 출력하면서 기본 동작만 수행한다. 실제 모델, 하드웨어 연동, 블록체인 연계는 추후 구현을 전제로 한다.

### 1.1 디렉터리 구조

```
sw/
 ├── ai_model/          # 객체 탐지, 라이브니스, 딥페이크 모듈 및 통합 추론
 │   └── tests/         # AI 모듈 단위 테스트
 ├── capture/           # 카메라 소스, 전처리, 프레임 캡처 파이프라인, MJPEG 스트리머
 │   └── tests/         # 캡처·전처리 단위 테스트
 ├── common/            # 설정, 로깅, 프레임 버퍼 등 공통 유틸리티
 ├── main.py            # 전체 시스템 실행 엔트리(라즈베리 파이 등에서 사용 예정)
 ├── README.md          # 본 문서
 ├── requirements.txt   # Python 의존성
 └── setup.sh           # 필요 시 환경 초기화 스크립트(작성 예정)
```

---

## 2. 개발 환경 준비

### 2.1 필수 요구 사항
- Python 3.10 이상 (aarch64 환경에서 검증)
- pip 최신 버전 권장 (`python3 -m pip install --upgrade pip`)
- (선택) 라즈베리 파이에서 실행할 경우 Picamera2 지원 환경

### 2.2 필수 패키지 설치

프로젝트 루트(`/srv/D-coA/sw`)에서 다음 명령을 실행한다.

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 일반 Linux 서버에서 필요한 최소 패키지만 정의되어 있다.  
라즈베리 파이용 `picamera2` 등은 주석 처리되어 있으며, 실제 Pi 환경에서만 주석을 해제하고 설치하면 된다.

### 2.3 추가 시스템 라이브러리

일부 패키지는 OS 수준의 헤더/라이브러리가 필요하다.

| 패키지 | 필요 라이브러리 | 비고 |
| ------ | ---------------- | ---- |
| mediapipe | `protobuf`, `OpenGL` 관련 패키지 | pip 설치 시 자동 해결됨 |
| picamera2 (선택) | `python-prctl`, `libcap-dev`, `libcamera-dev` | 라즈베리 파이 전용 |
| dlib (선택) | `cmake`, `boost` 등 | 현재 미사용, 필요 시 설치 |

---

## 3. 실행 및 테스트

### 3.1 단위 테스트

아래 명령으로 캡처/AI 모듈 테스트를 모두 실행할 수 있다.

```bash
cd /srv/D-coA/sw
python3 -m unittest ai_model.tests.test_inference capture.tests.test_capture
```

혹은 `pytest`를 사용하려면:

```bash
pytest -q
```

> **주의**
> - TODO가 남아 있는 함수는 경고 로그를 출력하며 더미 데이터를 반환한다.
> - 테스트는 이러한 전제 하에 작성되어 있으므로, TODO 구현 후에는 테스트를 업데이트해야 한다.
> - OpenCV, mediapipe 등 외부 라이브러리가 설치되어 있지 않으면 테스트가 실패할 수 있다.

### 3.2 프레임 캡처 파이프라인 실행

```bash
python3 capture/frame_capture.py \
    --source stream \
    --stream-url <영상 URL> \
    --mode multi \
    --stream-port 5000
```

옵션 설명:

| 옵션 | 설명 |
| --- | --- |
| `--source` | `dummy` / `video` / `stream` / `picamera` 중 선택 |
| `--video-path` | `--source video`일 때 영상 파일 경로 |
| `--stream-url` | `--source stream`일 때 네트워크 스트림 URL |
| `--mode` | `single`(단일 스레드) / `multi`(캡처 스레드 + 추론 프로세스 분리) |
| `--duration` | 실행 시간 제한(초), 미지정 시 무한 실행 |
| `--stream-host` | MJPEG 스트리머 호스트 (기본 `0.0.0.0`) |
| `--stream-port` | 스트리머 포트, 기본값 5000 (0이면 비활성화) |

실행 후 브라우저에서 `http://<호스트>:<포트>/stream.mjpg`(기본 `http://<호스트>:5000/stream.mjpg`)에 접속하면 실시간 프레임을 확인할 수 있다.

자세한 캡처 모듈 설명은 `sw/capture/README.md`를 참고한다.

#### 3.2.1 입력 소스별 예시

- **더미 소스 (랜덤 프레임)**
  ```bash
  python3 capture/frame_capture.py \
      --source dummy \
      --mode single \
      --duration 10 \
      --stream-port 5000
  ```
  - 랜덤 노이즈 프레임을 생성한다.
  - 스트리머 포트는 생략 시 5000이며, 웹에서 `http://localhost:5000/stream.mjpg`로 확인 가능하다.
  - 단시간 테스트에 적합하다.

- **비디오 파일을 카메라처럼 사용**
  ```bash
  python3 capture/frame_capture.py \
      --source video \
      --video-path ~/sample.mp4 \
      --mode multi \
      --stream-port 5000
  ```
  - 영상이 끝나면 처음으로 돌아가 반복 재생한다.
  - 해당 파일이 RGB가 아닌 경우 `_preprocess`에서 컬러 변환이 자동으로 처리된다.

- **네트워크 스트림 (RTSP/HTTP 등)**
  ```bash
  python3 capture/frame_capture.py \
      --source stream \
      --stream-url rtsp://user:pass@camera-ip:554/stream \
      --mode multi \
      --stream-port 5000
```
  - OpenCV가 지원하는 URL이라면 그대로 사용할 수 있다(예: `rtsp://`, `http://`, `https://` 등).
  - 연결 실패 시 `CAMERA_CONFIG["stream"]`의 재시도/백엔드/타임아웃 옵션으로 동작을 조정할 수 있다. FFmpeg/GStreamer 백엔드 전환, 인증 정보 삽입도 구성 파일에서 지원한다.

- **Picamera2 (라즈베리 파이 전용)**  
  Picamera2 라이브러리가 설치되어 있어야 하며(`sudo apt install -y python3-picamera2`). 해상도/포맷/컨트롤 값은 `CAMERA_CONFIG["picamera"]`에서 세부 조정할 수 있고, 캡처 방식은 배열/버퍼 중 선택 가능하다.

#### 3.2.2 SSH/원격 접속 환경에서 스트림 접근

- 외부 PC에서 스트림을 확인하려면 서버의 IP와 포트가 접근 가능해야 한다.
- SSH 포트 포워딩을 활용하면 간단히 확인할 수 있다.
  ```bash
  ssh -L 5000:localhost:5000 user@server-ip
  ```
  이후 로컬 브라우저에서 `http://localhost:5000/stream.mjpg` 접속.
- 보안을 위해 내부에서만 접근해야 한다면 `--stream-host 127.0.0.1`로 바인딩을 제한한다.

### 3.3 메인 엔트리

전체 시스템은 라즈베리 파이 운용을 염두에 둔 `main.py`로 실행할 수 있다.

```bash
python3 main.py --help
```

해당 스크립트는 `FrameCaptureSystem`을 래핑하며, 추후 블록체인 연동 및 UI/CLI 추가를 전제로 한다.

실행 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--source` | 캡처 소스 (`dummy`, `picamera`, `video`, `stream`) |
| `--video-path` | `--source video`일 때 필요한 파일 경로 |
| `--stream-url` | `--source stream`일 때 필요한 URL |
| `--mode` | 실행 방식 (`single`, `multi`), 기본은 `multi` |
| `--duration` | 실행 시간 제한(초), 미지정 시 무한 반복 |
| `--no-banner` | 시작 시 배너 출력 생략 |

예시:

```bash
python3 main.py --source dummy --mode single --duration 10 --no-banner
```

---

## 4. AI 모듈 사용 예시

각 AI 모듈은 아직 더미 로직이나 TODO가 남아 있지만, 인터페이스는 다음과 같이 고정되어 있다.

```python
from ai_model import inference

inference.setup()

pass_flag, result = inference.analyze(frame)  # frame: numpy.ndarray (H, W, 3) RGB
stats = inference.get_statistics()

inference.teardown()
```

모듈별 내부 구조:

- `ai_model/object_detection.py` : `ObjectDetector.detect()`
- `ai_model/liveness.py` : `LivenessDetector.verify()`
- `ai_model/deepfake.py` : `DeepfakeDetector.analyze()`

현재는 난수 기반 더미 데이터와 경고 로그만 출력한다. 각 TODO를 구현하면서 실제 모델 로직으로 대체해야 한다.

AI 모듈별 구현 지침은 `sw/ai_model/README.md`를 참고한다.

---

## 5. 설정/로그/공통 유틸리티

| 모듈 | 설명 |
| ---- | ---- |
| `common/config.py` | 카메라/모델/로깅 설정, 디렉터리 생성 |
| `common/logger.py` | 프로젝트 전역 로깅 설정 |
| `common/frame_buffer.py` | 스레드 안전 프레임 버퍼 & 성능 모니터링 |
| `common` 패키지 | `FrameBuffer`, `PerformanceMonitor`, `get_config` 등 노출 |
| `capture/preprocessor.py` | 현재 스켈레톤(입력 pass-through) 상태로, TODO 구현이 필요함 |

로그 파일은 기본적으로 `sw/logs/app.log`에 기록되며, TODO 영역을 호출하면 `WARNING` 로그로 알려 준다.

---

## 6. TODO 상세 가이드

각 TODO는 “함수명 + 코드 라인 범위 + 해야 할 일” 형식으로 주석에 남겨두었다.  
대표적인 구현 항목은 아래와 같다.

| 위치 | TODO 요약 |
| ---- | -------- |
| `ai_model/object_detection.py` | TFLite 모델 로딩, YOLO 전/후처리, 추론 파이프라인 |
| `ai_model/liveness.py` | Mediapipe/Dlib 랜드마크 검출, 움직임/3D 일관성, 판정 로직 |
| `ai_model/deepfake.py` | 픽셀 패턴, 압축 아티팩트, 시간 일관성 분석 |
| `capture/frame_capture.py` | 멀티프로세스 모드 고도화(재연결, 헬스체크, 프로세스 종료 복구) |
| `capture/preprocessor.py` | 고급 리사이즈/정규화/노이즈 제거/대비 강화 옵션 |
| `capture/camera_source.py` | 추가 센서/제어 파라미터 노출, 특수 스트림(암호화/토큰) 지원 |

`ai_model/object_detection.py`의 `detect()`는 전처리 → 추론 → 후처리 호출 흐름이 이미 들어 있으므로, `_load_model`, `_preprocess`, `_postprocess`를 실제 모델에 맞게 구현하면 바로 사용할 수 있다.
`capture/preprocessor.py`는 현재 모든 메서드가 스켈레톤(pass-through)이므로, 전처리 로직을 실제로 작성해야 한다.

TODO 주석 예시:

```
# TODO(ObjectDetector._load_model, L42-L60): TFLite 인터프리터 로딩
#  - Interpreter 생성 및 allocate_tensors 호출 흐름 작성
#  - input/output details 캐싱 및 예외 처리 포함
```

---

## 7. 자주 묻는 질문(FAQ) & 트러블슈팅

### Q1. `pip install -r requirements.txt` 실행 중 `python-prctl` 관련 오류가 발생한다.
- 라즈베리 파이가 아닌 환경이라면 `picamera2` 항목을 주석 처리한 상태로 유지한다.
- 만약 Pi 환경이라면 `sudo apt-get install libcap-dev` 후 다시 설치한다.

### Q2. 테스트 실행 시 경고 로그가 너무 많다.
- TODO를 구현하기 전까지는 경고 로그가 의도된 동작이다.
- 로그 레벨을 조정하려면 `common/logger.py`의 설정을 수정한다.

### Q3. MJPEG 스트리머가 포트 바인딩에 실패한다.
- 이미 사용 중인 포트일 수 있다. `--stream-port` 값을 변경한다.
- 서버 방화벽이나 SSH 포트 포워딩 설정도 확인해야 한다.

### Q4. 실제 모델을 연결하려면?
- `ai_model` 하위 TODO를 순차적으로 구현한 뒤, 모델 파일을 `sw/ai_model/models/`에 배치하고 config 경로를 수정한다.

---

## 8. 향후 계획 및 권장 워크플로

1. **캡처 파이프라인 확장** – Picamera2, 스트림 재연결, 성능 최적화.
2. **AI 모델 통합** – TFLite/ONNX/PyTorch 모델을 실제로 로딩하고 추론 결과를 검증.
3. **테스트 보강** – TODO 구현 후 통합 테스트, 성능·회귀 테스트 추가.

코드를 확장할 때는 테스트 &rarr; 실행 &rarr; 로그 확인 순으로 검증하는 것을 권장

---

## 9. 참고 자료

- OpenCV 공식 문서: https://docs.opencv.org/
- Mediapipe: https://developers.google.com/mediapipe
- TFLite Runtime: https://www.tensorflow.org/lite/guide/python
- Picamera2: https://github.com/raspberrypi/picamera2
