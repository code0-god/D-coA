## 캡처 모듈 가이드

`sw/capture` 패키지는 카메라 소스 추상화, 프레임 전처리, 캡처 파이프라인, MJPEG 스트리머를 포함한다. 대부분의 파이프라인은 동작 가능한 기본 구현이 있으며, 전처리/고급 제어 항목은 TODO로 남아 있다.

---

## 1. 구성 요소

| 파일 | 역할 |
| --- | --- |
| `camera_source.py` | Picamera2, 비디오 파일, 더미, 웹 스트림을 포함한 카메라 소스 구현. `CAMERA_CONFIG` 기반으로 해상도/컨트롤/재시도 옵션을 조정할 수 있다. |
| `preprocessor.py` | 현재 스켈레톤(pass-through) 상태로, 리사이즈/정규화/노이즈 제거/대비 향상 구현이 TODO로 남아 있다. |
| `frame_capture.py` | 재시도 로직과 추론 호출 흐름을 갖춘 캡처 → 전처리 → AI 모듈 호출 → 스트리밍 파이프라인을 수행한다. |
| `frame_streamer.py` | OpenCV `cv2.imencode`를 이용한 간단한 MJPEG HTTP 스트리머. SSH 환경에서 `imshow` 대신 사용한다. |
| `tests/` | 더미 소스, 전처리, 캡처 시스템에 대한 기본 단위 테스트. |

---

## 2. 사용 예시

```bash
python3 capture/frame_capture.py \
    --source dummy \
    --mode multi \
    --stream-port 5000
```

- `--source`는 `dummy`, `video`, `stream`, `picamera` 중에서 선택한다.
- `--stream-port`를 0으로 두면 MJPEG 스트리머를 끈다. 기본값은 5000이다.
- 브라우저에서 `http://<호스트>:5000/stream.mjpg`로 접속하면 프레임을 확인할 수 있다.
- `multi` 모드는 캡처 스레드와 추론 프로세스를 분리하며, 큐 기반 IPC와 결과 수집 스레드를 사용한다.
- 캡처 루프에는 실패 재시도(최대 30회)와 FPS 조절, 스트리머 연계 로직이 포함되어 있다.

다른 입력 소스 예시는 상위 README의 “3.2.1 입력 소스별 예시”를 참조한다.

### 2.1 주요 설정 포인트

- `common/config.py`의 `CAMERA_CONFIG["picamera"]`에서 센서 해상도, 포맷, 컨트롤(FrameRate 등), capture_array 사용 여부를 정의한다.
- `CAMERA_CONFIG["stream"]`은 재시도 횟수, 백엔드 선택(`auto`/`ffmpeg`/`gstreamer`), 인증 정보, FFmpeg 옵션, 타임아웃을 설정한다.
- 전처리 파이프라인은 아직 pass-through 상태이며, 필요한 경우 `FramePreprocessor`의 TODO를 구현해 사용한다.

---

## 3. 남아 있는 TODO

| 위치 | 할 일 |
| --- | --- |
| `camera_source.py` | 추가 센서 제어(수동 노출/화이트밸런스), 특수 인증/프로토콜 대응 |
| `preprocessor.py` | 레터박스, 채널 변환, 고급 대비/노이즈 제거 옵션 구현 |
| `frame_capture.py` | 멀티프로세스 상태 모니터링, 프로세스 재시작/재연결 전략, 성능 로깅 고도화 |

실제 서비스를 위해서는 위 TODO를 순차적으로 구현한 뒤, 테스트(`python3 -m unittest capture.tests.test_capture`)를 실행해 변경 사항을 검증한다.

---

## 4. 참고

- MJPEG 스트림을 외부에서 확인해야 한다면 SSH 포트 포워딩(`ssh -L 5000:localhost:5000 ...`)을 활용한다.
- 스트림 URL 기반 소스를 사용할 때 OpenCV가 연결에 실패하면 FFmpeg/GStreamer 백엔드로의 확장을 고려한다.
