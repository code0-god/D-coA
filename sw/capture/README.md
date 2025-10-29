## 캡처 모듈 가이드

`sw/capture` 패키지는 카메라 소스 추상화, 프레임 전처리, 캡처 파이프라인, MJPEG 스트리머를 포함한다. 아직 많은 TODO가 남아 있으므로, 실제 카메라/스트림과 연결하기 전에 아래 내용을 참고한다.

---

## 1. 구성 요소

| 파일 | 역할 |
| --- | --- |
| `camera_source.py` | Picamera2, 비디오 파일, 더미, 웹 스트림을 포함한 카메라 소스 구현. `create_camera_source()`로 인스턴스화한다. |
| `preprocessor.py` | 리사이즈/정규화/노이즈 제거/대비 향상 등 프레임 전처리 로직. 많은 고급 옵션이 TODO 상태다. |
| `frame_capture.py` | `FrameCaptureSystem`을 통해 캡처 → 전처리 → AI 모듈 호출 → 스트리밍 과정을 수행한다. |
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

다른 입력 소스 예시는 상위 README의 “3.2.1 입력 소스별 예시”를 참조한다.

---

## 3. 남아 있는 TODO

| 위치 | 할 일 |
| --- | --- |
| `camera_source.py` | Picamera2 실제 연동, 웹 스트림 재시도/타임아웃 처리 |
| `preprocessor.py` | 레터박스, 채널 변환, 고급 대비/노이즈 제거 옵션 구현 |
| `frame_capture.py` | 멀티프로세스 확장, 예외/재연결 처리, 성능 모니터링 고도화 |
| `frame_streamer.py` | JPEG 인코딩 실패/연결 종료 처리, 인증/HTTPS 지원 |

실제 서비스를 위해서는 위 TODO를 순차적으로 구현한 뒤, 테스트(`python3 -m unittest capture.tests.test_capture`)를 실행해 변경 사항을 검증한다.

---

## 4. 참고

- MJPEG 스트림을 외부에서 확인해야 한다면 SSH 포트 포워딩(`ssh -L 5000:localhost:5000 ...`)을 활용한다.
- 스트림 URL 기반 소스를 사용할 때 OpenCV가 연결에 실패하면 FFmpeg/GStreamer 백엔드로의 확장을 고려한다.

