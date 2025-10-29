## AI 모델 사용 가이드

현재 `sw/ai_model` 패키지는 **모델 로딩과 추론 로직이 구현되지 않은 상태**로 시작한다.  
구조와 TODO 주석만 준비되어 있으므로, 실제 모델을 연결하려면 아래 절차를 따른다.

---

## 1. 모델 파일 준비

### 1.1 객체 탐지 (YOLOv5n TFLite 예시)

1. Ultralytics YOLOv5 GitHub에서 nano 모델을 TFLite로 변환하거나, 미리 변환된 파일을 구한다.  
2. 변환된 파일을 `sw/ai_model/models/` 디렉터리에 저장한다.
   - 기본 경로는 `sw/ai_model/models/yolov5n.tflite`입니다.
   - 경로는 `common/config.py`의 `MODEL_CONFIG["yolo"]["model_path"]`로 제어됩니다.

필요한 경우 `MODEL_CONFIG["yolo"]`를 수정하여 다른 모델 파일이나 임계값을 지정할 수 있다.

### 1.2 라이브니스 (Mediapipe / Dlib)

- 기본 설정은 `MODEL_CONFIG["liveness"]["model"] = "mediapipe"`.
- Mediapipe는 pip 설치 시 모델이 자동 포함되므로 별도 파일이 필요 없음.
- Dlib을 사용하려면 `shape_predictor_68_face_landmarks.dat`와 같은 모델 파일을 준비하고, 경로를 설정해야 함.
- TODO 주석이 있는 `_load_detector`에서 모델 파일 로딩 코드를 직접 작성.

### 1.3 딥페이크

- 현재는 규칙 기반/추가 모델 없이 더미 지표만 계산.
- 심층 모델을 사용하게 되면 해당 모델 파일을 `MODEL_CONFIG["deepfake"]`에 등록하고, `DeepfakeDetector` 내부 TODO를 채워 넣기.

---

## 2. 코드 수정 포인트

아래 함수들은 현재 더미 구현이며, 실제 모델을 연결하는 코드를 작성해야 한다.

| 파일 | 함수 | 해야 할 일 |
| ---- | ---- | -------- |
| `object_detection.py` | `_load_model` | TFLite `Interpreter` 생성, `allocate_tensors()`, `input/output details` 캐싱 |
|  | `_preprocess` | 입력 리사이즈, 정규화, 채널 변환, 배치 차원 추가 (YOLO 규격) |
|  | `_postprocess` | 모델 출력 파싱, NMS 적용, COCO 클래스 매핑 |
|  | `detect` | `_load_model` / `_preprocess` / `_postprocess`가 제대로 구현되어 있다는 전제로, TFLite 추론 호출 흐름이 연결되어 있음 |
| `liveness.py` | `_load_detector` | Mediapipe FaceMesh 초기화 또는 Dlib 모델 로딩 |
|  | `_detect_landmarks` | 영상에서 얼굴 랜드마크 추출 |
|  | `_calculate_movement`, `_check_3d_consistency`, `verify` | 움직임/3D 일관성 기반 라이브니스 판단 |
| `deepfake.py` | `_analyze_pixel_patterns`, `_check_compression_artifacts`, `_verify_temporal_consistency` | 주파수/압축/시간 분석 알고리즘 구현 |

각 파일에 TODO 주석과 라인 범위가 명시되어 있으니 참고한다.

특히 `object_detection.py`의 `detect()`는 전처리 → 추론 → 후처리 호출 흐름이 이미 구현되어 있으므로, 위 표에 있는 `_load_model`, `_preprocess`, `_postprocess`를 실제 모델에 맞게 채워 넣으면 바로 동작한다.

---

## 3. 샘플 코드 (TFLite 로드)

```python
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="models/yolov5n.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 준비
input_tensor = np.random.rand(1, 640, 640, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]["index"], input_tensor)

# 추론 실행
interpreter.invoke()
outputs = interpreter.get_tensor(output_details[0]["index"])
```

해당 예시는 `object_detection.ObjectDetector`의 TODO 구현에 활용할 수 있다.

---

## 4. 모델 관리 팁

- 모델 파일은 `.gitignore`에 등록하여 저장소에는 포함하지 않는 것이 권장됨.
- 버전 관리를 위해 모델 파일명에 날짜나 버전을 붙여 관리하거나, `MODEL_CONFIG`에 버전 정보를 추가.

---

## 5. 테스트와 검증

모델을 연결한 후에는 아래 테스트를 실행해 기본 동작을 검증한다.

```bash
cd /srv/D-coA/sw
python3 -m unittest ai_model.tests.test_inference
```

TODO 구현 상태에 맞춰 테스트 코드를 업데이트하는 것도 잊지 마세요.
