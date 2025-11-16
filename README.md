<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="left">

<img src="D-coA.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# D-COA

<em>지능형 검증으로 미디어 무결성 보장</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/last-commit/code0-god/D-coA?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/code0-god/D-coA?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/code0-god/D-coA?style=flat&color=0080ff" alt="repo-language-count">

<em>이 프로젝트는 다음 도구와 기술을 기반으로 빌드되었습니다:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
<img src="https://img.shields.io/badge/MediaPipe-0097A7.svg?style=flat&logo=MediaPipe&logoColor=white" alt="MediaPipe">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Pytest-0A9EDC.svg?style=flat&logo=Pytest&logoColor=white" alt="Pytest">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">

</div>
<br>

---

## 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
  - [Project Index](#project-index)
- [시작하기](#시작하기)
  - [사전 준비](#사전-준비)
  - [설치](#설치)
  - [사용법](#사용법)
  - [테스트](#테스트)
- [로드맵](#로드맵)

---

## 개요

D-coA는 실시간 영상 캡처, AI 분석, 블록체인 검증을 결합해 미디어의 진위 여부를 검증하는 통합 플랫폼입니다. 오브젝트 디텍션, 딥페이크 탐지, 라이브니스 검증을 모듈화된 아키텍처에 통합해 다양한 하드웨어 환경에서 일관된 신뢰성을 제공합니다.

**왜 D-coA인가?**

- 🧩 **모듈형 AI 통합:** Object Detection, Deepfake, Liveness 모델을 필요에 맞게 조합해 종합적인 분석 파이프라인을 구성할 수 있습니다.
- 🚀 **실시간 처리:** Camera Source에서 입력되는 프레임을 전처리·추론·후처리까지 스트리밍 속도로 처리합니다.
- 🔒 **Blockchain Validation:** 검증 결과와 메타데이터를 블록체인에 고정해 위변조 여부를 추적합니다.
- 🎥 **라이브 모니터링:** MJPEG 스트리밍으로 처리된 프레임을 실시간 확인하며 디버깅할 수 있습니다.
- ⚙️ **HW/SW 확장성:** Raspberry Pi, 서버, FPGA와 같은 다양한 환경에서 구동되도록 공통 인터페이스를 제공합니다.
- 🛠️ **개발 친화성:** 구조화된 로깅, 구성 관리, 테스트 스위트를 통해 빠르게 실험하고 유지보수할 수 있습니다.

---

## 주요 기능

| 아이콘 | 컴포넌트 | 설명 |
| :--- | :--- | :--- |
| ⚙️ | **Architecture** | 코어 로직·데이터 처리·유틸리티를 분리한 모듈형 설계와 BSV 기반 HW 가속 모듈을 함께 제공합니다. |
| 🔩 | **Code Quality** | 일관된 코드 스타일, 타입 안전한 BSV/Python 구성, pytest·pytest-cov 기반 정적 검사 흐름을 유지합니다. |
| 📄 | **Documentation** | README, 인라인 주석, 설정 가이드를 통해 빠른 온보딩과 재현성을 지원합니다. |
| 🔌 | **Integrations** | NumPy, OpenCV, MediaPipe 등 Python 생태계와 Bluespec SystemVerilog, Shell Script를 유기적으로 엮었습니다. |
| 🧩 | **Modularity** | TimestampGenerator 같은 하드웨어 모듈과 Python 오케스트레이션 코드를 분리해 독립적으로 확장할 수 있습니다. |
| 🧪 | **Testing** | pytest 기반 단위 테스트와 mock 데이터를 통해 AI·Capture·HW 경계를 검증합니다. |
| ⚡️ | **Performance** | Bluespec 모듈로 고속 처리를 수행하고, Python 측에서는 Streaming/Buffer 최적화를 제공합니다. |

---

## 프로젝트 구조

```
.
├── hw/                # Bluespec HW 모듈과 테스트 인프라
├── sw/                # Python 기반 Capture · AI · Common 유틸리티
│   ├── ai_model/      # Object Detection, Deepfake, Liveness 로직
│   ├── capture/       # 카메라 입력, 프레임 버퍼, 스트리머
│   └── common/        # 공통 Config, Logger, Shared Memory
├── README.md          # 현재 문서
└── D-coA.png          # 프로젝트 로고
```

### Project Index

| 경로 | 설명 |
| :--- | :--- |
| `sw/main.py` | Capture 파이프라인, AI 분석, Blockchain 연동을 초기화하고 실행 흐름을 제어합니다. |
| `sw/common/logger.py` | 콘솔/파일 출력을 아우르는 공통 로깅 인터페이스를 제공해 모듈 전반의 진단을 단일화합니다. |
| `sw/common/config.py` | 디렉터리 경로, 카메라 설정, AI 모델 매개변수 등 시스템 전역 구성을 관리합니다. |
| `sw/capture/frame_capture.py` | Camera Source, Preprocessor, Inference 호출을 연결해 실시간 스트리밍 루프를 구성합니다. |
| `sw/ai_model/inference.py` | Frame 데이터를 받아 Object Detection, Liveness, Deepfake 파이프라인을 조합해 결과를 생성합니다. |
| `hw/src/TimeStampGenerator.bsv` | 미디어 프레임에 부착할 정밀 타임스탬프 메타데이터를 생성하는 Bluespec 모듈입니다. |
| `hw/runtb.sh` | `bsc` 호출, Bluesim 링크, 테스트 실행을 한 번에 처리하는 HW 시뮬레이션 스크립트입니다. |

상세한 파일 설명은 각 서브 디렉터리 README에서 확인할 수 있습니다.

---

## 시작하기

### 사전 준비

필수 의존성:

- **Programming Language:** Python 3.10+
- **Package Manager:** pip
- (옵션) Bluespec Compiler `bsc` 및 Bluesim 환경 (`hw/` 모듈 시뮬레이션 시 필요)

### 설치

소스를 클론하고 의존성을 설치합니다.

1. **레포지토리 클론**

    ```sh
    git clone https://github.com/code0-god/D-coA
    ```

2. **프로젝트 디렉터리로 이동**

    ```sh
    cd D-coA
    ```

3. **Python 패키지 설치**

    ```sh
    pip install -r sw/requirements.txt
    ```

### 사용법

기본 실행 진입점은 배포 환경에 맞게 `python {entrypoint}` 형태로 호출하도록 구성되어 있습니다. Raspberry Pi에서 Capture만 구동하거나, 서버에서 Inference만 실행하는 등 시나리오에 맞게 엔트리포인트를 선택하세요.

### 테스트

pytest 기반 테스트 스위트를 다음 명령으로 실행합니다.

```sh
pytest
```

하드웨어 시뮬레이션은 `hw` 디렉터리에서 아래와 같이 구동할 수 있습니다.

```sh
cd hw
./runtb.sh TbTimeStampGenerator::mkTbTimeStampGenerator +bscvcd
```

---

## 로드맵

- [x] **Phase 1:** Capture/AI 파이프라인 초기 버전 완성 및 Logger·Config 정비
- [x] **Phase 2:** Bluespec TimestampGenerator 시뮬레이션 플로 구축, Python-HW 연동 인터페이스 설계
- [ ] **Phase 3:** PUF, SHA-256, ECC 등 HW 보안 모듈 통합 및 Raspberry Pi ↔ FPGA 통신 검증
- [ ] **Phase 4:** Blockchain Anchoring, End-to-End 실시간 데모, CI 파이프라인 고도화
- [ ] **Phase 5:** 문서/테스트 보완, 사용자 가이드 및 배포 자동화

---

<div align="left"><a href="#top">⬆ Return</a></div>

---
