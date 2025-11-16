# AI-블록체인 기반 미디어 무결성 하드웨어 모듈

라즈베리파이 5와 연동되는 FPGA/ASIC RTL 모듈로, BlueSpec SystemVerilog (BSV)로 구현된 하드웨어 보안 서브시스템을 목표로 합니다. 현재 리포지토리에는 타임스탬프 제너레이터 참조 구현이 포함되어 있으며, 아래 계획에 따라 모듈을 확장합니다.

## 개요

### 주요 기능 로드맵

- **PUF (Physical Unclonable Function)**: 디바이스 고유 ID 생성
- **SHA-256 해시 엔진**: NIST FIPS 180-4 준수 암호화 해시
- **ECC 디지털 서명**: NIST P-256 ECDSA 서명 (시뮬레이션 버전)
- **하드웨어 타임스탬프**: RTC/GPS 기반 정확한 시각 기록
- **SPI 인터페이스**: 라즈베리파이와의 통신

현재 버전은 타임스탬프 경로(`TimeStampGenerator`)를 제공하며, 다른 기능은 계획 중입니다.

## 디렉토리 구조

```
hw/
├── src/                  # BSV 소스
│   ├── TimeStampGenerator.bsv
│   └── Types.bsv
├── tb/                   # 테스트벤치
│   └── TbTimeStampGenerator.bsv
├── build/                # bsc/bscsim 아티팩트 (runtb.sh가 채움)
├── runtb.sh              # 통합 컴파일·시뮬레이션 스크립트
├── dump.vcd              # 예시 파형 (옵션)
└── README.md
```

> 향후 구조 계획
>
> ```
> hw/
> ├── src/              # BSV 소스 코드
> │   ├── Types.bsv                  # 공통 타입 정의
> │   ├── PUFCore.bsv                # PUF 모듈
> │   ├── SHA256Engine.bsv           # SHA-256 해시 엔진
> │   ├── ECCSignature.bsv           # ECC 서명 모듈
> │   ├── TimestampGenerator.bsv     # 타임스탬프 제너레이터
> │   ├── SPIInterface.bsv           # SPI 슬레이브 인터페이스
> │   └── MediaIntegrityCore.bsv     # Top 모듈
> ├── tb/               # 테스트벤치
> │   ├── TbPUF.bsv
> │   └── TbSHA256.bsv
> ├── sim/              # 시뮬레이션 결과
> ├── syn/              # 합성 결과 및 제약 파일
> ├── doc/              # 문서
> └── scripts/          # 빌드/시뮬레이션 스크립트
> ```

## 개발 환경 설정

### 필수 도구

1. **Bluespec Compiler (bsc)**
   ```bash
   # BSC 설치 (Ubuntu 22.04 예시)
   wget https://github.com/B-Lang-org/bsc/releases/download/2023.07/bsc-2023.07-ubuntu-22.04.tar.gz
   tar -xzf bsc-2023.07-ubuntu-22.04.tar.gz
   export PATH="$PATH:$(pwd)/bsc-2023.07/bin"
   ```
2. **Bluesim (시뮬레이터)** – bsc 배포판에 포함
3. **FPGA 합성 도구(선택)** – Xilinx Vivado, Intel Quartus Prime 등

### 환경 변수 설정

```bash
export BLUESPECDIR=/path/to/bsc/lib
export PATH=$PATH:/path/to/bsc/bin
```

## 빌드 및 시뮬레이션

### TimeStampGenerator 하드웨어 개요

- 프레임 캡처 파이프라인에서 타임스탬프 메타데이터를 생성하는 Bluespec 설계
- 핵심 모듈: `src/TimeStampGenerator.bsv`, 타입 정의는 `src/Types.bsv`
- 검증용 테스트벤치: `tb/TbTimeStampGenerator.bsv`

### 스크립트 기반 워크플로 (`runtb.sh`)

1. Bluespec 툴이 PATH, `BLUESPECDIR`에 올바르게 설정되어 있는지 확인합니다.
2. 실행 권한을 부여합니다.
   ```bash
   cd hw
   chmod +x runtb.sh
   ```
3. 시뮬레이션을 실행합니다. 명시적으로 상위 패키지와 탑 모듈을 지정합니다.
   ```bash
   # 기본 실행 (콘솔 로그 출력)
   ./runtb.sh TbTimeStampGenerator::mkTbTimeStampGenerator

   # VCD 덤프 등 bsc 인수를 통과시키고 싶다면 추가 인자로 전달
   ./runtb.sh TbTimeStampGenerator::mkTbTimeStampGenerator +bscvcd
   ```

`runtb.sh`는 아래 단계를 자동화합니다.

1. 지정한 테스트벤치를 Bluesim 대상으로 컴파일 (`bsc -u -elab -sim`)
2. `build/sim.out` 실행 파일을 링크
3. 인자로 전달된 시뮬레이터 옵션과 함께 테스트를 구동

빌드 산출물은 모두 `hw/build/`에 모이며, 파형 출력은 `+bscvcd` 또는 `-V` 등의 옵션으로 제어할 수 있습니다. 파형 파일 기본 경로는 `dump.vcd`이며 스크립트나 테스트벤치에서 파일명을 바꿀 수 있습니다.

### 직접 bsc 명령 사용 (모듈별 실험)

```bash
# 소스 디렉터리로 이동
cd hw/src

# Types 모듈 컴파일
bsc -u Types.bsv

# TimeStampGenerator 모듈 단독 컴파일
bsc -u TimeStampGenerator.bsv

# tb에서 Bluesim 타겟 생성
cd ../tb
bsc -sim -g mkTbTimeStampGenerator -u TbTimeStampGenerator.bsv -p ../src:+:$BLUESPECDIR
bsc -sim -e mkTbTimeStampGenerator -o sim_ts
./sim_ts
```

향후 PUF, SHA-256, ECC 모듈이 추가되면 같은 방식으로 각 테스트벤치를 컴파일합니다.

### (계획) Makefile 통합

향후 `scripts/Makefile`을 추가해 다음 명령을 제공할 예정입니다.

```bash
make compile   # 모든 모듈 컴파일
make sim       # 등록된 테스트벤치 실행
make synth     # Vivado/Quartus 합성 진입
make clean     # build, sim 산출물 정리
```

## 아키텍처

### 파이프라인 플로우

```
1. SPI 수신: 라즈베리파이로부터 검증 요청
   ↓
2. PUF 인증: 디바이스 고유 ID 생성
   ↓
3. SHA-256 해싱: 프레임 데이터 해시
   ↓
4. ECC 서명: 해시에 디지털 서명
   ↓
5. 타임스탬프: 하드웨어 시각 기록
   ↓
6. 패킷 조립: SecurityPacket 생성
   ↓
7. SPI 송신: 라즈베리파이로 결과 전송
```

### 주요 인터페이스 (설계안)

`SecurityPacket` 구조체:

```bsv
typedef struct {
    Bit#(128) pufID;          // 디바이스 고유 ID
    Bit#(256) frameHash;      // SHA-256 해시
    Bit#(512) eccSignature;   // ECC 서명 (r || s)
    Bit#(64)  hwTimestamp;    // Unix timestamp
    Bit#(32)  sequenceNumber; // 단조 증가 카운터
    Bool      valid;          // 유효성 플래그
} SecurityPacket;
```

### 보안 고려사항

1. **Private Key 보호**: 하드웨어 보안 메모리에 저장
2. **Side-Channel Attack 방어**: 타이밍/전력 분석 대비
3. **PUF 재현성**: Error Correction Code 적용 필요

## 문제 해결

### 컴파일 에러

```bash
# 패키지 경로 확인
bsc -show-packages

# 의존성 체크
bsc -dep Types.bsv
```

### 시뮬레이션 실패

```bash
# Verbose 모드로 실행
./build/sim.out -V

# VCD 생성 (직접 bsc 호출)
bsc -sim -verilog -g mkTbTimeStampGenerator TbTimeStampGenerator.bsv
```

## 참고 자료

- [BlueSpec SystemVerilog Reference Guide](http://www.bluespec.com/forum/docs.php)
- [NIST FIPS 180-4: SHA-256 Specification](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
- [NIST FIPS 186-4: ECDSA Specification](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf)
