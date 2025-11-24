// Types.bsv
// 공통 타입 정의 및 인터페이스

package Types;

import Vector::*;

// -----------------------------------------------------------------------------
// 타임스탬프 / PUF 공통 파라미터
// -----------------------------------------------------------------------------

typedef 64 TIMESTAMP_WIDTH;        // Unix timestamp 비트 폭

// PUF 구성 파라미터
typedef 64  PUF_WIDTH;             // PUF ID 비트 폭
typedef 128 PUF_RO_COUNT;          // Ring Oscillator 수 (ID 비트당 2개)
typedef 16  PUF_COUNTER_WIDTH;     // RO 주파수 카운터 비트 폭
typedef 256 PUF_SAMPLE_CYCLES;     // 하나의 측정에 사용되는 사이클 수

typedef TLog#(PUF_SAMPLE_CYCLES) PUF_SAMPLE_COUNTER_WIDTH;
typedef TLog#(TAdd#(PUF_WIDTH, 1)) PUF_HAMMING_WIDTH;

endpackage
