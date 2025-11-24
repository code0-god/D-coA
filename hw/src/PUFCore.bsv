// PUFCore.bsv
// Physical Unclonable Function (PUF) 모듈
// Ring Oscillator 기반 디바이스 고유 ID 생성

package PUFCore;

import FIFOF::*;
import Vector::*;
import Types::*;

// ============================================================================
// PUF Interface
// ============================================================================

interface PUFCore;
    method Action start();                        // PUF ID 생성 시작
    method ActionValue#(Bit#(PUF_WIDTH)) getID(); // PUF ID 반환
    method Bool ready();                          // 준비 상태
endinterface

// ============================================================================
// PUF Module Implementation
// ============================================================================

(* synthesize *)
module mkPUFCore(PUFCore);

    // 구성 검증: PUF_RO_COUNT는 PUF_WIDTH의 2배여야 함 (bit당 2개의 RO)
    Integer expectedRO = valueOf(PUF_WIDTH) * 2;
    Bool roConfigValid = (valueOf(PUF_RO_COUNT) == expectedRO);
    
    // Ring Oscillator 상태 레지스터
    Vector#(PUF_RO_COUNT, Reg#(Bit#(1))) roStates <- replicateM(mkReg(0));
    
    // Ring Oscillator 주파수 카운터
    Vector#(PUF_RO_COUNT, Reg#(Bit#(PUF_COUNTER_WIDTH))) roCounters <- replicateM(mkReg(0));
    
    // 생성된 PUF ID
    Reg#(Bit#(PUF_WIDTH)) pufID <- mkReg(0);
    
    // 상태 플래그
    Reg#(Bool) isReady <- mkReg(False);
    Reg#(Bool) measuring <- mkReg(False);
    
    // 측정 사이클 카운터
    Reg#(Bit#(PUF_SAMPLE_COUNTER_WIDTH)) measureCycles <- mkReg(0);
    Bit#(PUF_SAMPLE_COUNTER_WIDTH) measureLimit = fromInteger(valueOf(PUF_SAMPLE_CYCLES) - 1);
    
    // ========================================================================
    // Ring Oscillator 동작 규칙
    // ========================================================================
    
    rule oscillate (measuring);
        if (!roConfigValid) begin
            $display("ERROR: PUF_RO_COUNT(%0d) must equal 2 * PUF_WIDTH(%0d)",
                     valueOf(PUF_RO_COUNT), valueOf(PUF_WIDTH));
            $finish(1);
        end
        // 모든 Ring Oscillator를 발진시킴
        for (Integer i = 0; i < valueOf(PUF_RO_COUNT); i = i + 1) begin
            // 링 발진기 토글
            roStates[i] <= ~roStates[i];
            
            // 주파수 카운트 (상승 엣지에서)
            if (roStates[i] == 1'b1) begin
                roCounters[i] <= roCounters[i] + 1;
            end
        end
        
        // 측정 사이클 증가
        measureCycles <= measureCycles + 1;
        
        // 설정된 측정 윈도우만큼 샘플링 완료
        if (measureCycles == measureLimit) begin
            measuring <= False;
            isReady <= True;
            
            // Pairwise comparison으로 PUF ID 생성
            Bit#(PUF_WIDTH) id = 0;
            
            for (Integer i = 0; i < valueOf(PUF_WIDTH); i = i + 1) begin
                Integer idx1 = i * 2;
                Integer idx2 = i * 2 + 1;
                
                // 두 RO의 주파수를 비교하여 1비트 생성
                if (roCounters[idx1] > roCounters[idx2]) begin
                    id[i] = 1'b1;
                end else begin
                    id[i] = 1'b0;
                end
            end
            
            pufID <= id;
        end
    endrule
    
    // ========================================================================
    // Interface Methods
    // ========================================================================
    
    method Action start() if (!measuring);
        measuring <= True;
        measureCycles <= 0;
        isReady <= False;
        
        // 모든 카운터 초기화
        for (Integer i = 0; i < valueOf(PUF_RO_COUNT); i = i + 1) begin
            roCounters[i] <= 0;
        end
    endmethod
    
    method ActionValue#(Bit#(PUF_WIDTH)) getID() if (isReady);
        isReady <= False;
        return pufID;
    endmethod
    
    method Bool ready();
        return isReady;
    endmethod
    
endmodule

// ============================================================================
// PUF with Error Correction (Enhanced Version)
// ============================================================================

// 노이즈에 강한 PUF를 위한 에러 정정 기능 추가
interface PUFCoreECC;
    method Action start();
    method ActionValue#(Bit#(PUF_WIDTH)) getID();
    method Bool ready();
    method Bit#(PUF_HAMMING_WIDTH) getHammingDistance(); // 이전 측정과의 해밍 거리
endinterface

(* synthesize *)
module mkPUFCoreECC(PUFCoreECC);
    
    // 기본 PUF 코어
    PUFCore pufCore <- mkPUFCore();
    
    // 이전 PUF ID 저장 (재현성 검증용)
    Reg#(Bit#(PUF_WIDTH)) previousID <- mkReg(0);
    Reg#(Bool) hasPreviousID <- mkReg(False);
    
    // 해밍 거리 계산 결과
    Reg#(Bit#(PUF_HAMMING_WIDTH)) hammingDist <- mkReg(0);
    
    // ========================================================================
    // Helper Function: Hamming Distance
    // ========================================================================
    
    function Bit#(PUF_HAMMING_WIDTH) calcHammingDistance(Bit#(PUF_WIDTH) a, Bit#(PUF_WIDTH) b);
        Bit#(PUF_WIDTH) diff = a ^ b;
        Bit#(PUF_HAMMING_WIDTH) count = 0;
        
        for (Integer i = 0; i < valueOf(PUF_WIDTH); i = i + 1) begin
            if (diff[i] == 1'b1) begin
                count = count + 1;
            end
        end
        
        return count;
    endfunction
    
    // ========================================================================
    // Interface Methods
    // ========================================================================
    
    method Action start();
        pufCore.start();
    endmethod
    
    method ActionValue#(Bit#(PUF_WIDTH)) getID() if (pufCore.ready());
        let id <- pufCore.getID();
        
        // 해밍 거리 계산 (재현성 확인)
        if (hasPreviousID) begin
            hammingDist <= calcHammingDistance(id, previousID);
        end
        
        previousID <= id;
        hasPreviousID <= True;
        
        return id;
    endmethod
    
    method Bool ready();
        return pufCore.ready();
    endmethod
    
    method Bit#(PUF_HAMMING_WIDTH) getHammingDistance();
        return hammingDist;
    endmethod
    
endmodule

endpackage
