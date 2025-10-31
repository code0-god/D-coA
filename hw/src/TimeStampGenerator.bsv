// TimestampGenerator.bsv

package GenerateTimestamp;

import Types::*;

// 인터페이스
interface GenerateTimestamp;
    method Action syncTime(Bit#(TIMESTAMP_WIDTH) unixTime);         // 절대초 동기화
    method ActionValue#(Bit#(TIMESTAMP_WIDTH)) getCurrentTimestamp(); // 현재 초 단위 타임스탬프
    method Bit#(32) getCycleCount();                                  // 1초 경계까지 사이클 카운트(디버깅)
endinterface

// 모듈
//  - clockFreqHz: 시스템 클록(Hz), 예: 100_000_000 (100MHz)
//  - 같은 rule에서 같은 레지스터를 두 번 쓰지 않도록 if/else 분기
(* synthesize *)
module mkGenerateTimestamp #(Integer clockFreqHz) (GenerateTimestamp);

    // 초 단위 Unix epoch
    Reg#(Bit#(TIMESTAMP_WIDTH)) currentTime <- mkReg(0);

    // 1초 카운팅용 사이클 카운터
    Reg#(Bit#(32)) cycleCounter <- mkReg(0);

    // 매 사이클: 사이클 카운트 -> 1초 경과 시 초 +1, 카운터 0
    rule incrementSecond;
        if (cycleCounter == fromInteger(clockFreqHz - 1)) begin
            currentTime  <= currentTime + 1;
            cycleCounter <= 0;
        end
        else begin
            cycleCounter <= cycleCounter + 1;
        end
    endrule

    // 동기화: 절대초 세팅 + 사이클 카운터 리셋
    method Action syncTime(Bit#(TIMESTAMP_WIDTH) unixTime);
        currentTime  <= unixTime;
        cycleCounter <= 0;
    endmethod

    // 현재 초 단위 타임스탬프
    method ActionValue#(Bit#(TIMESTAMP_WIDTH)) getCurrentTimestamp();
        return currentTime;
    endmethod

    // 디버깅용 사이클 카운터 노출
    method Bit#(32) getCycleCount();
        return cycleCounter;
    endmethod

endmodule

endpackage