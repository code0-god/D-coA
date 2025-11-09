package TimeStampGenerator;

import Types::*;

// 인터페이스
interface TimeStampGenerator;
  method Action syncTime(Bit#(TIMESTAMP_WIDTH) unixTime);
  method ActionValue#(Bit#(TIMESTAMP_WIDTH)) getCurrentTimestamp();
  method ActionValue#(Bit#(32)) getCycleCount();
endinterface

// 합성 모듈: 파라미터는 Bits 여야 하므로 UInt#(32) 사용
(* synthesize *)
module mkTimeStampGenerator #(UInt#(32) clockFreqHz) (TimeStampGenerator);

  Reg#(Bit#(TIMESTAMP_WIDTH)) currentTime  <- mkReg(0);
  Reg#(UInt#(32))             cycleCounter <- mkReg(0);

  // 1초: clockFreqHz 사이클
  rule incrementSecond;
    if (cycleCounter == (clockFreqHz - 1)) begin
      currentTime  <= currentTime + 1;
      cycleCounter <= 0;
    end
    else begin
      cycleCounter <= cycleCounter + 1;
    end
  endrule

  method Action syncTime(Bit#(TIMESTAMP_WIDTH) unixTime);
    currentTime  <= unixTime;
    cycleCounter <= 0;
  endmethod

  method ActionValue#(Bit#(TIMESTAMP_WIDTH)) getCurrentTimestamp();
    return currentTime;
  endmethod

  method ActionValue#(Bit#(32)) getCycleCount();
    return pack(cycleCounter); // 인터페이스는 Bit#(32)
  endmethod

endmodule

endpackage
