package TbTimeStampGenerator;

import Types::*;
import TimeStampGenerator::*;

typedef enum {
  INIT, SYNC_1000, CHECK_1000,
  WAIT_1S, CHECK_1001,
  WAIT_2S, CHECK_1002,
  MID_PROGRESS, SYNC_2000, CHECK_2000,
  WAIT_1S_AFTER_SYNC, CHECK_2001,
  DONE
} TbState deriving (Bits, Eq);

module mkTbTimeStampGenerator(Empty);

  Integer f = 10;  // 10Hz => 10 사이클 = 1초
  TimeStampGenerator dut <- mkTimeStampGenerator(fromInteger(f));

  Reg#(TbState)   st      <- mkReg(INIT);
  Reg#(UInt#(32)) waitCnt <- mkReg(0);

  function Action failMismatch(String lbl,
                               Bit#(TIMESTAMP_WIDTH) tsActual,
                               Bit#(32)             ccActual,
                               Bit#(TIMESTAMP_WIDTH) tsExpect,
                               Bit#(32)             ccExpect);
    action
      $display("TB FAIL: %s ts=%0d cc=%0d (expect ts=%0d, cc=%0d)",
               lbl, tsActual, ccActual, tsExpect, ccExpect);
      $finish(1);
    endaction
  endfunction

  function Action ok(String m);
    action
      $display("TB OK  : %s", m);
    endaction
  endfunction

  rule r_init (st == INIT);
    $display("=== TB start ===");
    st <= SYNC_1000;
  endrule

  rule r_sync_1000 (st == SYNC_1000);
    dut.syncTime(64'd1000);
    st <= CHECK_1000;
  endrule

  rule r_check_1000 (st == CHECK_1000);
    let ts <- dut.getCurrentTimestamp();
    let cc <- dut.getCycleCount();
    if (ts != 64'd1000 || cc != 32'd0)
      failMismatch("after sync(1000)", ts, cc, 64'd1000, 32'd0);
    else begin ok("sync(1000) check passed"); st <= WAIT_1S; end
  endrule

  rule r_wait_1s (st == WAIT_1S);
    if (waitCnt == fromInteger(f - 1)) begin
      waitCnt <= 0;
      st <= CHECK_1001;
    end
    else begin
      waitCnt <= waitCnt + 1;
    end
  endrule

  rule r_check_1001 (st == CHECK_1001);
    let ts <- dut.getCurrentTimestamp();
    let cc <- dut.getCycleCount();
    if (ts != 64'd1001 || cc != 32'd1)
      failMismatch("after 1s", ts, cc, 64'd1001, 32'd1);
    else begin ok("1s increment to 1001"); st <= WAIT_2S; end
  endrule

  rule r_wait_2s (st == WAIT_2S);
    if (waitCnt == fromInteger(f - 1)) begin
      waitCnt <= 0;
      st <= CHECK_1002;
    end
    else begin
      waitCnt <= waitCnt + 1;
    end
  endrule

  rule r_check_1002 (st == CHECK_1002);
    let ts <- dut.getCurrentTimestamp();
    let cc <- dut.getCycleCount();
    if (ts != 64'd1002 || cc != 32'd2)
      failMismatch("after 2s", ts, cc, 64'd1002, 32'd2);
    else begin ok("2s increment to 1002"); st <= MID_PROGRESS; end
  endrule

  rule r_mid_prog (st == MID_PROGRESS);
    if (waitCnt == 3) begin
      waitCnt <= 0;
      st <= SYNC_2000;
    end
    else begin
      waitCnt <= waitCnt + 1;
    end
  endrule

  rule r_sync_2000 (st == SYNC_2000);
    dut.syncTime(64'd2000);
    st <= CHECK_2000;
  endrule

  rule r_check_2000 (st == CHECK_2000);
    let ts <- dut.getCurrentTimestamp();
    let cc <- dut.getCycleCount();
    if (ts != 64'd2000 || cc != 32'd0)
      failMismatch("after mid-sync(2000)", ts, cc, 64'd2000, 32'd0);
    else begin ok("mid-sync to 2000"); st <= WAIT_1S_AFTER_SYNC; end
  endrule

  rule r_wait_after (st == WAIT_1S_AFTER_SYNC);
    if (waitCnt == fromInteger(f - 1)) begin
      waitCnt <= 0;
      st <= CHECK_2001;
    end
    else begin
      waitCnt <= waitCnt + 1;
    end
  endrule

  rule r_check_2001 (st == CHECK_2001);
    let ts <- dut.getCurrentTimestamp();
    let cc <- dut.getCycleCount();
    if (ts != 64'd2001 || cc != 32'd1)
      failMismatch("after 1s post-sync", ts, cc, 64'd2001, 32'd1);
    else begin ok("post-sync increment to 2001"); st <= DONE; end
  endrule

  rule r_done (st == DONE);
    $display("=== TB PASS ===");
    $finish(0);
  endrule

endmodule

endpackage
