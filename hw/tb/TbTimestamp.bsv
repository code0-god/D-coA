// TbGenerateTimestamp.bsv
package TbGenerateTimestamp;

import Types::*;
import TimestampGenerator::*;

// 테스트 진행 상태
typedef enum {
  INIT,
  SYNC_1000,
  CHECK_1000,
  WAIT_1S,
  CHECK_1001,
  WAIT_2S,
  CHECK_1002,
  MID_PROGRESS,       // 중간 진행(몇 사이클 경과)
  SYNC_2000,          // 중간에 동기화
  CHECK_2000,
  WAIT_1S_AFTER_SYNC,
  CHECK_2001,
  DONE
} TbState deriving (Bits, Eq);

module mkTbTimestampGenerator(Empty);

  // 시뮬 속도↑ 위해 10Hz로 1초 구성 (10 클록 = 1초)
  Integer F = 10;

  TimestampGenerator dut <- mkTimestampGenerator(F);

  Reg#(TbState)       st       <- mkReg(INIT);
  Reg#(UInt#(32))     waitCnt  <- mkReg(0);

  // 유틸: 실패 시 메시지 출력 후 종료
  function Action fail(string m);
    action
      $display("TB FAIL: %s", m);
      $finish(1);
    endaction
  endfunction

  // 유틸: 통과 로그
  function Action ok(string m);
    action
      $display("TB OK  : %s", m);
    endaction
  endfunction

  // 시작
  rule r_init (st == INIT);
    $display("=== TB start ===");
    st <= SYNC_1000;
  endrule

  // 절대초 1000으로 동기화
  rule r_sync_1000 (st == SYNC_1000);
    dut.syncTime(64'd1000);
    st <= CHECK_1000;
  endrule

  // 동기화 직후 값/카운터 검사
  rule r_check_1000 (st == CHECK_1000);
    let ts <- dut.getCurrentTimestamp();
    let cc  =  dut.getCycleCount();
    if (ts != 64'd1000 || cc != 32'd0)
      fail($format("after sync(1000): ts=%0d cc=%0d (expect ts=1000, cc=0)", ts, cc));
    else begin
      ok("sync(1000) check passed");
      st <= WAIT_1S;
    end
  endrule

  // 1초 기다리기 (10 사이클)
  rule r_wait_1s (st == WAIT_1S);
    waitCnt <= waitCnt + 1;
    if (waitCnt == fromInteger(F - 1)) begin
      waitCnt <= 0;
      st <= CHECK_1001;
    end
  endrule

  // 1초 후 ts==1001, cc==0 확인
  rule r_check_1001 (st == CHECK_1001);
    let ts <- dut.getCurrentTimestamp();
    let cc  =  dut.getCycleCount();
    if (ts != 64'd1001 || cc != 32'd0)
      fail($format("after 1s: ts=%0d cc=%0d (expect ts=1001, cc=0)", ts, cc));
    else begin
      ok("1s increment to 1001");
      st <= WAIT_2S;
    end
  endrule

  // 또 1초(10 사이클) 대기
  rule r_wait_2s (st == WAIT_2S);
    waitCnt <= waitCnt + 1;
    if (waitCnt == fromInteger(F - 1)) begin
      waitCnt <= 0;
      st <= CHECK_1002;
    end
  endrule

  // 두 번째 초 증가 확인
  rule r_check_1002 (st == CHECK_1002);
    let ts <- dut.getCurrentTimestamp();
    let cc  =  dut.getCycleCount();
    if (ts != 64'd1002 || cc != 32'd0)
      fail($format("after 2s: ts=%0d cc=%0d (expect ts=1002, cc=0)", ts, cc));
    else begin
      ok("2s increment to 1002");
      st <= MID_PROGRESS;
    end
  endrule

  // 중간 몇 사이클 진행(예: 3틱 진행 후 동기화 테스트)
  rule r_mid_prog (st == MID_PROGRESS);
    waitCnt <= waitCnt + 1;
    if (waitCnt == 3) begin
      waitCnt <= 0;
      st <= SYNC_2000;
    end
  endrule

  // 중간에 절대초 2000으로 동기화
  rule r_sync_2000 (st == SYNC_2000);
    dut.syncTime(64'd2000);
    st <= CHECK_2000;
  endrule

  // 동기화 직후 ts=2000, cc=0 확인
  rule r_check_2000 (st == CHECK_2000);
    let ts <- dut.getCurrentTimestamp();
    let cc  =  dut.getCycleCount();
    if (ts != 64'd2000 || cc != 32'd0)
      fail($format("after mid-sync(2000): ts=%0d cc=%0d (expect ts=2000, cc=0)", ts, cc));
    else begin
      ok("mid-sync to 2000");
      st <= WAIT_1S_AFTER_SYNC;
    end
  endrule

  // 동기화 후 1초 경과
  rule r_wait_after (st == WAIT_1S_AFTER_SYNC);
    waitCnt <= waitCnt + 1;
    if (waitCnt == fromInteger(F - 1)) begin
      waitCnt <= 0;
      st <= CHECK_2001;
    end
  endrule

  // 1초 뒤 ts==2001, cc==0 확인
  rule r_check_2001 (st == CHECK_2001);
    let ts <- dut.getCurrentTimestamp();
    let cc  =  dut.getCycleCount();
    if (ts != 64'd2001 || cc != 32'd0)
      fail($format("after 1s post-sync: ts=%0d cc=%0d (expect ts=2001, cc=0)", ts, cc));
    else begin
      ok("post-sync increment to 2001");
      st <= DONE;
    end
  endrule

  // 종료
  rule r_done (st == DONE);
    $display("=== TB PASS ===");
    $finish(0);
  endrule

endmodule

endpackage
