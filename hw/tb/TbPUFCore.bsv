package TbPUFCore;

import Types::*;
import PUFCore::*;

typedef enum { INIT, START_PUF, WAIT_PUF, READ_FIRST, WAIT_CLEAR1, RESTART,
               WAIT_PUF2, READ_SECOND, WAIT_CLEAR2, DONE } TbPUFState deriving (Bits, Eq);

typedef enum { ECC_INIT, ECC_START, ECC_WAIT1, ECC_READ1, ECC_RESTART,
               ECC_WAIT2, ECC_READ2, ECC_DONE } TbECCState deriving (Bits, Eq);

module mkTbPUFCore(Empty);

  PUFCore dut <- mkPUFCore();

  Reg#(TbPUFState) st <- mkReg(INIT);
  Reg#(Bit#(PUF_WIDTH)) firstID <- mkReg(0);
  Reg#(UInt#(16)) waitCycles <- mkReg(0);

  function Action fail(String msg);
    action
      $display("TB FAIL: %s", msg);
      $finish(1);
    endaction
  endfunction

  function Action pass(String msg);
    action
      $display("TB OK  : %s", msg);
    endaction
  endfunction

  rule r_init (st == INIT);
    $display("=== PUFCore TB start ===");
    st <= START_PUF;
  endrule

  rule r_start (st == START_PUF);
    if (dut.ready)
      fail("ready asserted before start");
    dut.start();
    pass("start() issued");
    st <= WAIT_PUF;
  endrule

  rule r_wait (st == WAIT_PUF);
    if (dut.ready) begin
      waitCycles <= 0;
      st <= READ_FIRST;
    end
    else begin
      waitCycles <= waitCycles + 1;
      if (waitCycles == 16'hffff)
        fail("timeout waiting for first ready");
    end
  endrule

  rule r_read_first (st == READ_FIRST);
    let id <- dut.getID();
    $display("TB OK  : first ID captured: 0x%0h", id);
    firstID <= id;
    waitCycles <= 0;
    st <= WAIT_CLEAR1;
  endrule

  rule r_wait_clear1 (st == WAIT_CLEAR1);
    if (!dut.ready) begin
      $display("TB OK  : ready dropped after first getID");
      st <= RESTART;
    end
    else begin
      waitCycles <= waitCycles + 1;
      if (waitCycles == 16'hffff)
        fail("timeout waiting ready low after first getID");
    end
  endrule

  rule r_restart (st == RESTART);
    dut.start();
    st <= WAIT_PUF2;
  endrule

  rule r_wait2 (st == WAIT_PUF2);
    if (dut.ready) begin
      waitCycles <= 0;
      st <= READ_SECOND;
    end
    else begin
      waitCycles <= waitCycles + 1;
      if (waitCycles == 16'hffff)
        fail("timeout waiting for second ready");
    end
  endrule

  rule r_read_second (st == READ_SECOND);
    let id <- dut.getID();
    $display("TB OK  : second ID captured: 0x%0h", id);
    if (id != firstID)
      fail("PUF core not deterministic in simulation");
    waitCycles <= 0;
    st <= WAIT_CLEAR2;
  endrule

  rule r_wait_clear2 (st == WAIT_CLEAR2);
    if (!dut.ready) begin
      $display("TB OK  : ready dropped after second getID");
      st <= DONE;
    end
    else begin
      waitCycles <= waitCycles + 1;
      if (waitCycles == 16'hffff)
        fail("timeout waiting ready low after second getID");
    end
  endrule

  rule r_done (st == DONE);
    $display("=== PUFCore TB PASS ===");
    $finish(0);
  endrule

endmodule

module mkTbPUFCoreECC(Empty);

  PUFCoreECC dut <- mkPUFCoreECC();

  Reg#(TbECCState) st <- mkReg(ECC_INIT);
  Reg#(Bit#(PUF_WIDTH)) firstID <- mkReg(0);

  function Action fail(String msg);
    action
      $display("TB FAIL: %s", msg);
      $finish(1);
    endaction
  endfunction

  function Action pass(String msg);
    action
      $display("TB OK  : %s", msg);
    endaction
  endfunction

  rule r_ecc_init (st == ECC_INIT);
    $display("=== PUFCoreECC TB start ===");
    st <= ECC_START;
  endrule

  rule r_ecc_start (st == ECC_START);
    dut.start();
    st <= ECC_WAIT1;
  endrule

  rule r_ecc_wait1 (st == ECC_WAIT1);
    if (dut.ready)
      st <= ECC_READ1;
  endrule

  rule r_ecc_read1 (st == ECC_READ1);
    let id <- dut.getID();
    firstID <= id;
    $display("TB OK  : ECC first ID: 0x%0h", id);
    st <= ECC_RESTART;
  endrule

  rule r_ecc_restart (st == ECC_RESTART);
    dut.start();
    st <= ECC_WAIT2;
  endrule

  rule r_ecc_wait2 (st == ECC_WAIT2);
    if (dut.ready)
      st <= ECC_READ2;
  endrule

  rule r_ecc_read2 (st == ECC_READ2);
    let id <- dut.getID();
    let hDist = dut.getHammingDistance();
    $display("TB OK  : ECC second ID: 0x%0h", id);
    $display("    Hamming distance vs first: %0d", hDist);
    if (hDist != 0)
      fail("expected identical IDs in deterministic sim");
    st <= ECC_DONE;
  endrule

  rule r_ecc_done (st == ECC_DONE);
    $display("=== PUFCoreECC TB PASS ===");
    $finish(0);
  endrule

endmodule

endpackage
