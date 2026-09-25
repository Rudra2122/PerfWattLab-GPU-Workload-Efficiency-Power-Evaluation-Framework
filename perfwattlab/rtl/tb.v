`timescale 1ns/1ps
// tb.v — testbench with a scoreboard and configurable stimulus.
//
// Plusargs:
//   +period=N   valid asserted once every N cycles (default 4 → 25% duty)
//   +cycles=N   stimulus cycles (default 400)
//   +hold=1     hold a,b,c constant on invalid cycles (correlated inputs);
//               default 0 = new random a,b,c every cycle (worst case for baseline)
//   +seed=S     $random seed
//   +vcd=file   waveform file name
// Compile with -DUSE_BASELINE for the baseline design.
//
// Prints one parseable line:  SCOREBOARD design=... transactions=N mismatches=M pending=P
module tb;
  reg clk = 0;
  always #5 clk = ~clk;

  reg rst = 1, valid = 0;
  reg [7:0]  a = 0, b = 0;
  reg [15:0] c = 0;
  wire [16:0] y;
  wire out_valid;

  integer period, cycles, hold, seed, i;
  reg [8*256-1:0] vcd;

`ifdef USE_BASELINE
  mac_baseline  dut(.clk(clk), .rst(rst), .valid(valid), .a(a), .b(b), .c(c), .y(y), .out_valid(out_valid));
  localparam NAME = "baseline";
`else
  mac_optimized dut(.clk(clk), .rst(rst), .valid(valid), .a(a), .b(b), .c(c), .y(y), .out_valid(out_valid));
  localparam NAME = "optimized";
`endif

  // expected-result FIFO
  reg [16:0] expq [0:4095];
  integer head = 0, tail = 0, n_tx = 0, n_bad = 0;

  // scoreboard: sample on the negedge, after the DUT's posedge updates
  always @(negedge clk) begin
    if (!rst && valid) begin
      expq[tail] = a * b + c;
      tail = tail + 1;
    end
    if (!rst && out_valid) begin
      if (head == tail || y !== expq[head]) begin
        n_bad = n_bad + 1;
        if (n_bad <= 5) $display("MISMATCH t=%0t y=%0d expected=%0d", $time, y, expq[head]);
      end
      head = head + 1;
      n_tx = n_tx + 1;
    end
  end

  initial begin
    if (!$value$plusargs("period=%d", period)) period = 4;
    if (!$value$plusargs("cycles=%d", cycles)) cycles = 400;
    if (!$value$plusargs("hold=%d", hold))     hold   = 0;
    if (!$value$plusargs("seed=%d", seed))     seed   = 1;
    if (!$value$plusargs("vcd=%s", vcd))       vcd    = (NAME == "baseline") ? "baseline.vcd" : "optimized.vcd";
    $dumpfile(vcd);
    $dumpvars(0, dut);

    repeat (5) @(posedge clk);
    rst <= 0;
    for (i = 0; i < cycles; i = i + 1) begin
      @(posedge clk);
      valid <= ((i % period) == 0);
      if (!hold || (i % period) == 0) begin
        a <= $random(seed);
        b <= $random(seed);
        c <= $random(seed);
      end
    end
    @(posedge clk) valid <= 0;
    repeat (6) @(posedge clk);
    $display("SCOREBOARD design=%0s transactions=%0d mismatches=%0d pending=%0d",
             NAME, n_tx, n_bad, tail - head);
    $finish;
  end
endmodule
