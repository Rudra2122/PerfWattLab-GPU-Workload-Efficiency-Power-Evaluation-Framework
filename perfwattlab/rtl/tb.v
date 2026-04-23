`timescale 1ns/1ps
// tb.v — Testbench for mac_baseline and mac_optimized
//
// Compile and simulate:
//   iverilog -g2012 -o sim_baseline -DUSE_BASELINE tb.v mac_baseline.v && vvp sim_baseline
//   iverilog -g2012 -o sim_opt tb.v mac_optimized.v && vvp sim_opt
//
// Then count toggles:
//   python perfwattlab/rtl/toggle_counter.py baseline.vcd optimized.vcd

module tb;
  reg clk = 0;
  always #5 clk = ~clk;  // 100 MHz

  reg rst   = 1;
  reg valid = 0;
  reg [7:0]  a = 0;
  reg [7:0]  b = 0;
  reg [15:0] c = 0;

  wire [15:0] y;

  integer i;

  `ifdef USE_BASELINE
    mac_baseline dut(.clk(clk), .rst(rst), .valid(valid), .a(a), .b(b), .c(c), .y(y));
    initial begin
      $dumpfile("baseline.vcd");
      $dumpvars(0, dut);
    end
  `else
    mac_optimized dut(.clk(clk), .rst(rst), .valid(valid), .a(a), .b(b), .c(c), .y(y));
    initial begin
      $dumpfile("optimized.vcd");
      $dumpvars(0, dut);
    end
  `endif

  initial begin
    rst   = 1;
    valid = 0;
    a = 0; b = 0; c = 0;

    repeat (5) @(posedge clk);
    rst = 0;

    // Drive 400 cycles with valid asserted 1-in-4 cycles
    // This models real inference: the MAC is active ~25% of the time
    for (i = 0; i < 400; i = i + 1) begin
      @(posedge clk);
      if ((i % 4) == 0) valid <= 1'b1;
      else               valid <= 1'b0;
      a <= $random;
      b <= $random;
      c <= $random;
    end

    repeat (10) @(posedge clk);
    $finish;
  end
endmodule
