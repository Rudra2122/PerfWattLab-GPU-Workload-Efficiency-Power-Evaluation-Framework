// mac_optimized.v — same MAC, VALID-GATED REGISTER UPDATES (operand isolation).
//
// Each stage's data registers load only when that stage receives valid data:
//   stage 1 enable = valid, stage 2 enable = v[0], stage 3 enable = v[1].
// When a stage is idle its registers HOLD, so the multiplier and adder inputs
// stay constant and the datapath does not switch.
//
// This is NOT clock gating: the clock still toggles every flop every cycle
// (a synthesis tool maps these to enable flops / hold muxes; converting them
// to integrated clock-gating cells would be a separate, verified step).
//
// Functionally equivalent to mac_baseline on valid transactions: same
// 3-cycle latency, same y whenever out_valid=1 (checked by the tb scoreboard).
module mac_optimized (
  input  wire        clk,
  input  wire        rst,
  input  wire        valid,
  input  wire [7:0]  a,
  input  wire [7:0]  b,
  input  wire [15:0] c,
  output reg  [16:0] y,
  output wire        out_valid
);
  reg [7:0]  a_r, b_r;
  reg [15:0] c_r, c_d, prod;
  reg [2:0]  v;

  assign out_valid = v[2];

  always @(posedge clk) begin
    if (rst) begin
      a_r <= 0; b_r <= 0; c_r <= 0; c_d <= 0; prod <= 0; y <= 0; v <= 0;
    end else begin
      if (valid) begin a_r <= a;  b_r <= b;  c_r <= c; end
      if (v[0])  begin prod <= a_r * b_r;  c_d <= c_r; end
      if (v[1])  begin y <= prod + c_d; end
      v <= {v[1:0], valid};            // 3 single-bit flops, always shift
    end
  end
endmodule
