// mac_baseline.v — 3-stage registered MAC:  y = a*b + c  (one transaction per valid cycle)
//
// Stage 1: a_r, b_r, c_r   <= a, b, c
// Stage 2: prod <= a_r*b_r;  c_d <= c_r     (c is delayed so it stays paired with its a,b)
// Stage 3: y    <= prod + c_d
// A 3-bit valid shift register marks which output cycles carry real results (out_valid).
//
// BASELINE: every data register loads every cycle, whether or not the data is valid.
// v1 bug fixed here: v1 computed y <= prod + c_r, adding the c of the NEXT transaction.
module mac_baseline (
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
  reg [2:0]  v;                       // v[0]: stage1 holds valid data, v[1]: stage2, v[2]: stage3

  assign out_valid = v[2];

  always @(posedge clk) begin
    if (rst) begin
      a_r <= 0; b_r <= 0; c_r <= 0; c_d <= 0; prod <= 0; y <= 0; v <= 0;
    end else begin
      a_r  <= a;   b_r <= b;   c_r <= c;
      prod <= a_r * b_r;       c_d <= c_r;
      y    <= prod + c_d;
      v    <= {v[1:0], valid};
    end
  end
endmodule
