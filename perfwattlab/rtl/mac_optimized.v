module mac_optimized (
  input  wire        clk,
  input  wire        rst,
  input  wire        valid,
  input  wire [7:0]  a,
  input  wire [7:0]  b,
  input  wire [15:0] c,
  output reg  [15:0] y
);
  reg [7:0]  a_r;
  reg [7:0]  b_r;
  reg [15:0] c_r;
  reg [15:0] prod;

  always @(posedge clk) begin
    if (rst) begin
      a_r  <= 0;
      b_r  <= 0;
      c_r  <= 0;
      prod <= 0;
      y    <= 0;
    end else if (valid) begin
      // Only latch inputs when valid is asserted.
      // When valid=0, hold state — multiplier inputs stay stable → no toggles.
      a_r  <= a;
      b_r  <= b;
      c_r  <= c;
      prod <= a_r * b_r;
      y    <= prod + c_r;
    end
    // else: hold everything — zero switching activity
  end
endmodule
