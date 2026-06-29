module dut (
    input  wire       clk_a,
    input  wire       rst_a,
    input  wire       clk_b,
    input  wire       rst_b,
    input  wire [7:0] data_in,
    input  wire       valid_in,
    output reg  [7:0] data_out,
    output reg        valid_out
);

    localparam [2:0] STRETCH = 3'd2;

    reg [7:0] data_q;
    reg [2:0] cnt;

    always @(posedge clk_a) begin
        if (rst_a) begin
            data_q <= 8'd0;
            cnt    <= 3'd0;
        end else if (valid_in) begin
            data_q <= data_in;
            cnt    <= STRETCH;
        end else if (cnt != 3'd0) begin
            cnt    <= cnt - 3'd1;
        end
    end

    wire req_a = (cnt != 3'd0);

    reg q1, q2, q3;
    reg [7:0] p1, p2;

    always @(posedge clk_b) begin
        if (rst_b) begin
            q1 <= 1'b0; q2 <= 1'b0; q3 <= 1'b0;
            p1 <= 8'd0; p2 <= 8'd0;
            data_out  <= 8'd0;
            valid_out <= 1'b0;
        end else begin
            q1 <= req_a;  q2 <= q1;  q3 <= q2;
            p1 <= data_q; p2 <= p1;
            valid_out <= q2;
            data_out  <= p2;
        end
    end

endmodule
