// 8N1 UART receiver and transmitter, CPB clocks per bit.
module uart_rx #(parameter CPB = 27) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);
    reg [1:0] sync;
    reg [15:0] cnt;
    reg [3:0] bitn;
    reg busy;
    reg [7:0] sh;
    always @(posedge clk) begin
        sync <= {sync[0], rx};
        valid <= 1'b0;
        if (rst) begin
            busy <= 1'b0; cnt <= 0; bitn <= 0; sync <= 2'b11;
        end else if (!busy) begin
            if (sync[1] == 1'b0) begin  // start bit edge
                busy <= 1'b1; cnt <= CPB / 2; bitn <= 0;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else begin
            cnt <= CPB - 1;
            if (bitn == 0) begin
                if (sync[1] != 1'b0) busy <= 1'b0;  // false start
                bitn <= 1;
            end else if (bitn <= 8) begin
                sh <= {sync[1], sh[7:1]};
                bitn <= bitn + 1'b1;
            end else begin  // stop bit
                busy <= 1'b0;
                if (sync[1]) begin data <= sh; valid <= 1'b1; end
            end
        end
    end
endmodule

module uart_tx #(parameter CPB = 27) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       send,
    output reg        tx,
    output wire       ready
);
    reg [15:0] cnt;
    reg [3:0] bitn;
    reg [9:0] sh;
    reg busy;
    assign ready = !busy;
    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0; tx <= 1'b1;
        end else if (!busy) begin
            tx <= 1'b1;
            if (send) begin
                sh <= {1'b1, data, 1'b0}; busy <= 1'b1; cnt <= 0; bitn <= 0;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else begin
            if (bitn == 10) begin
                busy <= 1'b0; tx <= 1'b1;
            end else begin
                tx <= sh[0]; sh <= {1'b1, sh[9:1]}; bitn <= bitn + 1'b1; cnt <= CPB - 1;
            end
        end
    end
endmodule

// Byte FIFO, 2^AB entries, registered read (data valid the clock after re).
module fifo #(parameter AB = 10) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] wdata,
    input  wire       we,
    input  wire       re,
    output reg  [7:0] rdata,
    output wire       empty,
    output wire       full
);
    reg [7:0] mem [0:(1 << AB) - 1];
    reg [AB:0] wp, rp;
    assign empty = (wp == rp);
    assign full = (wp[AB-1:0] == rp[AB-1:0]) && (wp[AB] != rp[AB]);
    always @(posedge clk) begin
        if (rst) begin
            wp <= 0; rp <= 0;
        end else begin
            if (we && !full) begin mem[wp[AB-1:0]] <= wdata; wp <= wp + 1'b1; end
            if (re && !empty) begin rdata <= mem[rp[AB-1:0]]; rp <= rp + 1'b1; end
        end
    end
endmodule
