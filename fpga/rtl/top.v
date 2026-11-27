// Tang Nano 20K top level: UART command interface to the microcode core.
//
// Commands (host to board), all multi-byte fields big-endian:
//   A1 ctx dy[23:0]        write dy (signed 24-bit) to register 16 of ctx, run STEP,
//                          reply the 4 codes, 2 bytes each (8 bytes)
//   A4 ctx                 run CODES on ctx, reply the 4 codes (8 bytes)
//   A2 ctx reg v[31:0]     write register reg of ctx (reg >= 128: shared constant), reply 5A
//   A3 ctx reg             read register reg of ctx, reply v[31:0] (sign-extended)
// Any other byte in command position is skipped.
module top #(
    parameter W = 16, WC = 12, RW = 26, AW = 58, CTXB = 8, PCB = 10,
    parameter CODES_PC = 10'd734,
    parameter CPB = 27,                 // 27 MHz / 1 Mbaud
    parameter CORE_DIV = 1,             // the core advances one clock in CORE_DIV
    parameter ROM_FILE = "rom.hex"
) (
    input  wire clk,
    input  wire uart_rx,
    output wire uart_tx
);
    localparam F = W - 2;

    // core clock enable
    reg [3:0] cediv = 0;
    always @(posedge clk) cediv <= (cediv == CORE_DIV - 1) ? 4'd0 : cediv + 1'b1;
    wire ce = (cediv == 0);

    // power-on reset
    reg [4:0] por = 0;
    wire rst = !por[4];
    always @(posedge clk) if (!por[4]) por <= por + 1'b1;

    // UART and FIFOs
    wire [7:0] rx_byte; wire rx_valid;
    uart_rx #(.CPB(CPB)) urx (.clk(clk), .rst(rst), .rx(uart_rx), .data(rx_byte), .valid(rx_valid));
    reg rf_re; wire [7:0] rf_q; wire rf_empty, rf_full;
    fifo #(.AB(10)) rxf (.clk(clk), .rst(rst), .wdata(rx_byte), .we(rx_valid), .re(rf_re), .rdata(rf_q),
                         .empty(rf_empty), .full(rf_full));
    reg [7:0] tf_d; reg tf_we; wire tf_re; wire [7:0] tf_q; wire tf_empty, tf_full;
    fifo #(.AB(10)) txf (.clk(clk), .rst(rst), .wdata(tf_d), .we(tf_we), .re(tf_re), .rdata(tf_q),
                         .empty(tf_empty), .full(tf_full));
    wire tx_ready;
    reg tx_pend, tx_send;
    assign tf_re = !tf_empty && tx_ready && !tx_pend && !tx_send;
    always @(posedge clk) begin
        if (rst) begin tx_pend <= 0; tx_send <= 0; end
        else begin
            tx_send <= 1'b0;
            if (tf_re) tx_pend <= 1'b1;             // byte arrives in tf_q next clock
            else if (tx_pend) begin tx_pend <= 1'b0; tx_send <= 1'b1; end
        end
    end
    uart_tx #(.CPB(CPB)) utx (.clk(clk), .rst(rst), .data(tf_q), .send(tx_send), .tx(uart_tx), .ready(tx_ready));

    // core
    reg start; reg [PCB-1:0] start_pc; reg [CTXB-1:0] ctx; wire busy;
    reg host_we, host_re; reg [7:0] host_addr; reg [RW-1:0] host_wdata; wire [RW-1:0] host_rdata;
    wire overflow;
    core #(.W(W), .WC(WC), .F(F), .RW(RW), .AW(AW), .CTXB(CTXB), .PCB(PCB), .ROM_FILE(ROM_FILE)) u_core (
        .clk(clk), .ce(ce), .rst(rst), .start(start), .start_pc(start_pc), .ctx(ctx), .busy(busy),
        .host_we(host_we), .host_addr(host_addr), .host_wdata(host_wdata), .host_re(host_re),
        .host_rdata(host_rdata), .overflow(overflow));

    // command parser
    localparam S_CMD = 0, S_ARGS = 1, S_EXEC = 2, S_RUN = 3, S_WAIT = 4, S_READ = 5, S_READW = 6, S_SEND = 7;
    reg [3:0] st;
    reg [7:0] cmd;
    reg [7:0] args [0:5];
    reg [2:0] nargs, got;
    // RX FIFO read handshake. rf_re is registered, so the FIFO reads one clock
    // after the parser asks, and rf_q is valid one clock after that:
    // rd_stage 1 = rf_re is high this clock, 2 = rf_q holds the byte.
    reg [1:0] rd_stage;
    wire byte_pend = (rd_stage == 2'd2);
    reg [2:0] ridx;                     // code index being read
    reg [31:0] outw;
    reg [2:0] nout;                     // bytes left to send from outw (MSB first)
    reg [2:0] after_send;               // 0 back to S_CMD, 1 next code
    // host_rdata as a 32-bit two's-complement value (sign-extend or keep the low bits)
    wire signed [63:0] rd64 = $signed(host_rdata);
    wire [31:0] rdx = rd64[31:0];

    always @(posedge clk) begin
        rf_re <= 1'b0; tf_we <= 1'b0; start <= 1'b0; host_we <= 1'b0; host_re <= 1'b0;
        if (rst) begin
            st <= S_CMD; rd_stage <= 2'd0; got <= 0;
        end else case (st)
            S_CMD: begin
                if (rd_stage == 2'd1) rd_stage <= 2'd2;
                else if (byte_pend) begin
                    rd_stage <= 2'd0;
                    cmd <= rf_q; got <= 0;
                    case (rf_q)
                        8'hA1: begin nargs <= 4; st <= S_ARGS; end
                        8'hA2: begin nargs <= 6; st <= S_ARGS; end
                        8'hA3: begin nargs <= 2; st <= S_ARGS; end
                        8'hA4: begin nargs <= 1; st <= S_ARGS; end
                        default: ;
                    endcase
                end else if (!rf_empty) begin rf_re <= 1'b1; rd_stage <= 2'd1; end
            end
            S_ARGS: begin
                if (rd_stage == 2'd1) rd_stage <= 2'd2;
                else if (byte_pend) begin
                    rd_stage <= 2'd0;
                    args[got] <= rf_q;
                    got <= got + 1'b1;
                    if (got + 1'b1 == nargs) st <= S_EXEC;
                end else if (!rf_empty) begin rf_re <= 1'b1; rd_stage <= 2'd1; end
            end
            S_EXEC: begin
                ctx <= args[0][CTXB-1:0];
                case (cmd)
                    8'hA1: begin
                        host_addr <= 8'd16;
                        host_wdata <= $signed({args[1], args[2], args[3]});
                        host_we <= 1'b1;
                        start_pc <= 0;
                        st <= S_RUN;
                    end
                    8'hA4: begin start_pc <= CODES_PC; st <= S_RUN; end
                    8'hA2: begin
                        host_addr <= args[1];
                        host_wdata <= $signed({args[2], args[3], args[4], args[5]});
                        host_we <= 1'b1;
                        outw <= {8'h5A, 24'd0}; nout <= 1; after_send <= 0; st <= S_SEND;
                    end
                    8'hA3: begin host_addr <= args[1]; host_re <= 1'b1; ridx <= 7; st <= S_READW; end
                    default: st <= S_CMD;
                endcase
            end
            S_RUN: begin start <= 1'b1; st <= S_WAIT; end
            S_WAIT: if (!start && !busy) begin ridx <= 0; st <= S_READ; end
            S_READ: begin host_addr <= 8'd17 + ridx; host_re <= 1'b1; st <= S_READW; end
            S_READW: if (!host_re) begin
                if (ridx == 7) begin outw <= rdx; nout <= 4; after_send <= 0; end
                else begin outw <= {rdx[15:0], 16'd0}; nout <= 2; after_send <= (ridx == 3) ? 3'd0 : 3'd1; end
                st <= S_SEND;
            end
            S_SEND: begin
                if (nout == 0) begin
                    if (after_send == 1) begin ridx <= ridx + 1'b1; st <= S_READ; end
                    else st <= S_CMD;
                end else if (!tf_full) begin
                    tf_d <= outw[31:24]; tf_we <= 1'b1; outw <= {outw[23:0], 8'd0}; nout <= nout - 1'b1;
                end
            end
            default: st <= S_CMD;
        endcase
    end
endmodule
