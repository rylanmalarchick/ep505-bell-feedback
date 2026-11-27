// UART-level testbench for fpga/rtl/top.v, driven by fpga/run_tb_top.py.
// host.txt: n, then n byte values (host to board), sent back to back.
// board.txt: every byte the board sends, one per line.
`timescale 1ns/1ps
module tb_top;
    parameter W = 16, WC = 12, RW = 26, AW = 58, CTXB = 5, PCB = 10, CODES_PC = 734, CPB = 27;
    parameter N_EXPECT = 0;
    reg clk = 0;
    always #18.5 clk = ~clk;  // about 27 MHz
    reg rx = 1;
    wire tx;
    top #(.W(W), .WC(WC), .RW(RW), .AW(AW), .CTXB(CTXB), .PCB(PCB), .CODES_PC(CODES_PC), .CPB(CPB),
          .ROM_FILE("rom.hex")) dut (.clk(clk), .uart_rx(rx), .uart_tx(tx));

    integer fi, fo, r, n, i, b, k, got;
    integer n_rx = 0;
    always @(posedge clk) if (dut.rx_valid) n_rx = n_rx + 1;
    task send_byte(input [7:0] v);
        begin
            rx = 0; repeat (CPB) @(posedge clk);
            for (k = 0; k < 8; k = k + 1) begin rx = v[k]; repeat (CPB) @(posedge clk); end
            rx = 1; repeat (CPB) @(posedge clk);
        end
    endtask

    // receiver: sample the board's tx line
    reg [7:0] rb;
    integer j;
    initial begin
        fo = $fopen("board.txt", "w");
        got = 0;
        forever begin
            @(negedge tx);
            repeat (CPB / 2) @(posedge clk);
            for (j = 0; j < 8; j = j + 1) begin repeat (CPB) @(posedge clk); rb[j] = tx; end
            repeat (CPB) @(posedge clk);
            $fwrite(fo, "%0d\n", rb);
            got = got + 1;
        end
    end

    initial begin
        fi = $fopen("host.txt", "r");
        r = $fscanf(fi, "%d", n);
        repeat (100) @(posedge clk);
        for (i = 0; i < n; i = i + 1) begin
            r = $fscanf(fi, "%d", b);
            send_byte(b[7:0]);
        end
        // wait for the replies (or time out)
        for (i = 0; i < 200000 && got < N_EXPECT; i = i + 1) @(posedge clk);
        repeat (2000) @(posedge clk);
        $fclose(fo);
        $display("tb_top: end state st=%0d cmd=%h got=%0d nargs=%0d byte_pend=%0d rf_empty=%0d tf_empty=%0d busy=%0d",
                 dut.st, dut.cmd, dut.got, dut.nargs, dut.byte_pend, dut.rf_empty, dut.tf_empty, dut.busy);
        $display("tb_top: sent %0d bytes, board uart_rx delivered %0d, received %0d", n, n_rx, got);
        $finish;
    end
endmodule
