// Testbench for fpga/rtl/core.v, driven by files from fpga/run_tb.py.
//
// stim.txt:  n_ctx n_steps n_init codes_entry_pc
//            n_init lines: ctx addr value          (host writes before the run)
//            then, for each step and each context, one line: entry dy
//            (entry 0 = STEP with dy, 1 = CODES only, dy ignored)
// out.txt:   one line per step and context: ctx step v0 .. v15 n0 .. n3 overflow
`timescale 1ns/1ps
module tb_core;
    parameter W = 16, WC = 12, F = 14, RW = 26, AW = 58, CTXB = 5, PCB = 10, CEDIV = 1;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;
    // Core clock enable: high one clock in CEDIV (CEDIV = 1: always high).
    integer cecnt = 0;
    always @(posedge clk) cecnt <= (cecnt + 1) % CEDIV;
    wire ce = (cecnt == 0);
    reg start = 0;
    reg [PCB-1:0] start_pc = 0;
    reg [CTXB-1:0] ctx = 0;
    wire busy;
    reg host_we = 0, host_re = 0;
    reg [7:0] host_addr = 0;
    reg [RW-1:0] host_wdata = 0;
    wire [RW-1:0] host_rdata;
    wire overflow;

    core #(.W(W), .WC(WC), .F(F), .RW(RW), .AW(AW), .CTXB(CTXB), .PCB(PCB), .ROM_FILE("rom.hex")) dut (
        .clk(clk), .ce(ce), .rst(rst), .start(start), .start_pc(start_pc), .ctx(ctx), .busy(busy),
        .host_we(host_we), .host_addr(host_addr), .host_wdata(host_wdata), .host_re(host_re),
        .host_rdata(host_rdata), .overflow(overflow));

    integer fi, fo, r, n_ctx, n_steps, n_init, codes_pc, i, s, c, a, entry, ok;
    integer v, dyv;
    integer cycles, total_cycles;
    reg signed [RW-1:0] rd;

    task wr(input integer cc, input integer addr, input integer val);
        begin
            @(negedge clk); ctx = cc; host_addr = addr; host_wdata = val; host_we = 1;
            @(negedge clk); host_we = 0;
        end
    endtask

    task rdreg(input integer cc, input integer addr, output reg signed [RW-1:0] val);
        begin
            @(negedge clk); ctx = cc; host_addr = addr; host_re = 1;
            @(negedge clk); host_re = 0; val = host_rdata;
        end
    endtask

    initial begin
        fi = $fopen("stim.txt", "r");
        fo = $fopen("out.txt", "w");
        r = $fscanf(fi, "%d %d %d %d", n_ctx, n_steps, n_init, codes_pc);
        repeat (4) @(negedge clk);
        rst = 0;
        for (i = 0; i < n_init; i = i + 1) begin
            r = $fscanf(fi, "%d %d %d", c, a, v);
            wr(c, a, v);
        end
        total_cycles = 0;
        for (s = 0; s < n_steps; s = s + 1) begin
            for (c = 0; c < n_ctx; c = c + 1) begin
                r = $fscanf(fi, "%d %d", entry, dyv);
                if (entry == 0) wr(c, 16, dyv);
                @(negedge clk); ctx = c; start_pc = (entry == 0) ? 0 : codes_pc; start = 1;
                @(negedge clk); start = 0;
                cycles = 1;
                while (busy) begin @(negedge clk); cycles = cycles + 1; end
                total_cycles = total_cycles + cycles;
                $fwrite(fo, "%0d %0d", c, s);
                for (a = 0; a < 16; a = a + 1) begin rdreg(c, a, rd); $fwrite(fo, " %0d", rd); end
                for (a = 17; a < 21; a = a + 1) begin rdreg(c, a, rd); $fwrite(fo, " %0d", rd); end
                $fwrite(fo, " %0d %0d\n", overflow, cycles);
            end
        end
        $fclose(fo);
        $display("done, mean cycles per run %0d", total_cycles / (n_steps * n_ctx));
        $finish;
    end
endmodule
