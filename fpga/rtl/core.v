// Microcode core for the fixed-point filter and offset law.
//
// Executes the program of fpga/isa.py from a ROM, one instruction per clock,
// in a four-stage pipeline:
//   S0  fetch      rom[pc] -> s1
//   S1  read       register RAM (two copies, one per operand) and constants
//   S2  multiply   product of the two operands, registered
//   S3  execute    CLR, MAC, ADD on the accumulator; ST writes rnd/sat(ACC);
//                  DIV runs a restoring divider and stalls the pipeline
// The compiler spaces every read of a register at least 3 slots after the
// write of that register (isa.PIPE_GAP), so the pipeline needs no forwarding.
//
// Registers 0..31 live in RAM per context (ctx selects the bank of 32).
// Registers 32..127 are temporaries in a second RAM shared by all contexts
// (fpga/isa.py: a run writes every temporary before it reads it).
// Registers 128..143 are constants, shared by all contexts. A host port
// writes and reads registers while the core is idle.
//
// Each RAM copy has one write port and one registered read port, so that
// synthesis maps it to block RAM. The host read shares the read port of
// copy A while the core is idle; host_rdata is valid one clock after the
// address.
module core #(
    parameter W = 16,              // state word length
    parameter WC = 12,             // command code length
    parameter F = 14,              // fraction bits of a state word (W - 2)
    parameter RW = 26,             // register width (W + 10)
    parameter AW = 58,             // accumulator width (2 RW + 6)
    parameter CTXB = 8,            // log2 of the number of contexts
    parameter PCB = 10,            // ROM address bits
    parameter ROM_FILE = "rom.hex"
) (
    input  wire              clk,
    input  wire              ce,        // core clock enable: the pipeline advances only when ce = 1
    input  wire              rst,
    // run control
    input  wire              start,
    input  wire [PCB-1:0]    start_pc,
    input  wire [CTXB-1:0]   ctx,
    output reg               busy,
    // host register port (use only while busy = 0)
    input  wire              host_we,
    input  wire [7:0]        host_addr,
    input  wire [RW-1:0]     host_wdata,
    input  wire              host_re,
    output wire [RW-1:0]     host_rdata,
    // simulation aid: a stored value did not fit RW bits
    output reg               overflow
);
    localparam AB = CTXB + 5;      // per-context RAM address: {ctx, reg[4:0]}
    localparam DEPTH = 1 << AB;
    localparam OP_NOP = 3'd0, OP_CLR = 3'd1, OP_MAC = 3'd2, OP_ADD = 3'd3,
               OP_ST = 3'd4, OP_DIV = 3'd5, OP_END = 3'd7;
    localparam DB = 2 * F + 2;     // dividend bits
    // Sized constants: an unsized 1 << k is a 32-bit integer in Verilog.
    localparam [AW-1:0] ONE_A = {{(AW-1){1'b0}}, 1'b1};
    localparam [RW-1:0] ONE_R = {{(RW-1){1'b0}}, 1'b1};
    localparam [DB-1:0] ONE_D = {{(DB-1){1'b0}}, 1'b1};

    // ---------------- memories ----------------
    reg [39:0] rom [0:(1 << PCB) - 1];
    initial $readmemh(ROM_FILE, rom);
    reg [RW-1:0] ram_a [0:DEPTH-1];   // per-context registers, copy for operand a
    reg [RW-1:0] ram_b [0:DEPTH-1];   // copy for operand b
    reg [RW-1:0] tmp_a [0:127];       // shared temporaries (32..127 used)
    reg [RW-1:0] tmp_b [0:127];
    reg signed [RW-1:0] cst [0:15];
    reg [RW-1:0] qa_c, qb_c, qa_t, qb_t;  // registered read data
    reg sel_ta, sel_tb;                   // the registered read is a temporary
    wire [RW-1:0] qa = sel_ta ? qa_t : qa_c;
    wire [RW-1:0] qb = sel_tb ? qb_t : qb_c;

    // ---------------- pipeline registers ----------------
    reg [PCB-1:0] pc;
    reg           run;             // fetching
    reg           s1_v, s2_v, s3_v;
    reg [39:0]    s1_i, s2_i, s3_i;
    reg           s2_ca, s2_cb;              // operand is a constant
    reg signed [RW-1:0] s2_ka, s2_kb;        // constant values
    reg signed [RW-1:0] s3_a;
    reg signed [2*RW-1:0] s3_p;
    reg signed [AW-1:0] acc;
    wire stall;

    wire [2:0] s1_op = s1_i[38:36];
    wire [7:0] s1_x = s1_i[35:28], s1_y = s1_i[27:20];
    wire [7:0] s1_ra = (s1_op == OP_DIV) ? s1_y : s1_x;
    wire [7:0] s1_rb = s1_y;

    // read addresses: the pipeline while busy, the host port while idle
    wire [7:0] ia = busy ? s1_ra : host_addr;     // register index read on port a
    wire [AB-1:0] raddr_a = {ctx, ia[4:0]};
    wire [AB-1:0] raddr_b = {ctx, s1_rb[4:0]};
    wire is_tmp_a = (ia[6:5] != 2'b00);           // 32..127 (constants are handled apart)
    wire is_tmp_b = (s1_rb[6:5] != 2'b00);
    assign host_rdata = host_addr[7] ? cst[host_addr[3:0]] : qa;

    // operand values in S2 (qa, qb hold the reads of the instruction now in S2)
    wire signed [RW-1:0] opa = s2_ca ? s2_ka : $signed(qa);
    wire signed [RW-1:0] opb = s2_cb ? s2_kb : $signed(qb);

    // ---------------- S3 decode ----------------
    wire [2:0] s3_op = s3_i[38:36];
    wire [7:0] s3_x = s3_i[35:28], s3_y = s3_i[27:20];
    wire [1:0] s3_z = s3_i[19:18];

    // rnd(acc, n): to nearest, ties to even
    wire [5:0] n = s3_y[5:0];
    wire signed [AW-1:0] q = acc >>> n;
    wire [AW-1:0] mask = (n == 0) ? {AW{1'b0}} : ((ONE_A << n) - ONE_A);
    wire [AW-1:0] rem = acc & mask;
    wire [AW-1:0] half = (n == 0) ? {AW{1'b0}} : (ONE_A << (n - 1));
    wire up = (n != 0) && ((rem > half) || ((rem == half) && q[0]));
    wire signed [AW-1:0] rv = q + $signed({{(AW-1){1'b0}}, up});
    // saturation to W or WC bits
    wire signed [AW-1:0] lim_w = $signed(ONE_A << (W - 1));
    wire signed [AW-1:0] lim_c = $signed(ONE_A << (WC - 1));
    wire signed [AW-1:0] sv = (s3_z == 2'd1) ? ((rv >= lim_w) ? lim_w - 1 : (rv < -lim_w) ? -lim_w : rv) :
                              (s3_z == 2'd2) ? ((rv >= lim_c) ? lim_c - 1 : (rv < -lim_c) ? -lim_c : rv) : rv;
    wire signed [AW-1:0] rw_max = $signed(ONE_A << (RW - 1));
    wire fits = (sv < rw_max) && (sv >= -rw_max);

    // ---------------- divider ----------------
    reg         dv_busy, dv_done;
    reg [DB-1:0] dv_num, dv_q;
    reg [RW+1:0] dv_rem;
    reg [RW-1:0] dv_t;
    reg [6:0]   dv_i;
    wire signed [RW-1:0] tfloor = $signed(ONE_R << (F - 2));
    wire signed [RW-1:0] tsel = (s3_a < tfloor) ? tfloor : s3_a;
    wire [RW+1:0] dv_sh = {dv_rem[RW:0], dv_num[DB-1]};
    assign stall = s3_v && (s3_op == OP_DIV) && !dv_done;

    // ---------------- the single write port of both RAM copies ----------------
    wire w_host = !busy && host_we && !host_addr[7];
    wire w_st = busy && ce && s3_v && (s3_op == OP_ST);
    wire w_div = busy && ce && s3_v && (s3_op == OP_DIV) && dv_done;
    wire we = w_host || w_st || w_div;
    wire [7:0] widx = w_host ? host_addr : s3_x;
    wire we_tmp = we && (widx[6:5] != 2'b00);
    wire we_ctx = we && (widx[6:5] == 2'b00);
    wire [AB-1:0] waddr = {ctx, widx[4:0]};
    wire [RW-1:0] wdata = w_host ? host_wdata : w_st ? sv[RW-1:0] : dv_q[RW-1:0];

    always @(posedge clk) begin
        if (we_ctx) begin
            ram_a[waddr] <= wdata;
            ram_b[waddr] <= wdata;
        end
        if (we_tmp) begin
            tmp_a[widx[6:0]] <= wdata;
            tmp_b[widx[6:0]] <= wdata;
        end
        // Hold the read data during a stall: qa, qb belong to the instruction in S2.
        if (!(busy && (stall || !ce))) begin
            qa_c <= ram_a[raddr_a];
            qb_c <= ram_b[raddr_b];
            qa_t <= tmp_a[ia[6:0]];
            qb_t <= tmp_b[s1_rb[6:0]];
            sel_ta <= is_tmp_a;
            sel_tb <= is_tmp_b;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            run <= 1'b0; busy <= 1'b0; s1_v <= 1'b0; s2_v <= 1'b0; s3_v <= 1'b0;
            acc <= 0; dv_busy <= 1'b0; dv_done <= 1'b0; overflow <= 1'b0; pc <= 0;
        end else begin
            if (!busy && host_we && host_addr[7]) cst[host_addr[3:0]] <= host_wdata;

            // ---- start ----
            if (!busy && start) begin
                busy <= 1'b1; run <= 1'b1; pc <= start_pc;
                s1_v <= 1'b0; s2_v <= 1'b0; s3_v <= 1'b0;
            end else if (busy && !stall && ce) begin
                // S0 -> S1
                s1_v <= run;
                s1_i <= rom[pc];
                if (run) pc <= pc + 1'b1;
                // S1 -> S2: the RAM reads of s1 land in qa, qb at this edge
                s2_v <= s1_v;
                s2_i <= s1_i;
                s2_ca <= s1_ra[7]; s2_cb <= s1_rb[7];
                s2_ka <= cst[s1_ra[3:0]]; s2_kb <= cst[s1_rb[3:0]];
                if (s1_v && s1_op == OP_END) run <= 1'b0;  // stop fetching after END
                // S2 -> S3
                s3_v <= s2_v;
                s3_i <= s2_i;
                s3_a <= opa;
                s3_p <= opa * opb;
            end

            // ---- S3 execute ----
            if (busy && s3_v && ce) begin
                case (s3_op)
                    OP_CLR: acc <= 0;
                    OP_MAC: acc <= s3_z[0] ? acc - s3_p : acc + s3_p;
                    OP_ADD: acc <= s3_z[0] ? acc - ($signed({{(AW-RW){s3_a[RW-1]}}, s3_a}) <<< s3_y[5:0])
                                           : acc + ($signed({{(AW-RW){s3_a[RW-1]}}, s3_a}) <<< s3_y[5:0]);
                    OP_ST: if (!fits) overflow <= 1'b1;
                    OP_DIV: begin
                        if (!dv_busy && !dv_done) begin
                            dv_t <= tsel;
                            dv_num <= (ONE_D << (2 * F)) + $unsigned(tsel >>> 1);
                            dv_rem <= 0; dv_q <= 0; dv_i <= 0; dv_busy <= 1'b1;
                        end else if (dv_busy) begin
                            // restoring division, one quotient bit per clock
                            if (dv_sh >= {2'b00, dv_t}) begin
                                dv_rem <= dv_sh - {2'b00, dv_t};
                                dv_q <= {dv_q[DB-2:0], 1'b1};
                            end else begin
                                dv_rem <= dv_sh;
                                dv_q <= {dv_q[DB-2:0], 1'b0};
                            end
                            dv_num <= {dv_num[DB-2:0], 1'b0};
                            dv_i <= dv_i + 1'b1;
                            if (dv_i == DB - 1) begin dv_busy <= 1'b0; dv_done <= 1'b1; end
                        end else begin  // dv_done: the write port stores the quotient this clock
                            dv_done <= 1'b0;
                        end
                    end
                    OP_END: begin busy <= 1'b0; run <= 1'b0; s1_v <= 1'b0; s2_v <= 1'b0; s3_v <= 1'b0; end
                    default: ;
                endcase
            end
        end
    end
endmodule
