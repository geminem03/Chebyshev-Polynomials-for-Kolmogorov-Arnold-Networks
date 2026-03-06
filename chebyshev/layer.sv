`timescale 1ns / 1ps

module layer #(
    parameter integer NUM_PES = 64,      
    parameter integer WIDTH = 16,        
    parameter integer DEGREE = 3,       
    parameter integer ACC_WIDTH = 22     
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic              start, 
    
    // Single 16-bit input port. Streams data over 64 cycles.
    input  logic signed [WIDTH-1:0] x_in,
    
    output logic              ready,     
    output logic              debug_bit 
);

    typedef enum logic [2:0] {IDLE, WAIT_FIRST_QUAD, FETCH, COMPUTE, ACCUM_1, ACCUM_2, FINISH} state_t;
    state_t state;
    
    logic signed [WIDTH-1:0] input_buffer [0:NUM_PES-1];
    logic [5:0] load_idx; 
    logic loading;
    
    logic [3:0] group_idx; 
    logic pe_start;
    logic [NUM_PES-1:0] pe_dones;
    
    logic signed [WIDTH-1:0] curr_x_A, curr_x_B, curr_x_C, curr_x_D;
    assign curr_x_A = input_buffer[{group_idx, 2'b00}]; 
    assign curr_x_B = input_buffer[{group_idx, 2'b01}];
    assign curr_x_C = input_buffer[{group_idx, 2'b10}];
    assign curr_x_D = input_buffer[{group_idx, 2'b11}];

    logic signed [WIDTH-1:0] pe_coeff_in [0:NUM_PES-1];
    logic [1:0] curr_thread [0:NUM_PES-1];
    logic [7:0] curr_k [0:NUM_PES-1];

    logic [7:0] mem_addr;
    assign mem_addr = {group_idx, 4'd0} + {6'd0, curr_thread[0], 2'd0} + (8'd3 - curr_k[0]);

    logic signed [ACC_WIDTH-1:0] accumulator [0:NUM_PES-1];
    logic signed [WIDTH-1:0] res_A [0:NUM_PES-1], res_B [0:NUM_PES-1], 
                             res_C [0:NUM_PES-1], res_D [0:NUM_PES-1];

    logic signed [ACC_WIDTH-1:0] sum_AB [0:NUM_PES-1];
    logic signed [ACC_WIDTH-1:0] sum_CD [0:NUM_PES-1];

    genvar i;
    generate
        for (i = 0; i < NUM_PES; i++) begin : pes
            localparam [7:0] D0 = 8'd48 + (i % 10);
            localparam [7:0] D1 = 8'd48 + ((i / 10) % 10);

            altsyncram #(
                .operation_mode("SINGLE_PORT"),
                .width_a(16),
                .widthad_a(8),
                .numwords_a(256),
                .ram_block_type("M20K"),
                .outdata_reg_a("CLOCK0"), 
                .init_file({"mem_init/weights_pe_", D1, D0, ".mif"})
            ) weight_ram (
                .clock0(clk),
                .address_a(mem_addr),
                .q_a(pe_coeff_in[i]),
                .wren_a(1'b0)
            );

            pe_quad #(.WIDTH(WIDTH), .DEGREE(DEGREE)) core (
                .clk(clk), .rst_n(rst_n), .start(pe_start),
                .x_A(curr_x_A), .x_B(curr_x_B), .x_C(curr_x_C), .x_D(curr_x_D),
                .coeff_in(pe_coeff_in[i]),
                .curr_thread(curr_thread[i]),
                .curr_k(curr_k[i]),
                .done(pe_dones[i]),
                .y_A(res_A[i]), .y_B(res_B[i]), .y_C(res_C[i]), .y_D(res_D[i])
            );
        end
    endgenerate

    // --- INDEPENDENT BACKGROUND LOADER ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            loading <= 0; load_idx <= 0;
            for(int p=0; p<NUM_PES; p++) input_buffer[p] <= 0;
        end else begin
            if (start) begin
                loading <= 1; load_idx <= 1;
                input_buffer[0] <= x_in;
            end else if (loading) begin
                input_buffer[load_idx] <= x_in;
                if (load_idx == 63) loading <= 0;
                else load_idx <= load_idx + 1;
            end
        end
    end

    // --- CONTROLLER FSM ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            group_idx <= 0; ready <= 0; pe_start <= 0;
            for (int p=0; p<NUM_PES; p++) begin 
                accumulator[p] <= 0;
                sum_AB[p] <= 0; sum_CD[p] <= 0;
            end
        end else begin
            case (state)
                IDLE: begin
                    ready <= 0;
                    if (start) begin
                        state <= WAIT_FIRST_QUAD;
                        group_idx <= 0;
                        for (int p=0; p<NUM_PES; p++) begin
                            accumulator[p] <= 0;
                            sum_AB[p] <= 0; sum_CD[p] <= 0;
                        end
                    end
                end
                WAIT_FIRST_QUAD: begin
                    // Proceed the moment the first 4 inputs are safely loaded
                    if (load_idx >= 4 || !loading) begin
                        state <= FETCH;
                    end
                end
                FETCH: begin
                    pe_start <= 1;
                    state <= COMPUTE;
                end
                COMPUTE: begin
                    pe_start <= 0;
                    if (pe_dones[0]) begin
                         state <= ACCUM_1; 
                    end
                end
                ACCUM_1: begin 
                     for (int p=0; p<NUM_PES; p++) begin 
                         sum_AB[p] <= res_A[p] + res_B[p];
                         sum_CD[p] <= res_C[p] + res_D[p];
                     end
                     state <= ACCUM_2;
                end
                ACCUM_2: begin 
                     for (int p=0; p<NUM_PES; p++) begin
                         accumulator[p] <= accumulator[p] + sum_AB[p] + sum_CD[p];
                     end
                     
                     if (group_idx == 15) begin 
                         state <= FINISH;
                     end else begin
                         group_idx <= group_idx + 1;
                         state <= FETCH;
                     end
                end
                FINISH: begin
                    ready <= 1; state <= IDLE;
                end
            endcase
        end
    end

    // --- I/O ---
    (* keep *) logic [NUM_PES*ACC_WIDTH-1:0] pe_accum_results;
    
    genvar j;
    generate
        for (j = 0; j < NUM_PES; j++) begin : pack_results
            assign pe_accum_results[j*ACC_WIDTH +: ACC_WIDTH] = accumulator[j];
        end
    endgenerate

    assign debug_bit = ^pe_accum_results; 

endmodule