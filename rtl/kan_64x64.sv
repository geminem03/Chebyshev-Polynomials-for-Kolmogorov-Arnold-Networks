`timescale 1ns / 1ps

module kan_64x64 #(
    parameter integer WIDTH = 16,
    parameter integer DEGREE = 3
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             start,
    input  logic signed [WIDTH-1:0] x_in, // Broadcast input for benchmark

    output logic             done,
    output logic signed [WIDTH-1:0] y_out [0:63]
);
    // --- State Machine ---
    typedef enum logic [1:0] {IDLE, FETCH, COMPUTE, FINISH} state_t;
    state_t state;

    // Widened to 8 bits to access full 256-depth memory for 64x64 matrix
    logic [7:0] quad_idx;

    logic       pe_start;
    logic [63:0] pe_dones;

    // --- Data Wires ---
    logic [63:0]             bank_data_out [0:63];
    logic signed [WIDTH-1:0] pe_coeffs     [0:63][0:DEGREE];
    
    // --- NEW: Widen accumulator to 22 bits to prevent overflow ---
    logic signed [WIDTH+5:0] accumulator   [0:63];

    logic signed [WIDTH-1:0] res_A [0:63], res_B [0:63], res_C [0:63], res_D [0:63];

    // --- 1. MEMORY BANKS (64x) ---
    genvar i;
    generate
        for (i = 0; i < 64; i++) begin : banks
            ram_bank_64 #(
                .BANK_ID(i), .WIDTH(64), .DEPTH(256) // Expanded to 32KB full matrix capacity
            ) mem_inst (
                .clk(clk),
                .addr(quad_idx), // Using full 8-bit address
                .data(bank_data_out[i])
            );

            assign pe_coeffs[i][0] = bank_data_out[i][15:0];
            assign pe_coeffs[i][1] = bank_data_out[i][31:16];
            assign pe_coeffs[i][2] = bank_data_out[i][47:32];
            assign pe_coeffs[i][3] = bank_data_out[i][63:48];
        end
    endgenerate

    // --- 2. COMPUTE ENGINES (64x) ---
    generate
        for (i = 0; i < 64; i++) begin : pes
            cheby_quad #(
                .WIDTH(WIDTH), .DEGREE(DEGREE)
            ) core (
                .clk(clk), .rst_n(rst_n), .start(pe_start),
  
                .x_A(x_in), .x_B(x_in), .x_C(x_in), .x_D(x_in), // Broadcast Input
                .coeffs(pe_coeffs[i]),
                .done(pe_dones[i]), 
                .y_A(res_A[i]), .y_B(res_B[i]), .y_C(res_C[i]), .y_D(res_D[i])
            );
        end
    endgenerate

    // --- 3. CONTROLLER ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            quad_idx <= 0;
            done <= 0;
            pe_start <= 0;
            for (int k=0; k<64; k++) accumulator[k] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= FETCH;
                        quad_idx <= 0;
                        // Reset accumulators at the start of a new calculation
                        for (int k=0; k<64; k++) accumulator[k] <= 0;
                    end
                end

                FETCH: begin
                    pe_start <= 1;
                    state <= COMPUTE;
                end

                COMPUTE: begin
                    pe_start <= 0;
                    if (pe_dones[0]) begin 
                         for (int k=0; k<64; k++) begin
                             accumulator[k] <= accumulator[k] + res_A[k] + res_B[k] + res_C[k] + res_D[k];
                         end
                         
                         // Loop 64 times to process all 256 memory addresses (64 quads)
                         if (quad_idx == 63) begin 
                             state <= FINISH;
                         end else begin
                             quad_idx <= quad_idx + 1;
                             state <= FETCH; 
                         end
                    end
                end

                FINISH: begin
                    // --- NEW: Output Requantization and Saturation ---
                    for (int k=0; k<64; k++) begin
                        // Clamp to Maximum Positive (16-bit: +32767)
                        if (accumulator[k] > 22'sd32767) begin
                            y_out[k] <= 16'sd32767;
                        end
                        // Clamp to Minimum Negative (16-bit: -32768)
                        else if (accumulator[k] < -22'sd32768) begin
                            y_out[k] <= -16'sd32768;
                        end
                        // Safe to cast back to 16-bit
                        else begin
                            y_out[k] <= accumulator[k][15:0];
                        end
                    end
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule