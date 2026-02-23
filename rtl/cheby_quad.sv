`timescale 1ns / 1ps

module cheby_quad #(
    parameter integer WIDTH = 16,
    parameter integer FRAC_BITS = 10,
    parameter integer DEGREE = 3          
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             start, 
    
    // Four Inputs (A, B, C, D)
    input  logic signed [WIDTH-1:0] x_A, x_B, x_C, x_D,
    input  logic signed [WIDTH-1:0] coeffs [0:DEGREE], 
    
    output logic             done, // Goes high when ALL 4 are done
    output logic signed [WIDTH-1:0] y_A, y_B, y_C, y_D
);

    // --- Control Logic ---
    // Counter 0..3 to rotate threads
    logic [1:0] thread_idx;
    logic [7:0] k; 
    logic active;

    // --- Input Capture ---
    // Register coefficients locally (Keep this optimization!)
    logic signed [WIDTH-1:0] coeffs_reg [0:DEGREE];
    always_ff @(posedge clk) coeffs_reg <= coeffs;

    // --- Thread State ---
    // We need state for 4 independent calculations
    logic signed [WIDTH-1:0] b_next [0:3];
    logic signed [WIDTH-1:0] b_prev [0:3];

    // --- STAGE 1: MULTIPLIER INPUT MUX ---
    logic signed [WIDTH-1:0] current_x;
    logic signed [WIDTH-1:0] current_b;
    
    always_comb begin
        case (thread_idx)
            0: begin current_x = x_A; current_b = b_next[0]; end
            1: begin current_x = x_B; current_b = b_next[1]; end
            2: begin current_x = x_C; current_b = b_next[2]; end
            3: begin current_x = x_D; current_b = b_next[3]; end
        endcase
    end

    // --- STAGE 2: PIPELINED MULTIPLIER ---
    logic signed [2*WIDTH-1:0] raw_product;
    assign raw_product = current_x * current_b;

    // Pipeline Reg 1 (Output of Mult)
    logic signed [2*WIDTH-1:0] mult_reg;
    logic [1:0] thread_idx_d1; // Track which thread this result belongs to
    
    always_ff @(posedge clk) begin
        mult_reg <= raw_product;
        thread_idx_d1 <= thread_idx;
    end

    // --- STAGE 3: PIPELINED ADDER ---
    // We calculate the term and add the coefficient HERE, then register it.
    // This breaks the "Adder Chain" bottleneck.
    logic signed [WIDTH-1:0] term;
    logic signed [2*WIDTH-1:0] scaled_product;
    logic signed [WIDTH-1:0] adder_result;
    
    always_comb begin
        // Note: k is stable for 4 cycles, so we use it directly.
        scaled_product = (k == 0) ? mult_reg : (mult_reg <<< 1);
        term = WIDTH'((scaled_product + (1 <<< (FRAC_BITS - 1))) >>> FRAC_BITS);
        
        // Pre-calculate the addition part: Coeff + Term
        adder_result = coeffs_reg[k] + term;
    end

    // Pipeline Reg 2 (Output of Adder)
    logic signed [WIDTH-1:0] add_reg;
    logic [1:0] thread_idx_d2;
    
    always_ff @(posedge clk) begin
        add_reg <= adder_result;
        thread_idx_d2 <= thread_idx_d1;
    end

    // --- STAGE 4: WRITEBACK (SUBTRACTOR) ---
    // Final operation: Result - b_prev
    // This is very fast (just one subtract).
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 0;
            thread_idx <= 0;
            k <= DEGREE;
            done <= 0;
            for(int i=0; i<4; i++) begin b_next[i]<=0; b_prev[i]<=0; end
        end else begin
            done <= 0;
            
            if (start) begin
                active <= 1;
                thread_idx <= 0;
                k <= DEGREE;
                for(int i=0; i<4; i++) begin b_next[i]<=0; b_prev[i]<=0; end
            end else if (active) begin
                // 1. Rotate Thread Input
                thread_idx <= thread_idx + 1;

                // 2. Decrement k only after a full rotation (every 4 cycles)
                if (thread_idx == 3) begin
                    if (k == 0) begin
                        active <= 0;
                        done <= 1; 
                    end else begin
                        k <= k - 1;
                    end
                end

                // 3. Update State (Using data from Pipeline Stage 2)
                // We are updating the thread indicated by 'thread_idx_d2'
                // This happens 2 cycles after the fetch.
                if (active) begin 
                   // Logic: b_new = (Coeff + Term) - b_old
                   // 'add_reg' holds (Coeff + Term)
                   // 'b_prev' needs to be updated.
                   
                   // Update history
                   b_prev[thread_idx_d2] <= b_next[thread_idx_d2];
                   
                   // Update next
                   b_next[thread_idx_d2] <= add_reg - b_prev[thread_idx_d2];

                   // Capture Output if we are at the end
                   if (k == 0) begin 
                       case (thread_idx_d2)
                           0: y_A <= add_reg - b_prev[0];
                           1: y_B <= add_reg - b_prev[1];
                           2: y_C <= add_reg - b_prev[2];
                           3: y_D <= add_reg - b_prev[3];
                       endcase
                   end
                end
            end
        end
    end

endmodule