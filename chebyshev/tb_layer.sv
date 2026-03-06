`timescale 1ns/1ps
`default_nettype none

module tb_layer;

    localparam NUM_PES = 64;
    localparam MEM_DEPTH = 256; // TDM uses 256 depth for 64 inputs * 4 coeffs

    localparam string TESTCASE_DIR = "testcase_data";
    localparam string INPUT_NPY = {TESTCASE_DIR, "/inputs_s16.npy"};
    localparam string GOLDEN_NPY = {TESTCASE_DIR, "/expected_s64.npy"};

    // -------------------------------------------------------------------------
    // 1. Signals & DUT
    // -------------------------------------------------------------------------
    logic clk;
    logic rst_n;
    logic start;
    logic [15:0] x_in;

    wire ready;
    wire debug_bit;

    // Access internal accumulator results for verification
    wire [NUM_PES*22-1:0] pe_accum_results; // Chebyshev ACC_WIDTH is 22

    // --- PERFORMANCE TRACKING ---
    integer cycle_count = 0;
    integer start_cycle = 0;
    integer end_cycle   = 0;
    integer total_cycles = 0;
    
    // 234.85 MHz clock period in nanoseconds
    real clock_period_ns = 4.258037; 
    real latency_ns;
    real latency_us;

    layer #(
        .NUM_PES(NUM_PES),
        .WIDTH(16),
        .DEGREE(3),
        .ACC_WIDTH(22)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .x_in(x_in),
        .ready(ready),
        .debug_bit(debug_bit)
    );

    assign pe_accum_results = dut.pe_accum_results;

    // Clock Generation (100 MHz Simulation Clock)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Global cycle counter
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
    end

    // -------------------------------------------------------------------------
    // 2. NumPy Test Vectors 
    // -------------------------------------------------------------------------
    shortint       inputs[$];
    longint signed expected_outputs[$];

    function automatic int read_byte_or_fatal(input int fd, input string what);
        int c = $fgetc(fd);
        if (c == -1) $fatal(1, "Unexpected EOF while reading %s", what);
        return c;
    endfunction

    function automatic int parse_shape_1d(input string header);
        int i, len, value;
        bit seen_paren, collecting;
        byte unsigned ch;

        len = header.len();
        value = 0; seen_paren = 0; collecting = 0;

        for (i = 0; i < len; i++) begin
            ch = header[i];
            if (!seen_paren) begin
                if (ch == "(") seen_paren = 1;
            end else if (!collecting) begin
                if (ch >= "0" && ch <= "9") begin
                    collecting = 1; value = ch - "0";
                end else if (ch == ")") break;
            end else begin
                if (ch >= "0" && ch <= "9") value = (value * 10) + (ch - "0");
                else if (ch == "," || ch == ")") return value;
            end
        end
        return collecting ? value : -1;
    endfunction

    task automatic load_npy_s16_1d(input string npy_path, ref shortint dst[$]);
        int fd, i, major, minor, header_len, elem_count, c;
        string header; byte unsigned b0, b1;
        dst.delete();
        fd = $fopen(npy_path, "rb");
        if (fd == 0) $fatal(1, "Could not open NPY file: %s", npy_path);

        // Skip Magic & Version checks for brevity in reading bytes
        repeat(6) c = read_byte_or_fatal(fd, "magic");
        major = read_byte_or_fatal(fd, "major");
        minor = read_byte_or_fatal(fd, "minor");

        b0 = read_byte_or_fatal(fd, "hlen0"); b1 = read_byte_or_fatal(fd, "hlen1");
        header_len = int'(b0) + (int'(b1) << 8);

        header = "";
        for (i = 0; i < header_len; i++) header = {header, byte'(read_byte_or_fatal(fd, "hdr"))};

        elem_count = parse_shape_1d(header);
        for (i = 0; i < elem_count; i++) begin
            b0 = read_byte_or_fatal(fd, "d0"); b1 = read_byte_or_fatal(fd, "d1");
            dst.push_back(shortint'({b1, b0}));
        end
        void'($fclose(fd));
    endtask

    task automatic load_npy_s64_1d(input string npy_path, ref longint signed dst[$]);
        int fd, i, major, minor, header_len, elem_count, c;
        string header; byte unsigned b0, b1, b2, b3, b4, b5, b6, b7; logic [63:0] raw;
        dst.delete();
        fd = $fopen(npy_path, "rb");
        if (fd == 0) $fatal(1, "Could not open NPY file: %s", npy_path);

        repeat(6) c = read_byte_or_fatal(fd, "magic");
        major = read_byte_or_fatal(fd, "major");
        minor = read_byte_or_fatal(fd, "minor");

        b0 = read_byte_or_fatal(fd, "hlen0"); b1 = read_byte_or_fatal(fd, "hlen1");
        header_len = int'(b0) + (int'(b1) << 8);

        header = "";
        for (i = 0; i < header_len; i++) header = {header, byte'(read_byte_or_fatal(fd, "hdr"))};

        elem_count = parse_shape_1d(header);
        for (i = 0; i < elem_count; i++) begin
            b0 = read_byte_or_fatal(fd,"d0"); b1 = read_byte_or_fatal(fd,"d1"); 
            b2 = read_byte_or_fatal(fd,"d2"); b3 = read_byte_or_fatal(fd,"d3");
            b4 = read_byte_or_fatal(fd,"d4"); b5 = read_byte_or_fatal(fd,"d5"); 
            b6 = read_byte_or_fatal(fd,"d6"); b7 = read_byte_or_fatal(fd,"d7");
            raw = {b7, b6, b5, b4, b3, b2, b1, b0};
            dst.push_back(longint'(raw));
        end
        void'($fclose(fd));
    endtask

    // -------------------------------------------------------------------------
    // 3. Test Execution
    // -------------------------------------------------------------------------
    int errors = 0;
    longint signed actual_val;

    initial begin
        load_npy_s16_1d(INPUT_NPY, inputs);
        load_npy_s64_1d(GOLDEN_NPY, expected_outputs);

        // Initialize signals
        start = 0;
        x_in = 0;
        rst_n = 0;

        // Reset sequence
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        $display("=========================================================");
        $display(" Starting Chebyshev Layer Testbench");
        $display("=========================================================");

        // CHEBYSHEV STIMULUS: Stream all 64 inputs consecutively 
        @(posedge clk);
        start_cycle = cycle_count; // <--- CAPTURE START CYCLE
        
        start <= 1'b1;
        x_in  <= inputs[0];
        @(posedge clk);
        start <= 1'b0; // Pulse start for 1 cycle

        for (int k = 1; k < 64; k++) begin
            x_in <= inputs[k];
            @(posedge clk);
        end

        // Wait for the computation to complete
        wait(ready);
        end_cycle = cycle_count; // <--- CAPTURE END CYCLE
        
        // Calculate Metrics
        total_cycles = end_cycle - start_cycle;
        latency_ns   = total_cycles * clock_period_ns;
        latency_us   = latency_ns / 1000.0;
    

        for (int i = 0; i < NUM_PES; i++) begin
            // Extract the 22-bit result and sign-extend it to 64-bit for comparison
            actual_val = $signed(pe_accum_results[i*22 +: 22]);
            
            if (actual_val !== expected_outputs[i]) begin
                $display("FAIL PE[%0d] | Expected: %0d | Hardware: %0d", i, expected_outputs[i], actual_val);
                errors++;
            end else begin
                $display("PASS PE[%0d] | Value: %0d", i, actual_val);
            end
        end

        $display("=========================================================");
        if (errors == 0) begin
            $display(" SUCCESS! All %0d PE outputs match the Hamiltonian Reference.", NUM_PES);
            $display("---------------------------------------------------------");
            $display(" Total Cycles : %0d", total_cycles);
        end else begin
            $display(" FAILED WITH %0d ERRORS.", errors);
        end
        $display("=========================================================");

        $stop;
    end

endmodule