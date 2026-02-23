`timescale 1ns / 1ps

module ram_bank_64 #(
    parameter integer BANK_ID = 0,
    parameter integer WIDTH   = 64,
    parameter integer DEPTH   = 64
)(
    input  logic             clk,
    input  logic [5:0]       addr, // 6 bits for 0..63
    output logic [WIDTH-1:0] data
);

    (* ramstyle = "M20K" *) logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        case (BANK_ID)
            0: $readmemh("coeffs/bank_0.mem", mem);
            1: $readmemh("coeffs/bank_1.mem", mem);
            2: $readmemh("coeffs/bank_2.mem", mem);
            3: $readmemh("coeffs/bank_3.mem", mem);
            4: $readmemh("coeffs/bank_4.mem", mem);
            5: $readmemh("coeffs/bank_5.mem", mem);
            6: $readmemh("coeffs/bank_6.mem", mem);
            7: $readmemh("coeffs/bank_7.mem", mem);
            8: $readmemh("coeffs/bank_8.mem", mem);
            9: $readmemh("coeffs/bank_9.mem", mem);
            10: $readmemh("coeffs/bank_10.mem", mem);
            11: $readmemh("coeffs/bank_11.mem", mem);
            12: $readmemh("coeffs/bank_12.mem", mem);
            13: $readmemh("coeffs/bank_13.mem", mem);
            14: $readmemh("coeffs/bank_14.mem", mem);
            15: $readmemh("coeffs/bank_15.mem", mem);
            16: $readmemh("coeffs/bank_16.mem", mem);
            17: $readmemh("coeffs/bank_17.mem", mem);
            18: $readmemh("coeffs/bank_18.mem", mem);
            19: $readmemh("coeffs/bank_19.mem", mem);
            20: $readmemh("coeffs/bank_20.mem", mem);
            21: $readmemh("coeffs/bank_21.mem", mem);
            22: $readmemh("coeffs/bank_22.mem", mem);
            23: $readmemh("coeffs/bank_23.mem", mem);
            24: $readmemh("coeffs/bank_24.mem", mem);
            25: $readmemh("coeffs/bank_25.mem", mem);
            26: $readmemh("coeffs/bank_26.mem", mem);
            27: $readmemh("coeffs/bank_27.mem", mem);
            28: $readmemh("coeffs/bank_28.mem", mem);
            29: $readmemh("coeffs/bank_29.mem", mem);
            30: $readmemh("coeffs/bank_30.mem", mem);
            31: $readmemh("coeffs/bank_31.mem", mem);
            32: $readmemh("coeffs/bank_32.mem", mem);
            33: $readmemh("coeffs/bank_33.mem", mem);
            34: $readmemh("coeffs/bank_34.mem", mem);
            35: $readmemh("coeffs/bank_35.mem", mem);
            36: $readmemh("coeffs/bank_36.mem", mem);
            37: $readmemh("coeffs/bank_37.mem", mem);
            38: $readmemh("coeffs/bank_38.mem", mem);
            39: $readmemh("coeffs/bank_39.mem", mem);
            40: $readmemh("coeffs/bank_40.mem", mem);
            41: $readmemh("coeffs/bank_41.mem", mem);
            42: $readmemh("coeffs/bank_42.mem", mem);
            43: $readmemh("coeffs/bank_43.mem", mem);
            44: $readmemh("coeffs/bank_44.mem", mem);
            45: $readmemh("coeffs/bank_45.mem", mem);
            46: $readmemh("coeffs/bank_46.mem", mem);
            47: $readmemh("coeffs/bank_47.mem", mem);
            48: $readmemh("coeffs/bank_48.mem", mem);
            49: $readmemh("coeffs/bank_49.mem", mem);
            50: $readmemh("coeffs/bank_50.mem", mem);
            51: $readmemh("coeffs/bank_51.mem", mem);
            52: $readmemh("coeffs/bank_52.mem", mem);
            53: $readmemh("coeffs/bank_53.mem", mem);
            54: $readmemh("coeffs/bank_54.mem", mem);
            55: $readmemh("coeffs/bank_55.mem", mem);
            56: $readmemh("coeffs/bank_56.mem", mem);
            57: $readmemh("coeffs/bank_57.mem", mem);
            58: $readmemh("coeffs/bank_58.mem", mem);
            59: $readmemh("coeffs/bank_59.mem", mem);
            60: $readmemh("coeffs/bank_60.mem", mem);
            61: $readmemh("coeffs/bank_61.mem", mem);
            62: $readmemh("coeffs/bank_62.mem", mem);
            63: $readmemh("coeffs/bank_63.mem", mem);
            default: $readmemh("coeffs/bank_0.mem", mem);
        endcase
    end

    always_ff @(posedge clk) begin
        data <= mem[addr];
    end
endmodule
