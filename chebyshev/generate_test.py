import os
import math
import numpy as np

# configuration
mif_dir = r'C:\Skule_Hardrive\kan-tpu\chebyshev\mem_init'
test_dir = r'C:\Skule_Hardrive\kan-tpu\chebyshev\testcase_data'
NUM_PES = 64
NUM_INPUTS = 64
FRAC_BITS = 10

def to_s16(val):
    """Convert float to S16.10 fixed point integer."""
    v = int(round(val * (1 << FRAC_BITS)))
    if v > 32767: v = 32767
    elif v < -32768: v = -32768
    return v

def to_hex_16(val):
    return f"{val & 0xFFFF:04x}"

def wrap_s16(val):
    """Mimics SystemVerilog 16-bit signed overflow/truncation."""
    val = val & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val

def wrap_s22_to_s64(val):
    """Sign-extends the 22-bit hardware accumulator up to 64-bit for the testbench."""
    val = val & 0x3FFFFF
    if val >= 0x200000:
        val -= 0x400000
    return val

def hw_accurate_clenshaw(x_int, c0, c1, c2, c3):
    """Simulates EXACT pipelined fixed-point math with 16-bit wrapping"""
    coeffs = {3: c3, 2: c2, 1: c1, 0: c0}
    b_next = 0
    b_prev = 0
    
    for k in range(3, -1, -1):
        raw_prod = x_int * b_next
        scaled = raw_prod if k == 0 else (raw_prod << 1)
        term = wrap_s16((scaled + (1 << (FRAC_BITS - 1))) >> FRAC_BITS)
        add_reg = wrap_s16(coeffs[k] + term)
        
        old_b_prev = b_prev
        b_prev = b_next
        b_next = wrap_s16(add_reg - old_b_prev)
        
        if k == 0:
            return wrap_s16(add_reg - old_b_prev)

if __name__ == "__main__":
    os.makedirs(mif_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Generating Hamiltonian Phase-Space Inputs...")
    inputs_float = [0.8 * math.cos(2 * math.pi * i / NUM_INPUTS) for i in range(NUM_INPUTS)]
    inputs_s16 = [to_s16(x) for x in inputs_float]
    
    # save as binary NumPy array (Int16)
    np.save(os.path.join(test_dir, 'inputs_s16.npy'), np.array(inputs_s16, dtype=np.int16))

    print("Generating PE Weights and calculating Hardware Reference...")
    expected_outputs = []

    for pe in range(NUM_PES):
        k_stiffness = 0.5 + (pe / 128.0) 
        
        c0_int = to_s16(0.25 * k_stiffness)
        c1_int = to_s16(0.0)
        c2_int = to_s16(0.25 * k_stiffness)
        c3_int = to_s16(0.0)

        mif_filepath = os.path.join(mif_dir, f'weights_pe_{pe:02d}.mif')
        pe_accumulator = 0
        
        with open(mif_filepath, 'w') as f:
            f.write("WIDTH=16;\nDEPTH=256;\nADDRESS_RADIX=UNS;\nDATA_RADIX=HEX;\nCONTENT BEGIN\n")
            
            addr = 0
            for i in range(NUM_INPUTS):
                # written in descending order to match hardware TDM logic (3 - curr_k)
                f.write(f"    {addr}   : {to_hex_16(c3_int)};\n")
                f.write(f"    {addr+1} : {to_hex_16(c2_int)};\n")
                f.write(f"    {addr+2} : {to_hex_16(c1_int)};\n")
                f.write(f"    {addr+3} : {to_hex_16(c0_int)};\n")
                addr += 4
                
                y = hw_accurate_clenshaw(inputs_s16[i], c0_int, c1_int, c2_int, c3_int)
                pe_accumulator = wrap_s22_to_s64(pe_accumulator + y)
                
            f.write("END;\n")
            
        expected_outputs.append(pe_accumulator)

    # save as binary NumPy array (Int64)
    np.save(os.path.join(test_dir, 'expected_s64.npy'), np.array(expected_outputs, dtype=np.int64))
    print("Test generation complete. MIF files and test data saved.")