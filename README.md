# Chebyshev Polynomials for Kolmogorov-Arnold Networks

## The Chebyshev Approach

Traditional Kolmogorov-Arnold Networks (KANs) utilize B-splines on their edges to learn non-linear relationships. While effective in software, B-splines are computationally expensive for FPGAs because they require complex piecewise logic and frequent memory lookups.

This project replaces B-splines with **Chebyshev Polynomials**. By using the Chebyshev recurrence relation:

Tₙ(x) = 2xTₙ₋₁(x) − Tₙ₋₂(x)

we can evaluate non-linearities using standard Multiply-Accumulate (MAC) operations. This allows for a hardware architecture that is significantly faster and more area-efficient than traditional spline-based implementations.

---

## Hardware Architecture

The system is designed to accelerate a **64×64 fully connected KAN layer** with **Degree 3 polynomials**.

---

## Key Innovations

### Staggered Quad-Core Pipeline

To solve the bottleneck caused by recursive data dependencies, each Processing Element (PE) processes **4 independent polynomial inputs (A, B, C, D)** simultaneously.

This staggers the inputs to ensure **100% utilization of DSP blocks**, eliminating idle cycles while waiting for recursive results.

### Massive Parallelism

The design instantiates **64 individual `cheby_quad` engines** to compute **256 polynomial edges in parallel**.

### Distributed Memory Banks

Weight data is stored across **64 independent M20K RAM blocks**.

Each bank provides a private silo of coefficients to its dedicated PE, ensuring maximum data bandwidth.

### Precision & Requantization

The controller uses a **22-bit widened accumulator** to prevent overflow during the summation of 64 edges.

Final results are:
- Requantized  
- Saturated (clamped)  
- Converted back to **16-bit signed integers** for the next layer  

---

## File Structure

### 1. Hardware Logic (RTL)

- `cheby_quad.sv`  
  The core compute engine. Implements the 4-thread staggered pipeline for Degree 3 Chebyshev polynomials.

- `kan_64x64.sv`  
  The top-level orchestrator. Manages the 64-PE array, the state machine, and final output accumulation/clamping.

- `ram_bank_64.sv`  
  A parameterized RAM wrapper that loads coefficients from `.mem` files into M20K memory blocks.

---

### 2. Machine Learning

- `ml_benchmarks/`  
  Contains the Python/PyTorch scripts used to validate the model's accuracy and parameter efficiency against standard MLPs.
