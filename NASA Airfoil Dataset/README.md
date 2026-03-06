# KAN vs. MLP: Iso-Accuracy Benchmark Study

Welcome to this benchmark study evaluating the parameter efficiency and performance of Kolmogorov-Arnold Networks (KANs) against traditional Multi-Layer Perceptrons (MLPs). The goal of this repository is to demonstrate how different neural network architectures perform when tasked with reaching a strict **95% accuracy threshold** on complex scientific data.

## The NASA Dataset and KAN Suitability

This study utilizes a NASA dataset (e.g., C-MAPSS, Airfoil Self-Noise, etc.) which consists of complex, continuous sensor and telemetry readings. 

**Why Kolmogorov-Arnold Networks (KANs)?**
Standard MLPs use fixed activation functions (like ReLU or Sigmoid) at each node and learn a matrix of linear weights. KANs fundamentally flip this architecture: they eliminate fixed node activations and instead place learnable univariate functions directly on the edges of the network. 

NASA datasets are typically governed by underlying low-dimensional physical or thermodynamic laws. KANs are theoretically uniquely suited for this type of data because:
1. **Mathematical Representation:** Their edge-based learnable functions excel at mapping the continuous, non-linear physical relationships inherent in scientific data. 
2. **Parameter Efficiency:** By adapting the actual shape of the activation functions, KANs can solve complex regression and classification tasks using a fraction of the parameters that a standard MLP would require. 

## The Three KAN Architectures

To explore the best basis functions for this physical data, we benchmarked three distinct flavors of KANs:

* **B-Spline KAN:** Uses piecewise polynomials (B-splines) as its learnable edge functions. B-splines offer excellent local control and adaptability, meaning the network can fine-tune specific segments of the function without globally distorting the rest of the curve.
* **Chebyshev KAN:** Employs orthogonal Chebyshev polynomials. These are exceptionally good at global approximation and are mathematically robust against Runge's phenomenon (oscillation at the edges of an interval), making them highly stable for smooth, continuous data.
* **Fourier KAN:** Uses sine and cosine harmonics. This architecture is ideal if the underlying physical telemetry exhibits strong periodic or oscillatory behavior.

## Methodology: An Apples-to-Apples Comparison

To ensure a fair comparison between the KANs and the MLP, we conducted an **Iso-Accuracy Study**. 

Instead of arbitrarily matching parameter counts (which disadvantages KANs) or matching depths (which ignores the complexity of KAN edges), we set a strict target: **95% testing accuracy**. We then ran extensive hyperparameter searches and generated Pareto fronts to identify the absolute minimum number of parameters required for each architecture to cross that 95% threshold. 

[cite_start]Because standard MLPs struggled to reach 95% with shallow networks [cite: 7][cite_start], an extended hyperparameter sweep was explicitly required for the MLP to find a configuration that met the target[cite: 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]. 

## Iso-Accuracy Results

To thoroughly benchmark parameter efficiency, we established two strict testing accuracy targets: **93.5%** and **95.0%**. We then analyzed the Pareto fronts for each architecture to identify the absolute minimum number of parameters required to cross these thresholds.

Because standard MLPs struggled to reach these high-accuracy marks with shallow or small networks, an extended hyperparameter sweep was explicitly required to find an MLP configuration that could successfully hit the targets.

### Target 1: $\ge$ 93.5% Accuracy

For a 93.5% target, the KAN architectures already demonstrate significant efficiency over the standard MLP. 

| Model | Architecture (FC Dims) | Minimum Params | Actual Accuracy |
| :--- | :--- | :--- | :--- |
| **Chebyshev KAN** | `[5, 12, 1]` | **288** | 93.72% |
| **Fourier KAN** | `[5, 20, 1]` | **741** | 94.43% |
| **B-Spline KAN** | `[[5, 0], [9, 0], [1, 0]]` | **922** | 94.33% |
| **Standard MLP** | `[5, 217, 208, 1]` | **46,855** | 93.59% |

*At this threshold, the Chebyshev KAN is over 160x more parameter-efficient than the standard MLP.*

### Target 2: $\ge$ 95.0% Accuracy

Pushing the models to a more stringent 95% threshold drastically widens the gap. While the KANs required only modest increases in parameter count to reach 95%, the standard MLP experienced a massive parameter explosion.

| Model | Architecture (FC Dims) | Minimum Params | Actual Accuracy |
| :--- | :--- | :--- | :--- |
| **Chebyshev KAN** | `[5, 17, 1]` | **408** | 95.13% |
| **Fourier KAN** | `[5, 28, 1]` | **1,037** | 95.46% |
| **B-Spline KAN** | `[[5, 0], [12, 0], [1, 0]]` | **1,213** | 95.26% |
| **Standard MLP** | `[5, 961, 206, 304, 1]` | **267,171** | 95.31% |

### Conclusion
The results highlight the massive parameter efficiency of Kolmogorov-Arnold Networks for modeling complex physical datasets. As the required accuracy increases from 93.5% to 95.0%, the standard MLP scales extremely poorly, requiring hundreds of thousands of additional parameters. Conversely, the Chebyshev KAN crossed the final 95% target using just 408 parameters—making it roughly **650x more parameter-efficient** than the best-performing Multi-Layer Perceptron.