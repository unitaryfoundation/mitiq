---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What happens when I use VD?

When you call {func}`mitiq.vd.vd.execute_with_vd`, the following steps occur automatically in the backend.

1. Two identical noisy copies of the quantum state of interest are prepared (e.g. $C\otimes C |0^{2n}\rangle$ where $C$ is the circuit of interest)
2. The two states are then entangled using the $B_i$ gate defined below. The correlations created by the entanglement allow the post-processing algorithm to extract information about the "virtually distilled" pure state $\frac{\rho^2}{\text{Tr}(\rho^2)}$, which has suppressed contributions from non-dominant eigenvectors compared to the original noisy state $\rho$.

The entire process requires only unitary operations and computational basis measurements, making it practical for near-term quantum devices while providing exponential error suppression in the ideal case.

## Step-by-Step Workflow

### 1. Circuit Duplication

The original circuit acting on $N$ qubits is duplicated to create two parallel copies, resulting in a circuit on $2N$ qubits.

### 2. Diagonalizing Gate Application

A diagonalizing gate $B_i^{(2)}$ is applied between corresponding qubits in the two copies:

$$
B_i^{(2)} =
\begin{bmatrix}
1 & 0                  & 0                   & 0 \\
0 & \frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} & 0 \\
0 & \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} & 0 \\
0 & 0                  & 0                   & 1
\end{bmatrix}
$$

This gate is applied to qubit pairs $(i, i + N)$ for $i = 0, 1, \ldots, N - 1$.

### 3. Measurement

All qubits in both copies are measured in the computational basis, yielding bitstrings $z^1$ and $z^2$ for the first and second copies respectively.

### 4. Post-Processing

The measurement results are processed to compute the error-mitigated expectation values using:

**Numerator for qubit $i$:**

$$
E_i = \frac{1}{2^N} \sum_{\text{shots}} \left( (-1)^{z^1_i} + (-1)^{z^2_i} \right) \prod_{j \neq i} \left( 1 + (-1)^{z^1_j} - (-1)^{z^2_j} + (-1)^{z^1_j} (-1)^{z^2_j} \right)
$$

**Denominator:**

$$
D = \frac{1}{2^N} \sum_{\text{shots}} \prod_{j=0}^{N-1} \left( 1 + (-1)^{z^1_j} - (-1)^{z^2_j} + (-1)^{z^1_j} (-1)^{z^2_j} \right)
$$

**Final Result:**

$$
\langle Z_i \rangle_{\text{corrected}} = \frac{E_i}{D}
$$

### 5. Return Values

The function returns a list of length $N$ containing the error-mitigated expectation values $\langle Z_i \rangle_{\text{corrected}}$ for each qubit in the original circuit.

## Implementation Details

- The circuit construction uses `_copy_circuit_parallel()` to create parallel copies
- The diagonalizing gate is applied via `_apply_diagonalizing_gate()` 
- Measurement results are processed by `combine_results()` which implements the mathematical formulas above
- The implementation handles both `LineQubit` and `GridQubit` qubit types, but has only been tested on contiguous groups of qubits.
 