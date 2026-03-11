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

# What is the theory behind VD?

Virtual Distillation (VD) is a quantum error mitigation technique introduced and explained in much greater detail in the paper {cite}`Huggins_2021`.
This method is designed to suppress noise in NISQ devices by using multiple noisy copies of a quantum state $\rho^{\otimes M}$ to estimate expectation values with respect to a "virtually distilled" state, $\frac{\rho^M}{\text{tr}(\rho^M)}$.

The error-free observable is approximated as $\langle O\rangle_{\text{corrected}} := \frac{\text{tr}(O \rho^M)}{\text{tr}(\rho^M)}$.
By doing so, VD effectively reduces the influence of non-dominant eigenvectors in the density matrix and the estimator converges exponentially quickly towards the closest pure state to $\rho$ as $M$ is increased.
Crucially, VD does not require the explicit preparation of the purified state, making it computationally efficient for near-term quantum devices.

To evaluate the numerator and denominators of the $\langle O\rangle_{\text{corrected}}$, the equality below can be used:

$$\text{tr}(O \rho^M) = \text{tr}(O^i S^{(M)} \rho^{\otimes M})$$

$O^i$ represents the observable acting upon one of the M subsystems, and $S^{(M)}$ is the cyclic shift operator performend on M subsystems.

To better explain the exact mechanics of the VD protocol, a specific of example for M=2 is expanded upon below:

## When $M = 2$ with observable $Z$

### Obtain Multiple Copies of the Noisy State.

The method begins by preparing M independent noisy copies of the quantum state $\rho$, with $M = 2$ being a common starting point for balancing resource requirements with error suppression.
The copies are assumed to experience similar noise characteristics.

### Diagonalizing Gate

First a symmetrized version of the observable is defined as $O^{(M)} = \frac{1}{M} \sum_{i=0}^{M} O^i$.

For the example of $M = 2$ and $O = Z$, this equalities can be rewritten as: $Z^{(2)}_k = \frac{1}{2}(Z^1_k + Z^2_k)$.
This can be used to rewrite the corrected observable as:

$$\langle O\rangle_{\text{corrected}} = \frac{\text{tr}(Z^{(2)}_k S^{(2)} \rho^{\otimes 2})}{\text{tr}(S^{(2)} \rho^{\otimes 2})}$$

Both $S^{(2)}$ and $Z^{(2)}S^{(2)}$ need to be diagonalized, which can be done using the two-qubit unitary that acts on the $i$ th qubit defined as $B^{(2)}_i$.

$$
B_i^{(2)} =
\begin{bmatrix}
1 & 0                  & 0                   & 0 \\
0 & \frac{\sqrt{2}}{2} &  \frac{\sqrt{2}}{2} & 0 \\
0 & \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} & 0 \\
0 & 0                  & 0                   & 1
\end{bmatrix}
$$

Next, the $B^{(2)}_i$ gate is applied between each qubit $i$ in the first copy and the corresponding qubit $i$ in the second copy.

### Measure and Compute Expectation value

Perform a standard measurement on all qubits in both copies of the quantum state.
These measurement outcomes are stored as $z^1_i$ for the $i$th qubit in the first copy and $z^2_i$ for the $i$th qubit in the second copy.

The numerator value can now be calculated by computing for each qubit:

$$
E_i = \frac{1}{2N} \left( z^1_i + z^2_i \right) \prod_{j \neq i} \left( 1 + z^1_j - z^2_j + z^1_j z^2_j \right)
$$

The denomiator is calculated using the equation:

$$
D = \frac{1}{2N} \prod_{j = 1}^{N} \left(1 + z^1_j - z^2_j + z^1_j z^2_j \right)
$$

The final value of $\langle Z_i \rangle_\text{corrected}$ can thus be calculated as $\frac{E_i}{D}$.

VD has been shown to reduce errors by orders of magnitude in numerical simulations for a variety of quantum circuits and noise models.
Its simplicity, effectiveness, and compatibility with NISQ devices make it a powerful addition to the suite of error mitigation techniques available today.
However, its performance depends on factors such as the quality of the noisy state copies and the underlying noise model.
