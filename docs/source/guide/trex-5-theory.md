---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What is the theory behind TREX?

Twirled Readout Error eXtinction (TREX) is a model-free readout error
mitigation technique introduced in {cite}`vandenBerg_2022_NatPhys`.
It is based on the idea of *readout twirling*: randomly applying $X$
gates before measurement and classically undoing the flips in
post-processing.

## Readout error model

When measuring $n$ qubits, readout errors are characterized by a
[left-stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix) $A$ where entry $A_{i,j}$ represents the
probability of measuring state $|i\rangle$ when the true state is
$|j\rangle$. This transforms the ideal probability distribution
$\mathbf{p}$ into a noisy one:

$$
\tilde{\mathbf{p}} = A \mathbf{p}
$$

Standard readout error mitigation (e.g., [REM](rem.md)) requires
explicit knowledge of $A$ or its inverse. TREX avoids this requirement.

## Readout twirling

The key idea of TREX is to apply a random string of Pauli $X$
operators $X^{\mathbf{s}}$ immediately before measurement, where
$\mathbf{s} \in \{0, 1\}^n$ is sampled uniformly at random. After
measurement, the classical outcome $\mathbf{x}$ is XOR'd with
$\mathbf{s}$ to undo the bit flips: $\mathbf{y} = \mathbf{x} \oplus \mathbf{s}$.

This creates a *twirled* noise map:

$$
A_\star = \frac{1}{2^n} \sum_{\mathbf{s}} X^{\mathbf{s}} A X^{\mathbf{s}}
$$

Note that the sum is over all $2^n$ possible bitstrings, but this is a
*mathematical* description of the twirled channel. In practice, we do
**not** need to enumerate all $2^n$ strings. Instead, we sample a small
number $N$ of random strings (the ``num_randomizations`` parameter) and
average the corrected estimates, which converges to the correct value.

The key insight is that this twirling transforms the
noise channel so that the computational basis states are eigenvectors
of $A_\star$, effectively *diagonalizing* the readout error channel.

## Eigenvalue correction

After twirling, each Pauli observable $P$ with support $\mathbf{w}$
(the set of qubits where $P$ acts non-trivially) has a corresponding
eigenvalue:

$$
\lambda_{\mathbf{w}} = \frac{1}{2^n} \sum_{\mathbf{a}, \mathbf{b}} (-1)^{\langle \mathbf{w}, \mathbf{a} + \mathbf{b} \rangle} A_{\mathbf{a}, \mathbf{b}}
$$

where $A_{\mathbf{a}, \mathbf{b}}$ is the entry of the readout error
matrix giving the probability of observing outcome $\mathbf{a}$ when
the true state is $\mathbf{b}$.

The true (noiseless) expectation value of $P$ can be recovered by
dividing the noisy (twirled) expectation by this eigenvalue:

$$
\langle P \rangle_{\text{true}} = \frac{\langle P \rangle_{\text{noisy, twirled}}}{\lambda_{\mathbf{w}}}
$$

## Calibration

The eigenvalue $\lambda_{\mathbf{w}}$ is estimated using *calibration
circuits*: circuits that prepare the $|0\ldots0\rangle$ state (no
quantum operations), apply the same readout twirling, and measure.
After XOR correction, the ideal calibration result should be all zeros.
Any deviation indicates readout errors, and the parity computed on the
support qubits gives an estimate of $\lambda_{\mathbf{w}}$.

## TREX protocol

The complete TREX protocol for estimating $\langle P \rangle$:

1. For each randomization $j = 1, \ldots, N$:
   - Sample a random $n$-bit string $\mathbf{s}_j$.
   - **Circuit execution**: Run $U$, apply $X^{\mathbf{s}_j}$, measure
     to get bitstrings $\{\mathbf{x}_k\}$. Compute
     $\mathbf{y}_k = \mathbf{x}_k \oplus \mathbf{s}_j$ and
     $f_j^{\text{circuit}} = \frac{1}{K}\sum_k (-1)^{\sum_{i \in \mathbf{w}} y_{k,i}}$.
   - **Calibration**: Prepare $|0\ldots0\rangle$, apply
     $X^{\mathbf{s}_j}$, measure to get $\{\mathbf{x}'_k\}$. Compute
     $\mathbf{y}'_k = \mathbf{x}'_k \oplus \mathbf{s}_j$ and
     $f_j^{\text{calib}} = \frac{1}{K}\sum_k (-1)^{\sum_{i \in \mathbf{w}} y'_{k,i}}$.
   - **Corrected**: $\hat{P}_j = f_j^{\text{circuit}} / f_j^{\text{calib}}$.

2. Final estimate:
   $\langle P \rangle_{\text{TREX}} = \frac{1}{N} \sum_{j=1}^N \hat{P}_j$.

## Asymmetric readout errors

TREX handles asymmetric readout errors (where $\mathrm{Pr}(0 \to 1)
\neq \mathrm{Pr}(1 \to 0)$) naturally. The readout twirling procedure
symmetrizes the noise channel by averaging over random $X$ flips,
effectively converting any asymmetric readout noise into a symmetric
(diagonal) form. The calibration circuits then estimate the resulting
eigenvalues, so no explicit knowledge of the asymmetry is required.

## References

The TREX technique is described in detail in:

- E. van den Berg, Z. K. Minev, and K. Temme, "Model-free readout-error
  mitigation for quantum expectation values,"
  [arXiv:2012.09738](https://arxiv.org/abs/2012.09738) (2020).
  Published in *Nature Physics* 18, 1116-1121 (2022).
