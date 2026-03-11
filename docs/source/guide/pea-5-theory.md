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

# What is the theory behind PEA?

Probabilistic error amplification (PEA) {cite}`Kim_2023_Nature` combines probabilistic sampling of noise-amplified circuits with an extrapolation step similar to zero-noise extrapolation (ZNE).

## Noise-amplified representations

Consider an ideal circuit made of operations $\mathcal G_i$.
PEA constructs, for each scale factor $s$, a *noise-amplified representation* in which each ideal operation is expressed as a linear combination of implementable noisy operations $\mathcal O_{i,\alpha}^{(s)}$ corresponding to a noise level scaled by $s$:

$$
\mathcal G_i = \sum_\alpha \eta_{i,\alpha}^{(s)}\, \mathcal O_{i,\alpha}^{(s)}.
$$

The coefficients $\eta_{i,\alpha}^{(s)}$ form a quasi-probability representation with one-norm $\gamma_i^{(s)} = \sum_\alpha |\eta_{i,\alpha}^{(s)}|$.

## Monte Carlo estimation at each scale factor

For a fixed scale factor $s$, the overall quasi-probability representation of the circuit induces a probability distribution $p(\vec{\alpha}) = |\eta_{\vec{\alpha}}^{(s)}|/\gamma^{(s)}$, where $\gamma^{(s)} = \prod_i \gamma_i^{(s)}$ is the product of the gate-wise one-norms.
Sampling a noisy circuit $\Phi_{\vec{\alpha}}^{(s)}$ from this distribution and applying a sign $\sigma_{\vec{\alpha}} = \mathrm{sign}(\eta_{\vec{\alpha}}^{(s)})$ yields an unbiased estimator of the expectation value at noise scale $s$:

$$
E^{(s)} = \gamma^{(s)}\, \mathbb E\left[\sigma_{\vec{\alpha}}\, \langle A \rangle_{\vec{\alpha}}\right].
$$

This is exactly what is computed when running {func}`mitiq.pea.pea.combine_results`.

## Extrapolation to the zero-noise limit

After obtaining $E^{(s)}$ for several scale factors $s$, PEA applies a ZNE inference method to extrapolate to the zero-noise limit.
The resulting value is the PEA estimate of the ideal expectation value.
See [](./zne-5-theory.md) for more details about the theory of extrapolation in ZNE.

```{tip}
Because the sampling overhead grows with $\gamma^{(s)}$, PEA is most effective when the amplified representations remain reasonably low-norm (which means the ideal operations are close to the implementable ones) and the chosen scale factors are not too large.
```
