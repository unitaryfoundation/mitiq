---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When should I use TREX?

## Advantages

- **Model-free**: Unlike [REM](rem.md), TREX does not require an explicit
  confusion matrix or prior knowledge of readout error rates. It
  automatically estimates and corrects readout errors using calibration
  circuits.

- **Handles correlated readout errors**: The readout twirling procedure
  diagonalizes the readout error channel, which means it works even when
  readout errors are correlated between qubits.

- **Simple integration**: TREX only requires that the executor returns raw
  bitstrings (``MeasurementResult``). It can be combined with other
  error mitigation techniques that address different noise sources (e.g.,
  [PT](pt.md) for gate errors, [ZNE](zne.md) for noise extrapolation).

- **Mathematically rigorous**: TREX provides provable error bounds on
  the corrected expectation values when the measurement errors are small.

## Disadvantages

- **Execution overhead**: TREX requires running additional circuits for
  both the readout twirling (multiple randomizations of the original
  circuit) and calibration (identity circuits with twirling). The total
  number of circuit executions scales as
  ``num_randomizations * (num_groups + 1)``, where ``num_groups`` is the
  number of commuting groups in the observable.

- **Shot overhead**: Each randomization requires its own set of measurement
  shots. For a fixed total shot budget, increasing the number of
  randomizations decreases the shots per randomization, which introduces
  a trade-off between twirling quality and per-circuit statistical
  precision.

- **Readout error sensitivity**: When readout errors are very large,
  the calibration factors can become small, leading to noisy corrected
  values. In extreme cases, the correction may amplify statistical noise.

## TREX vs REM

| Feature | TREX | REM |
|---------|------|-----|
| Requires confusion matrix | No | Yes |
| Handles correlated errors | Yes | Only with full confusion matrix |
| Additional circuits needed | Yes (calibration + twirling) | No |
| Scaling | Linear in randomizations | Depends on matrix inversion |

Find more information on TREX on the [QEM Zoo](https://qemzoo.com/technique.html?id=trex).
