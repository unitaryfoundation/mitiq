---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When should I use PEA?

## Advantages

Probabilistic error amplification (PEA) can be useful when:

- Provides higher accuracy than ZNE because it leverages a specified noise model rather than being noise-agnostic.
- Requires lower sampling overhead than PEC.
- Enables execution of deeper circuits than with ZNE, in cases where unitary folding would create circuits longer than qubit coherence times.
- Reuses information learned from ZNE experiments to improve PEA performance.

## Disadvantages

PEA also has limitations:

- Requires a reasonably accurate noise model and baseline noise estimate (e.g. by sparse Pauli–Lindblad tomography).
- The sampling overhead can become large as the scale factor increases, since the one-norm of the representation grows and more samples are required.
- The final extrapolation step can be sensitive to statistical noise and to the choice of scale factors.
- In Mitiq, PEA currently supports local and global depolarizing noise models and assumes circuits can be decomposed into one- and two-qubit operations.

## Example

For a demonstration of PEA on superconducting hardware, see the study in {cite}`Kim_2023_Nature`, and for more information generally about tradeoffs find PEA on [The QEM Zoo](https://qemzoo.com/technique.html?id=pea).
