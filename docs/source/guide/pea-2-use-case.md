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

- Is often more accurate than ZNE given PEA takes advantages of a given noise model, rather than being agnostic.
- Lower sampling overhead than required by PEC.
- Allows you to run deeper circuit than with ZNE where unitary folding would create circuits longer than qubit coherence times.

## Disadvantages

PEA also has limitations:

- Requires a reasonably accurate noise model and baseline noise estimate (e.g. by sparse Pauli–Lindblad tomography).
- The sampling overhead can become large as the scale factor increases, since the one-norm of the representation grows and more samples are required.
- The final extrapolation step can be sensitive to statistical noise and to the choice of scale factors.
- In Mitiq, PEA currently supports local and global depolarizing noise models and assumes circuits can be decomposed into one- and two-qubit operations.

## Example

For a demonstration of PEA on superconducting hardware, see the study in {cite}`Kim_2023_Nature`, and for more information generally about tradeoffs find PEA on [The QEM Zoo](https://qemzoo.com/technique.html?id=pea).
