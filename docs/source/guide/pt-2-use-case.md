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

# When should I use PT?

## Advantages

The main advantage of Pauli Twirling (PT) is that it converts arbitrary, and potentially coherent, noise into stochastic Pauli (incoherent) noise. This conversion is highly beneficial due to how these two types of noise accumulate:
- **Coherent noise** (e.g., systematic gate over-rotations, calibration errors) maintains quantum superposition but shifts the state vector in a deterministic direction. When similar coherent errors occur repeatedly in a deep circuit, they can add up constructively, causing the overall error to accumulate *quadratically* with the circuit depth.
- **Incoherent noise** (e.g., depolarization, stochastic Pauli bit-flips or phase-flips) represents a random mixture of quantum states. Because the errors are stochastic rather than deterministic, they do not add constructively. Thus, incoherent noise accumulates only *linearly* with circuit depth.

By randomly conjugating noisy gates with Pauli operators and averaging the results, PT effectively destroys the deterministic coherence of the errors, scattering them into random Pauli errors. This suppresses the worst-case quadratic error scaling and bounds the problem. More details are given in [What is the theory behind PT?](pt-5-theory.md).

Additionally, PT is noise-agnostic and easy to use, requiring no prior knowledge of the underlying noise model to operate. Because it restructures arbitrary errors into well-behaved stochastic channels, it is frequently used as a noise tailoring technique to improve the performance of downstream quantum error mitigation (QEM) methods, such as [ZNE](zne.md), [PEC](pec.md), and [CDR](cdr.md). See the [stacking PT with QEM section](pt-3-options.md) for details.

## Disadvantages

A practical disadvantage of PT in Mitiq is that inserting single-qubit Pauli twirling gates increases the overall circuit depth unless these gates are compiled into adjacent single-qubit layers. Because Mitiq does not currently perform this compilation step automatically, users execute slightly deeper circuits, although the overhead is often negligible since single-qubit gate errors are typically small.

Furthermore, PT does not reduce the total error rate; it only restructures the noise. If the original coherent errors are large relative to the gate fidelity, PT can structure the noise into a nearly completely depolarizing channel, causing the loss of quantum information. Therefore, PT should be viewed primarily as a noise *tailoring* technique rather than a standalone noise *reduction* method.
