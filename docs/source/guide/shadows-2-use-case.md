---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# When Should I Use Classical Shadows?

## Advantages

Classical shadows can predict $M$ expectation values simultaneously from a single set of measurements, with a sample complexity that grows only as $\mathcal{O}(\log M)$ in the number of observables.
This makes it substantially more efficient than running separate experiments for each observable, and exponentially more efficient than full state tomography.

The protocol requires no knowledge of the underlying noise model.
For noisy devices, the robust shadow estimation variant can calibrate out noise on the rotation gates and measurements using only a short calibration experiment on the $|0\rangle^{\otimes n}$ state, without requiring full gate tomography.

Once a set of shadow measurements is collected, it can be reused to estimate any observable whose shadow norm is within the bound used to determine the number of measurements.
This makes the protocol flexible when the set of observables of interest is not known in advance.

## Disadvantages

Classical shadows requires an executor that returns a **single-shot** measurement result — one bitstring per circuit execution.
Some backend interfaces return averaged expectation values rather than individual bitstrings, and may need to be adapted.

Mitiq currently implements only the **random Pauli (local Clifford)** measurement ensemble.
Global Clifford measurements, which offer better sample complexity for observables with bounded Hilbert-Schmidt norm, are not yet supported.

The calibration step in robust shadow estimation characterizes noise on the Clifford rotation gates and the computational basis measurement.
It does **not** mitigate noise that occurs during state preparation, so the technique is best applied when state preparation errors are small relative to measurement errors.

Density matrix reconstruction scales exponentially with the number of qubits, since the reconstructed matrix has $4^n$ entries.
For large systems, expectation value estimation is far more practical than full state reconstruction.

## Example

For a worked example showing classical shadows applied to state tomography and expectation value estimation, see:

[Classical Shadows Protocol with Cirq](../examples/shadows_tutorial.md)

For a worked example of robust shadow estimation under noise, see:

[Robust Shadow Estimation with Mitiq](../examples/rshadows_tutorial.md)
