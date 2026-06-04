---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{tags} cdr, cirq, basic
```

# CDR with a local noisy simulator

Clifford Data Regression (CDR) learns a correction for a noisy quantum
computer from circuits that are similar to the target circuit, but easier to
simulate classically. In this worked example, we use CDR to recover the
expectation value of a three-qubit variational circuit affected by
depolarizing noise.

Unlike zero-noise extrapolation, CDR does not fit results obtained at several
noise levels. It creates near-Clifford training circuits, evaluates them on
both the noisy executor and a noiseless simulator, and learns a map from noisy
to ideal expectation values. See the [CDR user guide](../guide/cdr.md) for a
complete description of the method and its options.

## Setup

This example uses only Cirq's local simulators, so it can run without access
to quantum hardware.

```{code-cell} ipython3
import warnings

import cirq
import matplotlib.pyplot as plt
import numpy as np

from mitiq import Observable, PauliString, cdr
from mitiq.interface.mitiq_cirq import compute_density_matrix

warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
```

## Define the target circuit

CDR requires circuits whose non-Clifford gates are single-qubit $R_Z$
rotations. The layer below combines those rotations with Clifford gates and
entangling CNOTs. Repeating it four times makes the effect of noise visible
while keeping the example quick to run.

```{code-cell} ipython3
q0, q1, q2 = cirq.LineQubit.range(3)

variational_layer = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.rz(0.37)(q1),
    cirq.CNOT(q1, q2),
    cirq.rz(-0.51)(q2),
    cirq.rx(np.pi / 2)(q0),
    cirq.rz(0.83)(q0),
)
circuit = variational_layer * 4

print(circuit)
```

We measure a small Hamiltonian made from two Pauli strings,

$$
O = Z_0 Z_1 + 0.5 X_1 X_2.
$$

The ideal expectation value is classically computable because this example
has only three qubits.

```{code-cell} ipython3
observable = Observable(
    PauliString("ZZI"),
    PauliString("IXX", coeff=0.5),
)
print(observable)
```

## Define ideal and noisy executors

An executor accepts a circuit and returns a result from which Mitiq can
compute the observable's expectation value. Here both executors return a
density matrix. The noisy executor adds single-qubit depolarizing noise after
each circuit moment, while the ideal executor has no noise.

The ideal executor serves two purposes:

1. It gives us a reference value for this small target circuit.
2. CDR uses it to label the near-Clifford training circuits.

For a larger problem, the second role could instead be handled by a dedicated
near-Clifford simulator.

```{code-cell} ipython3
noise_level = 0.03


def ideal_executor(circuit: cirq.Circuit) -> np.ndarray:
    """Return the noiseless final density matrix."""
    return compute_density_matrix(circuit, noise_level=(0.0,))


def noisy_executor(circuit: cirq.Circuit) -> np.ndarray:
    """Return a final density matrix affected by depolarizing noise."""
    return compute_density_matrix(circuit, noise_level=(noise_level,))
```

Before applying mitigation, compare the exact expectation value with the
value returned by the noisy executor.

```{code-cell} ipython3
ideal_value = observable.expectation(circuit, ideal_executor).real
noisy_value = observable.expectation(circuit, noisy_executor).real

print(f"Ideal expectation value: {ideal_value:.3f}")
print(f"Noisy expectation value: {noisy_value:.3f}")
```

## Execute with CDR

The four essential arguments to {func}`.cdr.execute_with_cdr` are:

- `circuit`: the target circuit whose expectation value we want.
- `executor`: the noisy device or simulator.
- `observable`: the Hermitian operator to measure.
- `simulator`: a noiseless simulator for the near-Clifford training circuits.

We also set the number of training circuits and a random seed so the example
is reproducible. For each training circuit, Mitiq obtains a noisy value from
`noisy_executor` and an ideal value from `ideal_executor`, then fits the
correction applied to the target circuit's noisy result.

```{code-cell} ipython3
cdr_value = cdr.execute_with_cdr(
    circuit,
    noisy_executor,
    observable=observable,
    simulator=ideal_executor,
    num_training_circuits=20,
    fraction_non_clifford=0.1,
    random_state=0,
).real

print(f"CDR-mitigated expectation value: {cdr_value:.3f}")
```

## Compare the results

The mitigated value is much closer to the ideal value than the raw noisy
result. The plot makes the correction visible, and the error comparison
quantifies the improvement.

```{code-cell} ipython3
noisy_error = abs(noisy_value - ideal_value)
cdr_error = abs(cdr_value - ideal_value)

print(f"Noisy absolute error: {noisy_error:.3f}")
print(f"CDR absolute error:   {cdr_error:.3f}")
print(f"Improvement factor:   {noisy_error / cdr_error:.1f}x")
```

```{code-cell} ipython3
labels = ["Ideal", "Noisy", "CDR"]
values = [ideal_value, noisy_value, cdr_value]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(labels, values, color=colors)
ax.axhline(ideal_value, color="#4C72B0", linestyle="--", alpha=0.7)
ax.set_ylabel("Expectation value")
ax.set_title("CDR recovers the ideal expectation value")
ax.set_ylim(0, 1)
plt.show()
```

CDR is most useful when the near-Clifford training circuits remain
classically tractable and their noisy behavior resembles that of the target
circuit. The number of training circuits, the fraction of non-Clifford gates,
and the fit function can all affect the result. The
[CDR options guide](../guide/cdr-3-options.md) explains these controls,
including variable-noise CDR through the `scale_factors` argument.
