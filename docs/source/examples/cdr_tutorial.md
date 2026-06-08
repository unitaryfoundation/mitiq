---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{tags} cdr, cirq, basic
```

# Clifford data regression (CDR) with Cirq

Clifford data regression (CDR) {cite}`Czarnik_2021_Quantum` mitigates errors by
learning a correction from circuits that resemble the circuit of interest but
are classically simulable. Mitiq builds near-Clifford *training circuits*, runs
each one on both the noisy executor and a noiseless simulator, and fits a map
from noisy to ideal expectation values. That map is then applied to the noisy
result of the target circuit.

This tutorial runs CDR end to end on a small circuit using only Cirq's local
simulators, so no quantum hardware is needed. It also covers variable-noise
CDR {cite}`Lowe_2021_PRR`, which combines CDR with extra noise-scaled data and
often improves the result further. For a full description of the method and its
options, see the [CDR user guide](../guide/cdr.md).

## Setup

```{code-cell} ipython3
import warnings

import cirq
import matplotlib.pyplot as plt
import numpy as np

from mitiq import cdr, Observable, PauliString
from mitiq.interface.mitiq_cirq import compute_density_matrix

warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
```

## Target circuit and observable

CDR expects the non-Clifford gates of the circuit to be single-qubit $R_Z$
rotations. The layer below mixes those rotations with Clifford gates ($H$,
`CNOT`, and an $R_X(\pi/2)$). Repeating it a few times makes the effect of
noise visible while keeping the circuit small enough to simulate exactly.

```{code-cell} ipython3
q0, q1, q2 = cirq.LineQubit.range(3)


def layer(theta: float) -> list:
    return [
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.rz(theta)(q1),
        cirq.CNOT(q1, q2),
        cirq.rz(theta / 2)(q2),
        cirq.rx(np.pi / 2)(q0),
        cirq.rz(1.3 * theta)(q0),
    ]


circuit = cirq.Circuit([layer(0.5), layer(-0.8), layer(1.1), layer(-0.4)])
print(circuit)
```

We estimate the expectation value of a two-term Hamiltonian,

$$
O = Z_0 Z_1 + 0.5 \, X_1 X_2 .
$$

```{code-cell} ipython3
observable = Observable(PauliString("ZZI"), PauliString("IXX", coeff=0.5))
print(observable)
```

## Ideal and noisy executors

An executor maps a circuit to a result that Mitiq uses to compute the
observable. Here both executors return a density matrix. The noisy one applies
single-qubit depolarizing noise after each moment; the ideal one is noiseless.
CDR uses the ideal simulator to label its training circuits, and we also use it
here to get a reference value for this small example.

```{code-cell} ipython3
noise_level = 0.03


def ideal_executor(circuit: cirq.Circuit) -> np.ndarray:
    return compute_density_matrix(circuit, noise_level=(0.0,))


def noisy_executor(circuit: cirq.Circuit) -> np.ndarray:
    return compute_density_matrix(circuit, noise_level=(noise_level,))
```

Before mitigating, compare the exact value with the noisy one.

```{code-cell} ipython3
ideal_value = observable.expectation(circuit, ideal_executor).real
noisy_value = observable.expectation(circuit, noisy_executor).real

print(f"Ideal expectation value: {ideal_value:.4f}")
print(f"Noisy expectation value: {noisy_value:.4f}")
```

## Standard CDR

The required arguments to {func}`.cdr.execute_with_cdr` are the target
`circuit`, the noisy `executor`, the `observable`, and a noiseless `simulator`
for the training circuits. We also fix `num_training_circuits` and a
`random_state` for reproducibility.

```{code-cell} ipython3
cdr_value = cdr.execute_with_cdr(
    circuit,
    noisy_executor,
    observable=observable,
    simulator=ideal_executor,
    num_training_circuits=40,
    fraction_non_clifford=0.15,
    random_state=1,
).real

print(f"CDR-mitigated value: {cdr_value:.4f}")
```

## Variable-noise CDR

Variable-noise CDR adds data taken at higher noise levels, combining CDR with
the noise scaling used in zero-noise extrapolation. Passing several
`scale_factors` is all that is needed.

```{code-cell} ipython3
vncdr_value = cdr.execute_with_cdr(
    circuit,
    noisy_executor,
    observable=observable,
    simulator=ideal_executor,
    num_training_circuits=40,
    fraction_non_clifford=0.15,
    scale_factors=(1, 2, 3),
    random_state=1,
).real

print(f"Variable-noise CDR value: {vncdr_value:.4f}")
```

## Compare the results

```{code-cell} ipython3
errors = {
    "Noisy": abs(noisy_value - ideal_value),
    "CDR": abs(cdr_value - ideal_value),
    "vnCDR": abs(vncdr_value - ideal_value),
}
for name, err in errors.items():
    print(f"{name:>6} absolute error: {err:.4f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(errors.keys(), errors.values(), color=["#DD8452", "#55A868", "#4C72B0"])
ax.set_ylabel("Absolute error")
ax.set_title("CDR and variable-noise CDR reduce the error")
plt.show()
```

Both methods move the noisy estimate toward the ideal value, and on this
circuit variable-noise CDR does noticeably better than standard CDR. The number
of training circuits, the fraction of non-Clifford gates, the scale factors,
and the fit function all influence the result. The
[CDR options guide](../guide/cdr-3-options.md) describes these controls in
detail.
