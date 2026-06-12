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

# CDR with a compiled Cirq circuit

Clifford Data Regression (CDR) learns a correction for a noisy quantum
computer from circuits that are similar to the target circuit, but easier to
simulate classically. In this worked example, we start from a four-qubit Cirq
circuit with a mix of high-level rotations and entangling gates, compile it to
a CDR-compatible gate set, and then mitigate a local depolarizing-noise
simulation.

CDR requires the non-Clifford content of the circuit to be contained in
single-qubit $R_Z$ rotations. This example makes that compilation step
explicit by translating the $R_Y$ rotations in the original circuit into the
$\{R_Z, \sqrt{X}, \mathrm{CNOT}\}$ basis before calling
{func}`.cdr.execute_with_cdr`.

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

## Define a circuit before compilation

The circuit below has four qubits, alternating entangling layers, and several
single-qubit rotations. It is intentionally written in a convenient Cirq form
first, using $R_Y$ rotations that are not part of the CDR-compatible basis.

```{code-cell} ipython3
q0, q1, q2, q3 = cirq.LineQubit.range(4)

problem_layer = cirq.Circuit(
    cirq.ry(0.41)(q0),
    cirq.ry(-0.27)(q1),
    cirq.ry(0.62)(q2),
    cirq.ry(-0.35)(q3),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q2, q3),
    cirq.rz(0.53)(q1),
    cirq.rz(-0.71)(q2),
    cirq.CNOT(q1, q2),
    cirq.rz(0.37)(q0),
    cirq.rz(-0.49)(q3),
    cirq.CNOT(q2, q3),
)
problem_circuit = problem_layer * 3

print(problem_circuit)
```

We measure a small Hamiltonian made from three Pauli strings,

$$
O = Z_0 Z_1 + 0.5 X_1 X_2 - 0.25 Z_2 Z_3.
$$

The ideal expectation value is classically computable because this example
has only four qubits.

```{code-cell} ipython3
observable = Observable(
    PauliString("ZZII"),
    PauliString("IXXI", coeff=0.5),
    PauliString("IIZZ", coeff=-0.25),
)
print(observable)
```

## Compile to a CDR-compatible gate set

The circuit of interest must be compiled so that all non-Clifford gates are
$R_Z$ rotations. Instead of manually rewriting individual gates, we use the
general `cirq.decompose` utility and tell it which operations should be kept
in the compiled circuit.

The predicate below keeps $R_Z$, $\sqrt{X}$, and CNOT operations. Cirq then
decomposes the unsupported $R_Y$ rotations from the original circuit into
that CDR-compatible basis.

```{code-cell} ipython3
def is_cdr_basis_operation(op: cirq.Operation) -> bool:
    gate = op.gate
    return (
        isinstance(gate, cirq.ZPowGate)
        or (
            isinstance(gate, cirq.XPowGate)
            and np.isclose(abs(float(gate.exponent)), 0.5)
        )
        or (
            isinstance(gate, cirq.CNotPowGate)
            and np.isclose(float(gate.exponent), 1.0)
        )
    )


circuit = cirq.Circuit(
    cirq.decompose(
        problem_circuit,
        keep=is_cdr_basis_operation,
        on_stuck_raise=lambda op: ValueError(
            f"Could not compile {op!r} to the CDR basis."
        ),
    )
)

print(circuit)
```

Check that the compiled circuit has the expected basis. The inverse
$\sqrt{X}$ gate is also Clifford, so it is allowed.

```{code-cell} ipython3
print(all(is_cdr_basis_operation(op) for op in circuit.all_operations()))
```

The compilation preserves the ideal expectation value up to numerical
precision.

```{code-cell} ipython3
def ideal_executor(circuit: cirq.Circuit) -> np.ndarray:
    """Return the noiseless final density matrix."""
    return compute_density_matrix(circuit, noise_level=(0.0,))


uncompiled_value = observable.expectation(problem_circuit, ideal_executor).real
compiled_value = observable.expectation(circuit, ideal_executor).real

print(f"Uncompiled ideal value: {uncompiled_value:.6f}")
print(f"Compiled ideal value:   {compiled_value:.6f}")
print(f"Absolute difference:    {abs(uncompiled_value - compiled_value):.2e}")
```

## Define ideal and noisy executors

An executor accepts a circuit and returns a result from which Mitiq can
compute the observable's expectation value. Here both executors return a
density matrix. The noisy executor adds single-qubit depolarizing noise after
each circuit moment, while the ideal executor has no noise.

```{code-cell} ipython3
noise_level = 0.018


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

- `circuit`: the compiled target circuit whose expectation value we want.
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
    num_training_circuits=24,
    fraction_non_clifford=0.12,
    random_state=0,
).real

print(f"CDR-mitigated expectation value: {cdr_value:.3f}")
```

## Compare the results

The plot and error comparison show how the CDR correction changes the raw
noisy estimate after the circuit has been compiled to a compatible basis.

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
ax.set_title("CDR after compiling to a compatible gate set")
ax.set_ylim(min(values) - 0.1, max(values) + 0.1)
plt.show()
```

CDR is most useful when the near-Clifford training circuits remain
classically tractable and their noisy behavior resembles that of the target
circuit. The number of training circuits, the fraction of non-Clifford gates,
and the fit function can all affect the result. The
[CDR options guide](../guide/cdr-3-options.md) explains these controls,
including variable-noise CDR through the `scale_factors` argument.
