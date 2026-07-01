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

# What additional options are available when using classical shadows?

```{include} shadows-experimental-note.md
```

```{code-cell} ipython3
:tags: [remove-cell]
from functools import partialmethod
import tqdm
tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)
```

```{code-cell} ipython3
import cirq
import numpy as np
from mitiq import MeasurementResult, PauliString
from mitiq.experimental import shadows
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)
```

The introductory section [How do I use classical shadows?](shadows-1-intro.md) covers
the basic usage of `shadows.shadow_quantum_processing`, `shadows.classical_post_processing`,
and `shadows.pauli_twirling_calibrate`.
Each of these functions has optional arguments that give you more control over the protocol.

## Problem setup

We use a 3-qubit GHZ state and a simple noiseless executor throughout this page.

```{code-cell} ipython3
qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
)

def execute(circuit: cirq.Circuit) -> MeasurementResult:
    return cirq_sample_bitstrings(circuit, noise_level=(0,), shots=1)
```

## Options for `shadow_quantum_processing`

`shadows.shadow_quantum_processing` accepts two optional arguments beyond the required
`circuit`, `executor`, and `num_total_measurements_shadow`.

**`random_seed`** seeds NumPy's random number generator before sampling Pauli bases.
This makes the measurement outcomes reproducible:

```{code-cell} ipython3
shadow_outcomes = shadows.shadow_quantum_processing(
    circuit, execute, num_total_measurements_shadow=200, random_seed=42
)
bitstrings, bases = shadow_outcomes
print("First five bases:", bases[:5])
```

Running again with the same seed produces the same bases:

```{code-cell} ipython3
shadow_outcomes_2 = shadows.shadow_quantum_processing(
    circuit, execute, num_total_measurements_shadow=200, random_seed=42
)
assert shadow_outcomes_2[1][:5] == bases[:5]
print("Bases match:", shadow_outcomes_2[1][:5] == bases[:5])
```

**`qubits`** explicitly specifies which qubits to measure.
This is useful when the circuit's qubit ordering differs from what you want to track,
or when you want to restrict measurements to a subset of qubits:

```{code-cell} ipython3
shadow_outcomes = shadows.shadow_quantum_processing(
    circuit,
    execute,
    num_total_measurements_shadow=200,
    qubits=[qubits[0], qubits[1], qubits[2]],
)
```

## Options for `classical_post_processing`

`shadows.classical_post_processing` supports two main modes — expectation value
estimation and state reconstruction — controlled by the `observables` and
`state_reconstruction` arguments.

### Expectation value estimation

Pass a list of `PauliString` observables to estimate their expectation values.
The `k_shadows` parameter controls the number of batches used in the median-of-means
estimator.
A larger `k_shadows` reduces the influence of outlier snapshots at the cost of using
more measurements per estimate:

```{code-cell} ipython3
shadow_outcomes = shadows.shadow_quantum_processing(
    circuit, execute, num_total_measurements_shadow=1000
)

observables = [
    PauliString("ZZ", support=(i, i + 1), coeff=1)
    for i in range(len(qubits) - 1)
]

# k_shadows=1 uses all snapshots in a single batch (default)
result_k1 = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    observables=observables,
    k_shadows=1,
)

# k_shadows=10 splits snapshots into 10 batches and takes the median
result_k10 = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    observables=observables,
    k_shadows=10,
)

print("k_shadows=1:", result_k1)
print("k_shadows=10:", result_k10)
```

### State reconstruction

Pass `state_reconstruction=True` to reconstruct the full density matrix by averaging
all classical snapshots.
This does not use `k_shadows` or `observables`:

```{code-cell} ipython3
result = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    state_reconstruction=True,
)
print("Reconstructed state shape:", result["reconstructed_state"].shape)
```

```{warning}
Density matrix reconstruction scales as $4^n$ with the number of qubits.
For large systems, expectation value estimation is far more practical.
```

## Options for `pauli_twirling_calibrate`

`shadows.pauli_twirling_calibrate` characterizes the noise channel for robust shadow
estimation.

**`k_calibration`** sets the number of batches in the median-of-means estimator for
the Pauli fidelities.
Increasing it reduces sensitivity to outlier calibration shots:

**`locality`** restricts which Pauli fidelity terms are estimated.
If your observables act on at most `w` qubits, setting `locality=w` reduces the number
of fidelity terms from $2^n$ to $\sum_{i=0}^{w} \binom{n}{i}$, which can substantially
reduce memory and computation for large systems.
The `locality` value should match the weight of the heaviest observable you plan to
estimate:

```{code-cell} ipython3
def noisy_execute(circuit: cirq.Circuit) -> MeasurementResult:
    return cirq_sample_bitstrings(circuit, noise_level=(0.05,), shots=1)

calibration_results = shadows.pauli_twirling_calibrate(
    k_calibration=2,
    locality=2,       # match the weight of your heaviest observable
    qubits=qubits,
    executor=noisy_execute,
    num_total_measurements_calibration=5000,
)
print("Number of fidelity terms:", len(calibration_results))
```

**`zero_state_shadow_outcomes`** lets you pass pre-collected calibration data instead
of having `pauli_twirling_calibrate` run new circuits.
This is useful when you have already collected calibration measurements and want to
avoid redundant circuit executions:

```{code-cell} ipython3
# Collect calibration data manually
import cirq as cirq_
zero_circuit = cirq_.Circuit()
calibration_outcomes = shadows.shadow_quantum_processing(
    zero_circuit,
    noisy_execute,
    num_total_measurements_shadow=5000,
    qubits=qubits,
)

# Pass the pre-collected data directly
calibration_results = shadows.pauli_twirling_calibrate(
    k_calibration=2,
    locality=2,
    zero_state_shadow_outcomes=calibration_outcomes,
)
print("Number of fidelity terms:", len(calibration_results))
```

Once calibration results are available, pass them to `classical_post_processing`:

```{code-cell} ipython3
shadow_outcomes = shadows.shadow_quantum_processing(
    circuit, noisy_execute, num_total_measurements_shadow=3000
)

calibrated = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    calibration_results=calibration_results,
    observables=observables,
    k_shadows=1,
)
print("Calibrated estimates:", calibrated)
```

```{note}
You do not need to re-run `pauli_twirling_calibrate` between experiments as long as
the number of qubits and the noise channel have not changed.
```
