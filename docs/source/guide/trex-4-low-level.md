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

# What happens when I use TREX?

The workflow of TREX in Mitiq consists of the following steps:

1. The user provides a `QPROGRAM` (quantum circuit), an executor returning
   `MeasurementResult`, and an `Observable`.
2. For each commuting group in the observable, Mitiq creates measurement
   circuits with appropriate basis rotations.
3. For each randomization pattern, Mitiq inserts random X gates before
   measurement (readout twirling) and creates corresponding calibration
   circuits.
4. All twirled and calibration circuits are executed via the executor.
5. The measurement results are classically post-processed: bits are XOR'd
   with the randomization pattern to undo the X flips.
6. For each Pauli string, the noisy expectation is divided by the
   calibration factor to correct readout errors.
7. The corrected expectation value is returned to the user.

As shown in [How do I use TREX?](trex-1-intro.md), the function
{func}`.execute_with_trex()` applies TREX behind the scenes and directly
returns the error-mitigated expectation value. In the next sections, we
show how one can apply TREX at a lower level using
{func}`mitiq.experimental.trex.trex.construct_circuits` and {func}`mitiq.experimental.trex.trex.combine_results`.

## Constructing twirled circuits

```{code-cell} ipython3
from functools import partial

import numpy as np
from cirq import LineQubit, Circuit, X
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)

from mitiq import MeasurementResult
from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString
from mitiq.experimental.trex import construct_circuits, combine_results

qreg = [LineQubit(i) for i in range(2)]
circuit = Circuit(X.on_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))

twirled_circuits, calibration_circuits, randomization_strings = \
    construct_circuits(circuit, observable, num_randomizations=8, random_state=42)

print(f"Number of twirled circuits: {len(twirled_circuits)}")
print(f"Number of calibration circuits: {len(calibration_circuits)}")
print(f"Number of randomization strings: {len(randomization_strings)}")
```

Let's look at an example twirled circuit:

```{code-cell} ipython3
print("Original circuit:")
print(circuit)
print("\nTwirled measurement circuit (first randomization):")
print(twirled_circuits[0])
print("\nCalibration circuit (first randomization):")
print(calibration_circuits[0])
```

## Executing and combining results

```{code-cell} ipython3
def noisy_readout_executor(circuit, p0, p1, shots=8192) -> MeasurementResult:
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings, qubit_indices=(0, 1))

noisy_executor = partial(noisy_readout_executor, p0=0.15, p1=0.15)

# Execute all circuits
all_circuits = twirled_circuits + calibration_circuits
all_results = [noisy_executor(c) for c in all_circuits]

n_twirled = len(twirled_circuits)
twirled_results = all_results[:n_twirled]
calibration_results = all_results[n_twirled:]

# Combine results with TREX correction
corrected_value = combine_results(
    twirled_results, calibration_results,
    randomization_strings, observable,
)
print(f"TREX-corrected expectation value: {corrected_value:.4f}")
print("Ideal value: -2.0")
```

## Using `mitigate_executor` and `trex_decorator`

TREX can also be applied using the executor wrapper pattern:

```{code-cell} ipython3
from mitiq.experimental.trex import mitigate_executor

mitigated = mitigate_executor(
    noisy_executor, observable,
    num_randomizations=16, random_state=42,
)
result = mitigated(circuit)
print(f"Mitigated result: {result:.4f}")
```

Or with the decorator pattern:

```{code-cell} ipython3
from mitiq.experimental.trex import trex_decorator

@trex_decorator(observable, num_randomizations=16, random_state=42)
def my_executor(circuit) -> MeasurementResult:
    return noisy_executor(circuit)

result = my_executor(circuit)
print(f"Decorated result: {result:.4f}")
```
