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

# What additional options are available when using TREX?

## Overview
The main options when using TREX control the number of randomizations
and the random state for reproducibility. Unlike [REM](rem.md), TREX
does not require the user to provide a confusion matrix.

## Number of randomizations

The ``num_randomizations`` parameter controls how many random readout
twirling patterns are used. More randomizations give better accuracy
but require more circuit executions.

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
from mitiq.experimental.trex import execute_with_trex

qreg = [LineQubit(i) for i in range(2)]
circuit = Circuit(X.on_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))

def noisy_readout_executor(circuit, p0, p1, shots=8192) -> MeasurementResult:
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings, qubit_indices=(0, 1))

noisy_executor = partial(noisy_readout_executor, p0=0.15, p1=0.15)
ideal_value = -2.0

for num_rand in [4, 8, 16, 32, 64]:
    result = execute_with_trex(
        circuit,
        noisy_executor,
        observable,
        num_randomizations=num_rand,
        random_state=42,
    )
    error = abs(ideal_value - result)
    print(f"num_randomizations={num_rand:3d}: result={result:.4f}, error={error:.4f}")
```

```{note}
On small circuits like this 2-qubit example, the improvement from
additional randomizations may not be monotonic due to finite sampling
noise. The benefits of more randomizations become clearer on larger
circuits with more qubits.
```

## Random state for reproducibility

The ``random_state`` parameter accepts an integer seed or a
``np.random.RandomState`` object, allowing reproducible results.
Note that ``random_state`` controls the TREX twirling patterns;
for full reproducibility the executor must also be deterministic
(e.g., a noiseless simulator).

```{code-cell} ipython3
def deterministic_executor(circuit, shots=8192) -> MeasurementResult:
    """Noiseless executor for reproducibility demonstration."""
    simulator = NoisySingleQubitReadoutSampler(0, 0)
    result = simulator.run(circuit, repetitions=shots)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings, qubit_indices=(0, 1))

result1 = execute_with_trex(
    circuit, deterministic_executor, observable,
    num_randomizations=16, random_state=123,
)
result2 = execute_with_trex(
    circuit, deterministic_executor, observable,
    num_randomizations=16, random_state=123,
)
print(f"Result 1: {result1:.6f}")
print(f"Result 2: {result2:.6f}")
print(f"Reproducible: {result1 == result2}")
```

## Full output

Setting ``full_output=True`` returns a dictionary containing all
intermediate data from the TREX process, which can be useful for
debugging or analysis.

```{code-cell} ipython3
result, data = execute_with_trex(
    circuit, noisy_executor, observable,
    num_randomizations=8, random_state=42,
    full_output=True,
)

print(f"TREX value: {result:.4f}")
print(f"Keys: {list(data.keys())}")
for key, value in data.items():
    if isinstance(value, list):
        print(f"  {key}: list of {len(value)} items")
    else:
        print(f"  {key}: {value}")
```
