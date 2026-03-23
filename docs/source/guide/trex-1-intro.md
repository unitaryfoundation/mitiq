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

# How do I use TREX?

As with all techniques, TREX is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

## Problem setup
In this example we will simulate a noisy device to demonstrate the capabilities of TREX. This method requires an {doc}`observable <observables>` to be defined, and we use $Z_0 + Z_1$ as an example.
Since the circuit includes an $X$ gate on each qubit, the noiseless expectation value should be $-2$.

```{code-cell} ipython3
from cirq import LineQubit, Circuit, X

from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString

qreg = [LineQubit(i) for i in range(2)]
circuit = Circuit(X.on_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))

print(circuit)
```

Next we define a simple noisy readout executor function which takes a
circuit as input, executes the circuit on a noisy simulator, and
returns the raw measurement results. See the [Executors](executors.md)
section for more information on how to define more advanced executors.

```{warning}
TREX executors require bitstrings as output since the technique applies
to raw measurement results.
```

```{code-cell} ipython3
from functools import partial

import numpy as np
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)

from mitiq import MeasurementResult

def noisy_readout_executor(circuit, p0, p1, shots=8192) -> MeasurementResult:
    # Replace with code based on your frontend and backend.
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings, qubit_indices=(0, 1))
```

The [executor](executors.md) can be used to evaluate noisy (unmitigated)
expectation values.

```{code-cell} ipython3
from mitiq.raw import execute as raw_execute

# Compute the expectation value of the observable.
# Use a noisy executor that has a 15% chance of bit flipping
p_flip = 0.15
noisy_executor = partial(noisy_readout_executor, p0=p_flip, p1=p_flip)
noisy_value = raw_execute(circuit, noisy_executor, observable)

ideal_executor = partial(noisy_readout_executor, p0=0, p1=0)
ideal_value = raw_execute(circuit, ideal_executor, observable)
error = abs((ideal_value - noisy_value) / ideal_value)
print(f"Error without mitigation: {error:.3}")
```

## Apply TREX
TREX can be easily applied with the function {func}`.execute_with_trex()`.
Unlike [REM](rem.md), TREX does not require an explicit inverse confusion
matrix. It automatically runs calibration circuits to estimate and correct
readout errors.

```{code-cell} ipython3
from mitiq.experimental.trex import execute_with_trex

mitigated_result = execute_with_trex(
    circuit,
    noisy_executor,
    observable,
    num_randomizations=32,
    random_state=42,
)
```

```{code-cell} ipython3
error = abs((ideal_value - mitigated_result) / ideal_value)
print(f"Error with mitigation (TREX): {error:.3}")
```

Here we observe that the application of TREX reduces the readout error when
compared to the unmitigated result.

The section [What additional options are available when using TREX?](trex-3-options.md) contains more information on
configuring TREX parameters.
