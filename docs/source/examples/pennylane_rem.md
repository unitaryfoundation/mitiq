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

```{tags} rem, pennylane, basic
```

# Error mitigation with PennyLane: Readout Error Mitigation (REM)

Mitiq already has a PennyLane tutorial demonstrating [zero-noise extrapolation (ZNE)](pennylane-ibmq-backends.md) via `pennylane.mitigate_with_zne`.
This example complements it by showing how to use [readout error mitigation (REM)](../guide/rem.md) with a circuit defined in PennyLane.

ZNE changes the effective noise level during circuit execution and extrapolates results to the zero-noise limit.
REM instead targets the classical readout channel: it corrects measurement results using a confusion matrix (or its inverse).

In this tutorial we:

1. Define a simple PennyLane circuit.
2. Convert it to a Mitiq circuit with {func}`mitiq.interface.mitiq_pennylane.from_pennylane`.
3. Simulate readout noise with a custom [executor](../guide/executors.md).
4. Apply REM and compare ideal, noisy, and mitigated $\langle Z \rangle$ values.

## Setup

```{code-cell} ipython3
import cirq
import numpy as np
import pennylane as qml
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)

from mitiq import MeasurementResult, rem
from mitiq.interface.mitiq_pennylane import from_pennylane
from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString
from mitiq.raw import execute as raw_execute
```

## Define a PennyLane circuit

We use the same single-qubit circuit as in the PennyLane + ZNE tutorial: ten Pauli $X$ gates on one wire.
The noiseless expectation value of $Z$ **should evaluate to one**, matching the ZNE example linked above.

```{code-cell} ipython3
def pennylane_circuit():
    for _ in range(10):
        qml.PauliX(wires=0)


tape = qml.tape.make_qscript(pennylane_circuit)()
circuit = from_pennylane(tape)
observable = Observable(PauliString("Z"))

print(circuit)
```

## Noisy readout executor

REM requires an executor that returns raw measurement results as a {class}`mitiq.MeasurementResult`.
When using an {class}`mitiq.Observable`, Mitiq automatically appends the correct measurement operations before calling the executor.

We model independent single-qubit readout errors with Cirq's `NoisySingleQubitReadoutSampler`.
The executor factory below keeps the `-> MeasurementResult` return annotation, which Mitiq uses to route results correctly.

```{code-cell} ipython3
P0 = 0.15
P1 = 0.15
SHOTS = 10_000


def make_readout_executor(p0: float, p1: float, shots: int = SHOTS):
    def executor(circuit: cirq.Circuit) -> MeasurementResult:
        simulator = NoisySingleQubitReadoutSampler(p0, p1)
        result = simulator.run(circuit, repetitions=shots)
        bitstrings = np.column_stack(list(result.measurements.values()))
        return MeasurementResult(bitstrings, qubit_indices=(0,))

    return executor


ideal_executor = make_readout_executor(p0=0.0, p1=0.0)
noisy_executor = make_readout_executor(p0=P0, p1=P1)
```

## Compare ideal, noisy, and REM-mitigated results

We first evaluate the ideal and noisy expectation values, then apply REM using an inverse confusion matrix generated with {func}`mitiq.rem.generate_inverse_confusion_matrix`.

```{code-cell} ipython3
ideal = raw_execute(circuit, ideal_executor, observable)
noisy = raw_execute(circuit, noisy_executor, observable)

inverse_confusion_matrix = rem.generate_inverse_confusion_matrix(
    1, p0=P0, p1=P1
)
mitigated = rem.execute_with_rem(
    circuit,
    noisy_executor,
    observable,
    inverse_confusion_matrix=inverse_confusion_matrix,
)

print(f"Ideal <Z>:      {ideal.real:.5f}")
print(f"Noisy <Z>:      {noisy.real:.5f}")
print(f"REM <Z>:        {mitigated.real:.5f}")
print(f"Error (noisy):  {abs((ideal - noisy) / ideal).real:.3f}")
print(f"Error (REM):    {abs((ideal - mitigated) / ideal).real:.3f}")
```

REM recovers the ideal expectation value in this example because the simulated readout noise matches the confusion model used to build the inverse matrix.

More options for generating and applying confusion matrices are described in the [REM user guide](../guide/rem.md).
