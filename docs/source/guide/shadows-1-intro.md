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

# How Do I Use Classical Shadows?

```{include} shadows-experimental-note.md
```

The `mitiq.experimental.shadows` module estimates expectation values and reconstructs density matrices from random Pauli measurements.
It also supports robust shadow estimation, which calibrates out noise on the rotation gates and measurements.
For the theory behind the technique, see [](shadows-5-theory.md).

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

## Problem setup

Define a circuit of interest. Here we use a 3-qubit GHZ state:

```{code-cell} ipython3
qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
)
print(circuit)
```

The classical shadows protocol requires an executor that returns a **single-shot** `MeasurementResult` — one bitstring per call:

```{code-cell} ipython3
def execute(circuit: cirq.Circuit) -> MeasurementResult:
    return cirq_sample_bitstrings(circuit, noise_level=(0,), shots=1)
```

## Apply classical shadows

`shadow_quantum_processing` runs the circuit with random Pauli rotations and collects measurement outcomes.
`classical_post_processing` uses those outcomes to estimate expectation values or reconstruct the density matrix.

```{code-cell} ipython3
shadow_outcomes = shadows.shadow_quantum_processing(
    circuit, execute, num_total_measurements_shadow=500
)
```

To estimate expectation values, pass a list of `PauliString` observables:

```{code-cell} ipython3
observables = [
    PauliString("ZZ", support=(i, i + 1), coeff=1)
    for i in range(len(qubits) - 1)
]

result = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    observables=observables,
    k_shadows=1,
)
print("Shadow estimates:", result)
```

To reconstruct the full density matrix instead, use `state_reconstruction=True`:

```{code-cell} ipython3
result = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    state_reconstruction=True,
)
print("Reconstructed state shape:", result["reconstructed_state"].shape)
```

For a detailed walkthrough with plots and error analysis, see [Classical Shadows Protocol with Cirq](../examples/shadows_tutorial.md).

## Apply robust shadow estimation

Robust shadow estimation calibrates out noise that acts on the Pauli rotation gates and the measurement.
It requires an additional calibration step using measurements on the $|0\rangle^{\otimes n}$ state.

Define a noisy executor that inserts a layer of depolarizing noise immediately before the measurement:

```{code-cell} ipython3
def noisy_execute(circuit: cirq.Circuit) -> MeasurementResult:
    *operations, measurement = circuit
    noise = cirq.Moment(cirq.depolarize(0.2).on_each(*circuit.all_qubits()))
    noisy_circuit = cirq.Circuit(*operations, noise, measurement)
    return cirq_sample_bitstrings(noisy_circuit, noise_level=(0,), shots=1)
```

```{note}
This is a demonstration noise model, deliberately constructed so that robust shadow estimation can remove the noise completely: all of the noise is confined to the measurement stage, and state preparation is noiseless.
On real hardware state preparation is also noisy, and robust shadow estimation does not correct that portion.
```

Run `pauli_twirling_calibrate` to characterize the noise channel.
The `locality` parameter should match the weight of the heaviest observable you plan to estimate:

```{code-cell} ipython3
calibration_results = shadows.pauli_twirling_calibrate(
    k_calibration=1,
    locality=2,
    qubits=qubits,
    executor=noisy_execute,
    num_total_measurements_calibration=5000,
)
```

Then run the shadow protocol with the noisy executor and pass `calibration_results` to post-processing:

```{code-cell} ipython3
shadow_outcomes = shadows.shadow_quantum_processing(
    circuit, noisy_execute, num_total_measurements_shadow=5000
)

uncalibrated = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    observables=observables,
    k_shadows=1,
)
calibrated = shadows.classical_post_processing(
    shadow_outcomes=shadow_outcomes,
    calibration_results=calibration_results,
    observables=observables,
    k_shadows=1,
)
print("Uncalibrated:", uncalibrated)
print("Calibrated:  ", calibrated)
```

```{note}
You do not need to re-run `pauli_twirling_calibrate` between experiments as long as the number of qubits and the noise channel have not changed, and the new observables act on no more qubits than the calibrated `locality`.
Reusing a calibration for an observable of larger weight raises a `ValueError`, since no Pauli fidelity was estimated for its support.
```

For a detailed walkthrough of robust shadow estimation, see [Robust Shadow Estimation with Mitiq](../examples/rshadows_tutorial.md).
