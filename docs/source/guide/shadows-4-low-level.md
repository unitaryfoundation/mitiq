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

# What happens when I use classical shadows?

```{include} shadows-experimental-note.md
```

```{code-cell} ipython3
:tags: [remove-cell]
from functools import partialmethod
import tqdm
tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)
```

The high-level functions `shadows.shadow_quantum_processing` and
`shadows.classical_post_processing` each compose several lower-level steps.
This page walks through those steps explicitly, using functions from
`mitiq.experimental.shadows.quantum_processing` and `mitiq.experimental.shadows.classical_postprocessing`.

```{figure} ../img/classicalshadow_workflow.png
---
width: 700px
name: shadows-workflow-low-level
---
Workflow of the classical shadow estimation protocol in Mitiq.
```

## Problem setup

```{code-cell} ipython3
import cirq
import numpy as np
from mitiq import MeasurementResult, PauliString
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)

qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
)

def execute(circuit: cirq.Circuit) -> MeasurementResult:
    return cirq_sample_bitstrings(circuit, noise_level=(0,), shots=1)
```

## Step 1: Quantum processing

### Sample random Pauli bases

The first step is to choose, for each measurement round, a random Pauli basis for
each qubit.
`sample_random_pauli_bases` draws uniformly from $\{X, Y, Z\}$ per qubit:

```{code-cell} ipython3
from mitiq.experimental.shadows.quantum_processing import sample_random_pauli_bases

num_snapshots = 300
num_qubits = len(qubits)

pauli_bases = sample_random_pauli_bases(num_qubits, num_snapshots)
print("First five bases:", pauli_bases[:5])
```

### Rotate circuits for each basis

`get_rotated_circuits` appends single-qubit Clifford gates and measurement gates to
the original circuit so that measuring in the $Z$ basis is equivalent to measuring in
the chosen Pauli basis:

```{code-cell} ipython3
from mitiq.experimental.shadows.quantum_processing import get_rotated_circuits

rotated_circuits = get_rotated_circuits(circuit, pauli_bases, qubits=qubits)
print(f"Example Pauli basis: {pauli_bases[0]}")
print("Example rotated circuit:")
print(rotated_circuits[0])
```

### Execute the circuits

Each rotated circuit is executed once to obtain a single bitstring.
The results together form the raw shadow data:

```{code-cell} ipython3
results = [execute(circ) for circ in rotated_circuits]
bitstrings = [list(result.get_counts().keys())[0] for result in results]
print("First five bitstrings:", bitstrings[:5])
```

The `(bitstrings, pauli_bases)` tuple is exactly what `shadow_quantum_processing`
returns and what `classical_post_processing` expects as its `shadow_outcomes` argument.

## Step 2: Classical post-processing

### Reconstruct individual snapshots

Each `(bitstring, paulistring)` pair defines a single classical snapshot — an estimate
of the quantum state from one measurement.
`classical_snapshot` applies the inverse channel $\mathcal{M}^{-1}$ to convert the
measurement outcome into a $2^n \times 2^n$ matrix:

```{code-cell} ipython3
from mitiq.experimental.shadows.classical_postprocessing import classical_snapshot

snapshot = classical_snapshot(bitstrings[0], pauli_bases[0])
print("Snapshot shape:", snapshot.shape)
print("Snapshot (real part):\n", np.round(np.real(snapshot), 2))
```

### Reconstruct the density matrix

`shadow_state_reconstruction` averages all snapshots to obtain an estimate of the
full density matrix $\rho$:

```{code-cell} ipython3
from mitiq.experimental.shadows.classical_postprocessing import shadow_state_reconstruction

shadow_outcomes = (bitstrings, pauli_bases)

rho_shadow = shadow_state_reconstruction(shadow_outcomes)
print("Reconstructed density matrix shape:", rho_shadow.shape)
print("Trace:", np.round(np.real(np.trace(rho_shadow)), 3))
```

### Estimate expectation values

`expectation_estimation_shadow` estimates the expectation value of a single observable
using the median-of-means estimator.
Only snapshots measured in a basis that matches the observable's Pauli support
contribute:

```{code-cell} ipython3
from mitiq.experimental.shadows.classical_postprocessing import expectation_estimation_shadow

observable = PauliString("ZZ", support=(0, 1), coeff=1)

estimate = expectation_estimation_shadow(
    shadow_outcomes,
    observable,
    num_batches=1,
)
print(f"Estimated ⟨ZZ⟩ on qubits (0,1): {estimate:.4f}")
```

## Step 3: Robust shadow estimation (low level)

The robust variant additionally estimates Pauli fidelities from measurements on the
$|0\rangle^{\otimes n}$ state, then uses those fidelities to calibrate the inverse
channel.

### Estimate Pauli fidelities for a single shot

`get_single_shot_pauli_fidelity` computes the Pauli fidelity contribution from a
single calibration measurement:

```{code-cell} ipython3
from mitiq.experimental.shadows.classical_postprocessing import get_single_shot_pauli_fidelity

single_fidelities = get_single_shot_pauli_fidelity(
    bitstring="000",
    paulistring="XYZ",
    locality=2,
)
print("Single-shot Pauli fidelities (locality=2):", single_fidelities)
```

### Estimate Pauli fidelities from calibration data

`get_pauli_fidelities` aggregates single-shot fidelities over many calibration
measurements using the median-of-means estimator:

```{code-cell} ipython3
from mitiq.experimental.shadows.classical_postprocessing import get_pauli_fidelities

def noisy_execute(circuit: cirq.Circuit) -> MeasurementResult:
    *operations, measurement = circuit
    noise = cirq.Moment(cirq.depolarize(0.2).on_each(*circuit.all_qubits()))
    noisy_circuit = cirq.Circuit(*operations, noise, measurement)
    return cirq_sample_bitstrings(noisy_circuit, noise_level=(0,), shots=1)

# Collect calibration measurements on the zero state
zero_circuit = cirq.Circuit()
cal_pauli_bases = sample_random_pauli_bases(num_qubits, 3000)
cal_rotated = get_rotated_circuits(zero_circuit, cal_pauli_bases, qubits=qubits)
cal_results = [noisy_execute(circ) for circ in cal_rotated]
cal_bitstrings = [list(r.get_counts().keys())[0] for r in cal_results]

calibration_outcomes = (cal_bitstrings, cal_pauli_bases)
fidelities = get_pauli_fidelities(calibration_outcomes, num_batches=2, locality=2)
print("Estimated Pauli fidelities:", fidelities)
```

### Apply calibration to state reconstruction and expectation values

The estimated `fidelities` dictionary can be passed directly to `classical_snapshot`,
`shadow_state_reconstruction`, and `expectation_estimation_shadow` via their
`fidelities` argument.
This replaces the ideal inverse channel $\mathcal{M}^{-1}$ with the calibrated
inverse channel $\widehat{\mathcal{M}}^{-1}$:

```{code-cell} ipython3
# Collect shadow measurements with the noisy executor
noisy_pauli_bases = sample_random_pauli_bases(num_qubits, num_snapshots)
noisy_rotated = get_rotated_circuits(circuit, noisy_pauli_bases, qubits=qubits)
noisy_results = [noisy_execute(circ) for circ in noisy_rotated]
noisy_bitstrings = [list(r.get_counts().keys())[0] for r in noisy_results]
noisy_outcomes = (noisy_bitstrings, noisy_pauli_bases)

# Uncalibrated estimate
uncalibrated = expectation_estimation_shadow(noisy_outcomes, observable, num_batches=1)

# Calibrated estimate using the estimated fidelities
calibrated = expectation_estimation_shadow(
    noisy_outcomes, observable, num_batches=1, fidelities=fidelities
)

sim_result = cirq.Simulator().simulate(circuit)
ideal_zz = np.real(
    (cirq.Z(qubits[0]) * cirq.Z(qubits[1])).expectation_from_state_vector(
        sim_result.final_state_vector,
        qubit_map={q: i for i, q in enumerate(qubits)},
    )
)

print(f"Ideal ⟨ZZ⟩:         {ideal_zz:.4f}")
print(f"Uncalibrated ⟨ZZ⟩: {uncalibrated:.4f}")
print(f"Calibrated ⟨ZZ⟩:   {calibrated:.4f}")
```

The high-level function `mitiq.experimental.shadows.pauli_twirling_calibrate` performs the calibration
steps above automatically, and `mitiq.experimental.shadows.classical_post_processing` with
`calibration_results` passes the fidelities through to these low-level functions.
