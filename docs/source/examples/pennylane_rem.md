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

# Readout error mitigation in a PennyLane workflow

Mitiq already has a PennyLane tutorial demonstrating [zero-noise extrapolation (ZNE)](pennylane-ibmq-backends.md) via `pennylane.mitigate_with_zne`.
This example complements it by showing how to use [readout error mitigation (REM)](../guide/rem.md) with a circuit that is defined *and executed* entirely in PennyLane.

Instead of converting to another framework, we keep the whole workflow in PennyLane and go one step beyond the [REM user guide](../guide/rem.md) in two ways:

1. We model **asymmetric, per-qubit** readout errors, so that flipping a measured `0` to `1` is not equally likely as flipping a `1` to `0`, and the two qubits are miscalibrated differently.
2. We **estimate the confusion matrix from calibration data** rather than assuming we already know the error rates, which is what one has to do on real hardware.

## Setup

```{code-cell} ipython3
import numpy as np
import pennylane as qml

from mitiq import MeasurementResult
from mitiq.rem import (
    generate_tensored_inverse_confusion_matrix,
    mitigate_executor,
)

rng = np.random.default_rng(1967)
```

## Define a PennyLane circuit

We use a two-qubit GHZ state, whose ideal measurement distribution is an equal mix of `00` and `11`.
The device returns raw computational-basis samples via `qml.sample`, which is exactly the kind of measurement data REM operates on.

```{code-cell} ipython3
n_wires = 2
shots = 40_000

dev = qml.device("default.qubit", wires=n_wires, shots=shots, seed=rng)


@qml.qnode(dev)
def ghz_state():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.sample()
```

## Model asymmetric readout error

Readout error acts on the *classical* measurement outcomes: after the quantum circuit runs, each measured bit can be reported incorrectly.
We model it directly on the PennyLane samples with per-qubit rates:

- `p0[w]`: probability of reading `1` when qubit `w` was actually `0`,
- `p1[w]`: probability of reading `0` when qubit `w` was actually `1`.

The rates below are asymmetric (`p0 != p1`) and differ between the two qubits.

```{code-cell} ipython3
p0 = np.array([0.05, 0.10])
p1 = np.array([0.20, 0.15])


def apply_readout_error(samples: np.ndarray) -> np.ndarray:
    samples = np.atleast_2d(np.asarray(samples, dtype=int))
    noisy = samples.copy()
    for w in range(n_wires):
        draws = rng.random(samples.shape[0])
        noisy[(samples[:, w] == 0) & (draws < p0[w]), w] = 1
        noisy[(samples[:, w] == 1) & (draws < p1[w]), w] = 0
    return noisy
```

REM requires an executor that returns raw measurement results as a {class}`mitiq.MeasurementResult`.
The `-> MeasurementResult` return annotation is important: Mitiq uses it to route the results through the REM post-processing step.

```{code-cell} ipython3
def readout_executor(circuit: qml.QNode) -> MeasurementResult:
    clean_samples = circuit()
    noisy_samples = apply_readout_error(clean_samples)
    return MeasurementResult(
        noisy_samples, qubit_indices=tuple(range(n_wires))
    )
```

## Ideal and noisy distributions

We first look at the noiseless distribution and the distribution obtained through our noisy readout.

```{code-cell} ipython3
def distribution(result: MeasurementResult) -> dict[str, float]:
    dist = result.prob_distribution()
    return {state: round(dist.get(state, 0.0), 3) for state in ("00", "01", "10", "11")}


ideal_result = MeasurementResult(
    np.atleast_2d(ghz_state()), qubit_indices=tuple(range(n_wires))
)
noisy_result = readout_executor(ghz_state)

print("Ideal distribution:", distribution(ideal_result))
print("Noisy distribution:", distribution(noisy_result))
```

The asymmetric readout error leaks probability into the `01` and `10` outcomes, and the two qubits are affected differently.

## Estimate the confusion matrix

On hardware the error rates are unknown, so we estimate them with calibration circuits: prepare a known input, measure it through the *same* noisy readout, and read off how often each qubit is misreported.

```{code-cell} ipython3
@qml.qnode(dev)
def prepare_zeros():
    for w in range(n_wires):
        qml.Identity(wires=w)
    return qml.sample()


@qml.qnode(dev)
def prepare_ones():
    for w in range(n_wires):
        qml.PauliX(wires=w)
    return qml.sample()


measured_zeros = readout_executor(prepare_zeros).asarray
measured_ones = readout_executor(prepare_ones).asarray

# Fraction of 1s given a true 0, and fraction of 0s given a true 1.
est_p0 = measured_zeros.mean(axis=0)
est_p1 = 1 - measured_ones.mean(axis=0)

print("Estimated p0:", np.round(est_p0, 3), "(true:", p0, ")")
print("Estimated p1:", np.round(est_p1, 3), "(true:", p1, ")")
```

Each qubit gets its own confusion matrix, whose columns are the true state and whose rows are the measured state.

```{code-cell} ipython3
confusion_matrices = [
    np.array(
        [
            [1 - est_p0[w], est_p1[w]],
            [est_p0[w], 1 - est_p1[w]],
        ]
    )
    for w in range(n_wires)
]

for w, cm in enumerate(confusion_matrices):
    print(f"Confusion matrix for qubit {w}:\n{np.round(cm, 3)}")
```

## Apply REM

We tensor the per-qubit confusion matrices into a single inverse confusion matrix with {func}`mitiq.rem.generate_tensored_inverse_confusion_matrix`, then wrap the noisy executor with {func}`mitiq.rem.mitigate_executor`.

```{code-cell} ipython3
inverse_confusion_matrix = generate_tensored_inverse_confusion_matrix(
    n_wires, confusion_matrices
)

mitigated_executor = mitigate_executor(
    readout_executor,
    inverse_confusion_matrix=inverse_confusion_matrix,
)
mitigated_result = mitigated_executor(ghz_state)

print("Ideal distribution:    ", distribution(ideal_result))
print("Noisy distribution:    ", distribution(noisy_result))
print("REM distribution:      ", distribution(mitigated_result))
```

Even though the confusion matrix was *estimated* rather than assumed, and the readout error is asymmetric and qubit-dependent, REM removes most of the spurious `01` and `10` population and restores a distribution close to the ideal `00`/`11` mix.

More options for generating and applying confusion matrices are described in the [REM user guide](../guide/rem.md).
