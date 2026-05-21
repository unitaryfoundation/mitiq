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

```{tags} qibo, zne, intermediate
```

# Combining error correction and mitigation with Mitiq + Qibo

This tutorial shows a simple hybrid workflow where we combine:

1. an error-correcting code (the 3-qubit repetition code),
2. an error-detection strategy (post-selection on the code space),
3. and error mitigation with [Zero Noise Extrapolation](../guide/zne.md) (ZNE) in Mitiq.

The setup is intentionally lightweight so that the core integration pattern is clear.

## Imports

```{code-cell} ipython3
import os
os.environ["QIBO_LOG_LEVEL"] = "3"  # suppress Qibo INFO logging

import matplotlib.pyplot as plt
import numpy as np
from qibo import Circuit, gates
from mitiq import zne
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global
```

## Build the physical and encoded circuits

We prepare the logical \(|1\rangle\) state in two ways:

- physical: a single qubit in \(|1\rangle\),
- encoded: a 3-qubit repetition encoding \(|1_L\rangle = |111\rangle\).

```{code-cell} ipython3
def make_physical_logical_one():
    circuit = Circuit(1, density_matrix=True)
    circuit.add(gates.X(0))
    return circuit


def make_encoded_logical_one():
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.X(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(0, 2))
    return circuit
```

## Add a simple noise model

For demonstration, we add a depolarizing channel after every non-measurement gate.
We use density-matrix simulation so expectation values are deterministic (no shot noise),
which keeps the tutorial output stable.

```{code-cell} ipython3
def add_depolarizing_noise(circuit, lam=0.05):
    noisy = Circuit(circuit.nqubits, density_matrix=True)
    for gate in circuit.queue:
        noisy.add(gate)
        if isinstance(gate, (gates.M, gates.Channel)):
            continue
        qubits = gate.qubits if len(gate.qubits) > 1 else gate.qubits[0]
        noisy.add(gates.DepolarizingChannel(qubits, lam=lam))
    return noisy
```

## Define executors for three strategies

The observable is the logical \(\langle Z \rangle\), where ideal logical \(|1\rangle\) maps to \(-1\).

1. **Physical baseline**: single-qubit circuit, no code.
2. **Error detection**: encoded circuit with post-selection on code words `000` and `111`.
3. **Error correction**: encoded circuit with majority-vote decoding.

```{code-cell} ipython3
def physical_executor(circuit, lam=0.05):
    probs = add_depolarizing_noise(circuit, lam=lam)().probabilities()
    p1 = probs[1]
    return 1 - 2 * p1


def _detection_observable_and_acceptance(circuit, lam=0.05):
    probs = add_depolarizing_noise(circuit, lam=lam)().probabilities()
    accepted = probs[0] + probs[7]

    if accepted == 0:
        return 0.0, 0.0

    p1 = probs[7] / accepted
    return 1 - 2 * p1, accepted


def detection_executor(circuit, lam=0.05):
    expval, _ = _detection_observable_and_acceptance(circuit, lam=lam)
    return expval


def correction_executor(circuit, lam=0.05):
    probs = add_depolarizing_noise(circuit, lam=lam)().probabilities()
    logical_one = 0.0
    nqubits = circuit.nqubits
    for idx, prob in enumerate(probs):
        if format(idx, f"0{nqubits}b").count("1") >= 2:
            logical_one += prob
    return 1 - 2 * logical_one


def z_to_success_probability(expval_z):
    # For a logical |1> target, p_success = P(logical 1) = (1 - <Z>) / 2.
    return (1 - expval_z) / 2
```

## Run the three strategies with and without ZNE

```{code-cell} ipython3
physical_circuit = make_physical_logical_one()
encoded_circuit = make_encoded_logical_one()

factory_physical = LinearFactory(scale_factors=[1, 2, 3])
factory_detection = LinearFactory(scale_factors=[1, 2, 3])
factory_correction = LinearFactory(scale_factors=[1, 2, 3])

physical_raw = physical_executor(physical_circuit)
physical_zne = zne.execute_with_zne(
    physical_circuit,
    physical_executor,
    factory=factory_physical,
    scale_noise=fold_global,
)

detection_raw, acceptance = _detection_observable_and_acceptance(encoded_circuit)
detection_zne = zne.execute_with_zne(
    encoded_circuit,
    detection_executor,
    factory=factory_detection,
    scale_noise=fold_global,
)

correction_raw = correction_executor(encoded_circuit)
correction_zne = zne.execute_with_zne(
    encoded_circuit,
    correction_executor,
    factory=factory_correction,
    scale_noise=fold_global,
)

print(f"Code-space acceptance rate (detection): {acceptance:.3f}")
```

## Compare logical success probabilities

The extrapolated estimate can be slightly outside the physical \([0, 1]\) interval.
This is a common artifact of extrapolation and is expected in ZNE workflows.

```{code-cell} ipython3
labels = [
    "Physical only",
    "Detection only",
    "Correction only",
]

raw_values = [
    z_to_success_probability(physical_raw),
    z_to_success_probability(detection_raw),
    z_to_success_probability(correction_raw),
]

zne_values = [
    z_to_success_probability(physical_zne),
    z_to_success_probability(detection_zne),
    z_to_success_probability(correction_zne),
]

for label, raw, mitigated in zip(labels, raw_values, zne_values):
    print(
        f"{label:>16}: raw={raw:.4f}, raw+ZNE={mitigated:.4f}, "
        f"delta={mitigated - raw:+.4f}"
    )
```

```{code-cell} ipython3
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width / 2, raw_values, width, label="Raw")
ax.bar(x + width / 2, zne_values, width, label="Raw + ZNE")

ax.set_ylabel("Logical success probability")
ax.set_ylim(0.9, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=10)
ax.set_title("Hybrid QEC + mitigation workflow with Mitiq + Qibo")
ax.legend()
plt.tight_layout()
plt.show()
```

## Optional: visualize one extrapolation fit

```{code-cell} ipython3
factory_correction.plot_fit()
plt.title("ZNE extrapolation fit (correction executor)")
plt.show()
```

In this toy experiment, the exact gains depend on circuit structure and noise strength,
but the key point is that Mitiq integrates naturally with executors that already
include error detection or correction logic.
