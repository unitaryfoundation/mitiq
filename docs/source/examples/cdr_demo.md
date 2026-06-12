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

```{tags} cdr, cirq, intermediate
```

# Clifford Data Regression (CDR) with Cirq

This notebook demonstrates Clifford Data Regression (CDR) {cite}`Czarnik_2021_Quantum`, a
learning-based error mitigation technique that trains a noise correction model directly from
the quantum device being used.

CDR differs from [Zero Noise Extrapolation](../guide/zne.md) (ZNE) in an important way:
instead of amplifying noise to extrapolate to zero, CDR *learns* the relationship between
noisy and exact expectation values from a set of near-Clifford *training circuits* that closely
resemble the target circuit but can be efficiently simulated classically.
This makes CDR a good choice when the noise model is unknown or hard to characterize, as no
noise model is required — only a classical simulator.

For a detailed description of the method and its parameters, see the
[CDR user guide](../guide/cdr.md).

+++

## Setup

We begin by importing the relevant modules and libraries that we will require
for the rest of this tutorial.

```{code-cell} ipython3
import warnings

# Plotting imports.
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 15})
%matplotlib inline

# Third-party imports.
import cirq
import numpy as np

# Mitiq imports.
from mitiq import cdr, Observable, PauliString
from mitiq.interface.mitiq_cirq import compute_density_matrix

warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
```

## Define parameters

```{code-cell} ipython3
# Random seed for reproducibility.
seed: int = 1

# Depolarizing noise level applied to each qubit per moment.
noise_level: float = 0.02

# Number of near-Clifford training circuits used for regression.
num_training_circuits: int = 20

# Fraction of non-Clifford gates retained in each training circuit.
fraction_non_clifford: float = 0.1
```

## Define the circuit

CDR requires the circuit to be compiled so that all non-Clifford gates are single-qubit
$R_Z(\theta)$ rotations.
We build a four-qubit hardware-efficient variational ansatz that already satisfies this
constraint: each layer applies parameterized $R_Z$ rotations to every qubit (preceded by
Hadamard gates to prepare a useful starting state) followed by a chain of CNOT gates that
entangle neighboring qubits.

```{code-cell} ipython3
qubits = cirq.LineQubit.range(4)


def ansatz_layer(angles: list[float]) -> list[cirq.Operation]:
    """Returns one layer of the variational ansatz.

    Each layer consists of single-qubit H + Rz rotations followed
    by a CNOT chain entangling neighboring qubits.
    """
    ops = []
    for qubit, theta in zip(qubits, angles):
        ops += [cirq.H(qubit), cirq.rz(theta)(qubit)]
    for i in range(len(qubits) - 1):
        ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return ops


circuit = cirq.Circuit(
    ansatz_layer([0.4, 1.1, -0.7, 0.9]),
    ansatz_layer([-0.5, 0.8, 1.3, -0.6]),
    ansatz_layer([1.2, -0.3, 0.6, -1.1]),
)

print(circuit)
```

The $R_Z$ rotations with arbitrary angles are non-Clifford gates — CDR will use them
to construct near-Clifford training circuits.
We can confirm the circuit contains the expected 12 non-Clifford $R_Z$ gates
(4 qubits × 3 layers).

```{code-cell} ipython3
non_clifford_ops = [
    op
    for op in circuit.all_operations()
    if isinstance(op.gate, cirq.ZPowGate)
    and not np.isclose(float(op.gate.exponent) % 1, 0)
    and not np.isclose(float(op.gate.exponent) % 0.5, 0)
]
print(f"Number of non-Clifford Rz gates: {len(non_clifford_ops)}")
```

## Define the observable

We estimate the expectation value of a two-term Hamiltonian:

$$
O = Z_0 Z_1 + 0.5 \, X_2 X_3.
$$

```{code-cell} ipython3
obs = Observable(PauliString("ZZII"), PauliString("IIXX", coeff=0.5))
print(obs)
```

## Define the executors

CDR requires two executor functions:

- A **noisy executor** that runs on the actual device (or a noisy simulator).
- A **near-Clifford simulator** used to label the training circuits with their ideal
  expectation values.

Here we use Cirq's density matrix simulator for both.
The noisy executor applies single-qubit depolarizing noise after each moment, while the
simulator runs without any noise.

```{code-cell} ipython3
def ideal_executor(circuit: cirq.Circuit) -> np.ndarray:
    """Simulates the circuit without noise."""
    return compute_density_matrix(circuit, noise_level=(0.0,))


def noisy_executor(circuit: cirq.Circuit) -> np.ndarray:
    """Simulates the circuit with depolarizing noise."""
    return compute_density_matrix(circuit, noise_level=(noise_level,))
```

```{note}
For larger circuits, replace `ideal_executor` with an efficient near-Clifford simulator such
as [Qrack](https://github.com/unitaryfoundation/qrack), whose stabilizer-hybrid mode runs in
time that scales with the number of non-Clifford gates rather than the number of qubits.
See [CDR with Qrack as Near-Clifford Simulator](./cdr_qrack.md) for an example.
```

## Baseline: ideal and noisy values

Before applying CDR, we establish the ideal (noiseless) and unmitigated (noisy)
expectation values.
This lets us measure how much CDR reduces the error.

```{code-cell} ipython3
ideal_value = obs.expectation(circuit, ideal_executor).real
noisy_value = obs.expectation(circuit, noisy_executor).real

print(f"Ideal expectation value:       {ideal_value:.4f}")
print(f"Unmitigated expectation value: {noisy_value:.4f}")
```

The gap between the two is the error we are trying to correct.

## Running CDR

We call {func}`.cdr.execute_with_cdr` with four required arguments and a few optional
parameters to control the training procedure.

| Argument | Role |
|---|---|
| `circuit` | The target circuit whose expectation value we want to mitigate. |
| `executor` | Runs circuits on the noisy device (or noisy simulator). |
| `observable` | The Hermitian operator whose expectation value to compute. |
| `simulator` | Runs training circuits on a noiseless near-Clifford simulator. |
| `num_training_circuits` | How many near-Clifford training circuits to generate. |
| `fraction_non_clifford` | Fraction of non-Clifford gates kept in each training circuit. |

CDR generates `num_training_circuits` near-Clifford approximations of the target circuit by
randomly replacing `1 - fraction_non_clifford` of the $R_Z$ rotations with nearby Clifford
gates.
Each training circuit is run on both the noisy executor and the noiseless simulator, giving
a set of (noisy, ideal) pairs.
A linear model is then fit to those pairs, and the resulting correction is applied to the
target circuit's noisy result.

```{code-cell} ipython3
cdr_value = cdr.execute_with_cdr(
    circuit,
    noisy_executor,
    observable=obs,
    simulator=ideal_executor,
    num_training_circuits=num_training_circuits,
    fraction_non_clifford=fraction_non_clifford,
    random_state=seed,
).real

print(f"CDR-mitigated expectation value: {cdr_value:.4f}")
```

## Results

```{code-cell} ipython3
noisy_error = abs(noisy_value - ideal_value)
cdr_error = abs(cdr_value - ideal_value)
improvement = noisy_error / cdr_error

print(f"Unmitigated error: {noisy_error:.4f}")
print(f"CDR error:         {cdr_error:.4f}")
print(f"Improvement:       {improvement:.1f}x")
```

```{code-cell} ipython3
labels = ["Ideal", "Noisy", "CDR"]
values = [ideal_value, noisy_value, cdr_value]
colors = ["#4C72B0", "#DD8452", "#55A868"]

plt.figure(figsize=(7, 5))
plt.bar(labels, values, color=colors)
plt.axhline(ideal_value, color="#4C72B0", linestyle="--", alpha=0.6, label="Ideal")
plt.ylabel("Expectation value")
plt.title("CDR reduces the noise-induced error")
plt.legend()
plt.show()
```

CDR moves the noisy estimate noticeably closer to the ideal value, without requiring any
knowledge of the underlying noise model.

## Variable-noise CDR

Variable-noise CDR (vnCDR) {cite}`Lowe_2021_PRR` extends the standard method by also running
the training circuits at scaled noise levels, providing additional data points for the
regression.
This combines the learning aspect of CDR with the noise-scaling idea from ZNE, and often
achieves a better correction than standard CDR alone.

Enabling vnCDR requires only one additional argument: `scale_factors`, a sequence of noise
scale factors to apply.

```{code-cell} ipython3
vncdr_value = cdr.execute_with_cdr(
    circuit,
    noisy_executor,
    observable=obs,
    simulator=ideal_executor,
    num_training_circuits=num_training_circuits,
    fraction_non_clifford=fraction_non_clifford,
    scale_factors=(1.0, 2.0, 3.0),
    random_state=seed,
).real

print(f"vnCDR-mitigated expectation value: {vncdr_value:.4f}")
```

```{code-cell} ipython3
vncdr_error = abs(vncdr_value - ideal_value)

errors = {"Noisy": noisy_error, "CDR": cdr_error, "vnCDR": vncdr_error}

print(f"Unmitigated error: {noisy_error:.4f}")
print(f"CDR error:         {cdr_error:.4f}  ({noisy_error / cdr_error:.1f}x improvement)")
print(f"vnCDR error:       {vncdr_error:.4f}  ({noisy_error / vncdr_error:.1f}x improvement)")
```

```{code-cell} ipython3
err_colors = ["#DD8452", "#55A868", "#C44E52"]

plt.figure(figsize=(7, 5))
bars = plt.bar(errors.keys(), errors.values(), color=err_colors, width=0.5)
for bar, (label, val) in zip(bars, errors.items()):
    annotation = (
        f"{val:.3f}"
        if label == "Noisy"
        else f"{noisy_error / val:.1f}\u00d7"
    )
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.003,
        annotation,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold" if label != "Noisy" else "normal",
    )
plt.ylabel("Absolute error")
plt.title("Error reduction: CDR and vnCDR vs noisy baseline")
plt.show()
```

Both CDR and vnCDR reduce the absolute error relative to the noisy baseline.
On this circuit vnCDR achieves a comparable or better result by leveraging the extra
noise-scaled training data.

For more details on all available options — including custom fit functions and alternative
training circuit generation methods — see the [CDR options guide](../guide/cdr-3-options.md).
If you have any questions, please open a [GitHub discussion](https://github.com/unitaryfoundation/mitiq/discussions)
or reach out on [Discord](http://discord.unitary.foundation).
