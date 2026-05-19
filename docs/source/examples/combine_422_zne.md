---
jupytext:
  formats: md:myst
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
```{tags} rem, zne, cirq, intermediate
```

# Composing techniques: [[4,2,2]] Error Detection and Zero Noise Extrapolation

Noise in quantum computers can arise from a variety of sources, and sometimes applying multiple error suppression strategies can be more beneficial than applying a single technique alone.

Here we combine the **[[4,2,2]] quantum error-detecting code** with **Zero Noise Extrapolation (ZNE)** on a logical circuit.
The [[4,2,2]] code encodes 2 logical qubits into 4 physical qubits and can detect (but not correct) any single-qubit error by measuring two stabilizer generators, $XXXX$ and $ZZZZ$.
Any single-qubit error anticommutes with at least one stabilizer, flipping its outcome from $+1$ to $-1$ and signalling that the shot should be discarded.

The four logical codewords are:
```{math}
\begin{align}
|0_L 0_L\rangle &= \tfrac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle) \\
|0_L 1_L\rangle &= \tfrac{1}{\sqrt{2}}(|0011\rangle + |1100\rangle) \\
|1_L 0_L\rangle &= \tfrac{1}{\sqrt{2}}(|0101\rangle + |1010\rangle) \\
|1_L 1_L\rangle &= \tfrac{1}{\sqrt{2}}(|0110\rangle + |1001\rangle)
\end{align}
```


A useful reference for the encoding circuit and stabilizer structure is the [[4,2,2]] section of the [Quantum Error Correction Zoo](https://errorcorrectionzoo.org/c/stab_4_2_2).
After post-selecting the surviving shots with {func}`mitiq.rem.post_select`, we apply [ZNE](../guide/zne.md) to further suppress the residual noise that escaped detection.

In [ZNE](../guide/zne.md), the expectation value of the observable of interest is computed at different noise levels, and subsequently the ideal expectation value is inferred by extrapolating the measured results to the zero-noise limit.
More information on ZNE can be found in the [corresponding section of the user guide](../guide/zne.md).

This approach is motivated by *Zhong et al. arXiv (2025)* {cite}`Zhong_2025_arxiv_422` ([arXiv:2510.01181](https://arxiv.org/abs/2510.01181)), which develops a hybrid error suppression protocol combining error-detecting codes with error mitigation.

+++

## Setup

We begin by importing the relevant modules and libraries required for the rest of this tutorial.

```{code-cell} ipython3
import cirq
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from mitiq import MeasurementResult
from mitiq.rem import post_select
from mitiq import zne
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global
```

## Task

We will demonstrate error detection followed by ZNE on a logical state prepared with the [[4,2,2]] code.
We use 6 qubits in total: `q[0]`–`q[3]` are the data qubits, `q[4]` is the ancilla for the $XXXX$ stabilizer, and `q[5]` is the ancilla for the $ZZZZ$ stabilizer.

```{code-cell} ipython3
# q[0..3] = data qubits, q[4] = XXXX ancilla, q[5] = ZZZZ ancilla
q = cirq.LineQubit.range(6)
```

We prepare the logical state $|0_L 0_L\rangle = \tfrac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$ with a Hadamard gate on `q[0]` followed by three CNOT gates, then apply a transversal logical $\bar{X}\bar{X}$ gate (bitwise X on all four data qubits) to map the state to $|1_L 1_L\rangle$.

```{code-cell} ipython3
def build_circuit() -> cirq.Circuit:
    """Prepare |0_L 0_L>, apply logical X_L X_L, then measure both stabilizers."""
    circuit = cirq.Circuit()

    # Encode |0_L 0_L> = (|0000> + |1111>) / sqrt(2)
    circuit.append(cirq.H(q[0]))
    circuit.append([cirq.CNOT(q[0], q[i]) for i in range(1, 4)])

    # Logical X_L X_L: transversal X on all data qubits -> |1_L 1_L>
    circuit.append([cirq.X(q[i]) for i in range(4)])

    # XXXX stabilizer via ancilla q[4]
    circuit.append(cirq.H(q[4]))
    circuit.append([cirq.CNOT(q[4], q[i]) for i in range(4)])
    circuit.append(cirq.H(q[4]))

    # ZZZZ stabilizer via ancilla q[5]
    circuit.append([cirq.CNOT(q[i], q[5]) for i in range(4)])

    # Measure syndrome ancillae then data qubits
    circuit.append(cirq.measure(q[4], q[5], key="syndromes"))
    circuit.append(cirq.measure(*q[:4], key="data"))
    return circuit

circuit = build_circuit()
print(circuit)
```

## Noise model and executor

The noise in this example is a local depolarizing channel applied after every gate.
A depolarizing probability of 1% per gate is a realistic and illustrative value for near-term superconducting devices.
We use an [executor function](../guide/executors.md) to run the quantum circuit with the noise model applied.

The executor returns a `MeasurementResult` whose bitstrings contain **6 bits each**: bits 0–3 are the data qubits and bits 4–5 are the syndrome ancillae ($XXXX$, $ZZZZ$ respectively).

```{code-cell} ipython3
def execute(
    circuit: cirq.Circuit,
    noise_level: float = 0.01,
    repetitions: int = 5000,
) -> MeasurementResult:
    """Execute a circuit with depolarizing noise of strength ``noise_level``."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(noise_level))
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(noisy_circuit, repetitions=repetitions)

    data_bits     = result.measurements["data"]      # shape (reps, 4)
    syndrome_bits = result.measurements["syndromes"] # shape (reps, 2)
    # Concatenate: [data | syndromes] so bits[4] and bits[5] are the syndrome qubits
    combined = np.concatenate([data_bits, syndrome_bits], axis=1)
    return MeasurementResult(combined)
```

## Observable

In this example, the observable of interest is $\langle ZZZZ \rangle$ on the four data qubits.
After the logical $\bar{X}\bar{X}$ gate, the ideal state is $|1_L 1_L\rangle$ and the ideal expectation value of $ZZZZ$ is $+1$.

```{code-cell} ipython3
def expect_zzzz(mr: MeasurementResult) -> float:
    """Compute <ZZZZ> from data bits (indices 0–3) of a MeasurementResult."""
    data = mr.filter_qubits([0, 1, 2, 3])   # shape (shots, 4)
    z_vals = 1 - 2 * data                   # +1 if bit=0, -1 if bit=1
    return float(np.mean(np.prod(z_vals, axis=1)))
```

For the circuit defined above, the ideal (noiseless) expectation value is $+1$:

```{code-cell} ipython3
ideal_mr = execute(circuit, noise_level=0.0, repetitions=2000)
ideal    = expect_zzzz(ideal_mr)
print("Ideal value:", "{:.5f}".format(ideal))
```

The unmitigated (noisy) result is substantially degraded by the depolarizing errors:

```{code-cell} ipython3
noisy_mr = execute(circuit, noise_level=0.01, repetitions=5000)
noisy    = expect_zzzz(noisy_mr)
print("Unmitigated value:", "{:.5f}".format(noisy))
```

### Applying error detection with post-selection

We now discard shots where an error was detected.
A syndrome outcome of 0 on `q[4]` (resp. `q[5]`) means the $XXXX$ (resp. $ZZZZ$) stabilizer measured $+1$, i.e. no X-type (Z-type) error was flagged.
We keep only shots satisfying both conditions using {func}`mitiq.rem.post_select`:

```{note}
The [[4,2,2]] code detects errors but cannot correct them — shots where an error is detected are simply discarded. This reduces the number of usable shots, which is a real overhead cost. To achieve a target number of *clean* shots, you must collect significantly more *raw* shots than you would without error detection.
```

```{code-cell} ipython3
# Keep only shots where both syndrome qubits are 0 (no error detected)
ps_result = post_select(noisy_mr, lambda bits: bits[4] == 0 and bits[5] == 0)

retained_pct = 100 * ps_result.shots / noisy_mr.shots
print(f"Shots retained after post-selection: {ps_result.shots} / {noisy_mr.shots} ({retained_pct:.1f}%)")
print("Mitigated value obtained with error detection:", "{:.5f}".format(expect_zzzz(ps_result)))
```

We can see that error detection improves the results, but errors that escaped detection (two-qubit correlated errors commuting with both stabilizers) still remain.

### Zero Noise Extrapolation alone

For comparison, we apply ZNE on its own, without post-selection.

```{code-cell} ipython3
def execute_scalar(circuit, noise_level=0.01, repetitions=5000):
    """Execute and return <ZZZZ> directly (no post-selection)."""
    return expect_zzzz(execute(circuit, noise_level=noise_level, repetitions=repetitions))

zne_executor = zne.mitigate_executor(
    execute_scalar,
    scale_noise=fold_global,
    factory=LinearFactory([1, 2, 3]),
)
zne_result = zne_executor(circuit)
print("Mitigated value obtained with ZNE:", "{:.5f}".format(zne_result))
```

### Error detection + Zero Noise Extrapolation

Finally, we apply a combination of error detection and ZNE.
Post-selection is applied first inside the executor so that at each noise-scaled circuit variant ZNE only sees shots that passed the syndrome check.

```{code-cell} ipython3
def execute_with_postselect(circuit, noise_level=0.01, repetitions=5000):
    """Execute, post-select syndrome-passing shots, then return <ZZZZ>."""
    mr = execute(circuit, noise_level=noise_level, repetitions=repetitions)
    mr_clean = post_select(mr, lambda bits: bits[4] == 0 and bits[5] == 0)
    if mr_clean.shots == 0:
        return 0.0
    return expect_zzzz(mr_clean)

combined_executor = zne.mitigate_executor(
    execute_with_postselect,
    scale_noise=fold_global,
    factory=LinearFactory([1, 2, 3]),
)
combined_result = combined_executor(circuit)
print("Mitigated value obtained with error detection + ZNE:", "{:.5f}".format(combined_result))

# Calculate and show the additional improvement
ps_error = abs(ideal - expect_zzzz(ps_result))
combined_error = abs(ideal - combined_result)
improvement = 100 * (1 - combined_error / ps_error)
print(f"Additional error reduction from ZNE: {improvement:.1f}%")
```

```{code-cell} ipython3
:tags: [hide-input]

labels = ["Ideal", "Noisy\n(unmitigated)", "Error detection\n(post-select only)", "ZNE only", "Error detection\n+ ZNE"]
values = [ideal, noisy, expect_zzzz(ps_result), zne_result, combined_result]
colors = ["#4CAF50", "#E53935", "#FB8C00", "#7B1FA2", "#1565C0"]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, alpha=0.9)
ax.axhline(y=ideal, color="#4CAF50", linestyle="--", linewidth=1.4, label="Ideal value")
ax.set_ylabel("⟨ZZZZ⟩ expectation value")
ax.set_title("[[4,2,2]] Error Detection + ZNE (depolarizing noise p = 0.01)")
ax.set_ylim(min(values) - 0.15, ideal + 0.15)
ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:+.3f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )
plt.tight_layout()
plt.show()
```

From this example we can see that each technique affords some improvement, and the combination of [[4,2,2]] error detection and ZNE is more effective in suppressing errors than either technique applied alone.

We encourage users to experiment with different noise levels and scale factor sets to explore the trade-off between shot overhead (from post-selection) and the accuracy gains of the combined approach.


