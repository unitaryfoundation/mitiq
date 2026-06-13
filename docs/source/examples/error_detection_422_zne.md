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
```{tags} zne, rem, cirq, intermediate
```

# Error detection with the [[4,2,2]] code, combined with ZNE

Error detection and error mitigation are complementary strategies for dealing with noise on
near-term quantum hardware. Error-detecting codes can flag (but not correct) certain physical
errors, allowing us to simply discard the corresponding measurement shots. Error mitigation
techniques such as Zero Noise Extrapolation (ZNE) instead use the *statistics* of many noisy
runs to infer what an ideal, noiseless expectation value would have been.

In this tutorial we combine both approaches:

1. We encode two logical qubits into four physical qubits using the
[[4,2,2]] quantum error-detecting code.
2. We measure the code's two stabilizers using ancilla qubits, and use
{func}`mitiq.rem.post_select` to discard shots where an error was detected.
3. We then apply [ZNE](../guide/zne.md) on top of the post-selected results, recovering
additional accuracy from the surviving shots.

More information on the [[4,2,2]] code, including its stabilizers and logical operators, can be
found in the [Quantum Error Correction Zoo](https://errorcorrectionzoo.org/c/stab_4_2_2). The
idea of combining a small error-detecting code with a software-level mitigation technique such as
ZNE is discussed further in [arXiv:2510.01181](https://arxiv.org/abs/2510.01181).

+++

## The [[4,2,2]] code

The [[4,2,2]] code is the smallest quantum error-*detecting* code. It encodes two logical qubits
into four physical qubits, and has two stabilizer generators, $S_X = XXXX$ and $S_Z = ZZZZ$. Any
single-qubit Pauli error anticommutes with at least one of these stabilizers, flipping the
corresponding syndrome measurement from $+1$ to $-1$ and revealing that an error occurred
somewhere in the block.

The four logical codewords are

$$
\begin{align}
|0_L 0_L\rangle &= \tfrac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle) \\
|0_L 1_L\rangle &= \tfrac{1}{\sqrt{2}}(|0011\rangle + |1100\rangle) \\
|1_L 0_L\rangle &= \tfrac{1}{\sqrt{2}}(|0101\rangle + |1010\rangle) \\
|1_L 1_L\rangle &= \tfrac{1}{\sqrt{2}}(|0110\rangle + |1001\rangle)
\end{align}
$$

+++

## Setup

We begin by importing the modules used throughout this tutorial.

```{code-cell} ipython3
import cirq
import numpy as np
import matplotlib.pyplot as plt

from mitiq import MeasurementResult, zne
from mitiq.rem import post_select
from mitiq.zne.inference import RichardsonFactory

np.random.seed(42)
```

We use six qubits in total: four data qubits encoding the logical state, and two ancilla
qubits used to read out the $S_X = XXXX$ and $S_Z = ZZZZ$ stabilizers.

```{code-cell} ipython3
q0, q1, q2, q3, a_x, a_z = cirq.LineQubit.range(6)
qubit_order = [q0, q1, q2, q3, a_x, a_z]
```

+++

## Encoding circuit and logical operators

Preparing $|0_L 0_L\rangle = \tfrac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$ is simply a
4-qubit GHZ state: a Hadamard on `q0` followed by CNOTs onto the other three data qubits.

```{code-cell} ipython3
encode = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q0, q2),
    cirq.CNOT(q0, q3),
)
```

The logical $X$ and $Z$ operators for this code act on pairs of physical qubits:

- $\overline{X}_1 = X_{q_1} X_{q_3}$, $\quad \overline{X}_2 = X_{q_2} X_{q_3}$
- $\overline{Z}_1 = Z_{q_0} Z_{q_1}$, $\quad \overline{Z}_2 = Z_{q_0} Z_{q_2}$

Both logical $X$ operators commute with the two stabilizers $S_X$ and $S_Z$ (each shares an
even-weight overlap with both), so applying them moves us between codewords without tripping
either syndrome. We apply $\overline{X}_1$ to prepare the state
$|1_L 0_L\rangle = \tfrac{1}{\sqrt{2}}(|0101\rangle + |1010\rangle)$, which will be our target
state for the rest of the tutorial.

```{code-cell} ipython3
logical_x1 = cirq.Circuit(cirq.X(q1), cirq.X(q3))
```

We can quickly check that this prepares the expected codeword:

```{code-cell} ipython3
state = cirq.Simulator().simulate(encode + logical_x1).final_state_vector
for i, amp in enumerate(state):
    if abs(amp) > 1e-6:
        print(format(i, "04b"), np.round(amp, 3))
```

+++

## Syndrome measurement

To measure the stabilizers without destroying the logical state, each stabilizer is measured
using its own ancilla qubit:

- $S_X = XXXX$: prepare the ancilla `a_x` in $|+\rangle$, apply a CNOT from `a_x` onto each data
qubit, then measure `a_x` in the $Z$ basis.
- $S_Z = ZZZZ$: prepare the ancilla `a_z` in $|+\rangle$, apply a CZ between `a_z` and each data
qubit, then measure `a_z` in the $Z$ basis.

A syndrome outcome of $0$ corresponds to stabilizer eigenvalue $+1$ (no error detected), while
an outcome of $1$ corresponds to eigenvalue $-1$ (an error was detected).

```{code-cell} ipython3
syndrome_extraction = cirq.Circuit(
    cirq.H(a_x),
    cirq.CNOT(a_x, q0), cirq.CNOT(a_x, q1), cirq.CNOT(a_x, q2), cirq.CNOT(a_x, q3),
    cirq.H(a_x),
    cirq.H(a_z),
    cirq.CZ(a_z, q0), cirq.CZ(a_z, q1), cirq.CZ(a_z, q2), cirq.CZ(a_z, q3),
    cirq.H(a_z),
)

circuit = encode + logical_x1 + syndrome_extraction
print(circuit)
```

As a sanity check, in the absence of noise both ancillas should always be measured as $0$, and
the logical observable $\overline{Z}_1 = Z_{q_0} Z_{q_1}$ should have expectation value $-1$ for
the $|1_L 0_L\rangle$ state we prepared.

```{code-cell} ipython3
ideal_check = circuit + cirq.measure(*qubit_order, key="m")
samples = cirq.Simulator().run(ideal_check, repetitions=1000).measurements["m"]

print("Syndrome outcomes (a_x, a_z):", set(map(tuple, samples[:, 4:])))
print("<Z_L1> =", np.mean((-1.0) ** (samples[:, 0] ^ samples[:, 1])))
```

+++

## Noise model and executor

We add local depolarizing noise of strength `NOISE_LEVEL` after every gate using a simulator. To
keep the tutorial fast, we compute the final noisy density matrix once per circuit and sample
measurement outcomes from its diagonal, rather than running each shot through an independent
trajectory.

```{code-cell} ipython3
NOISE_LEVEL = 0.02  # 2% depolarizing probability per gate
SHOTS = 10_000

dm_simulator = cirq.DensityMatrixSimulator()


def sample_result(circuit: cirq.Circuit, noise_level: float, shots: int = SHOTS) -> MeasurementResult:
    """Simulate `circuit` with local depolarizing noise and return `shots` measurement outcomes."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(noise_level)) if noise_level > 0 else circuit
    rho = dm_simulator.simulate(noisy_circuit, qubit_order=qubit_order).final_density_matrix
    probabilities = np.clip(np.real(np.diag(rho)), 0, None)
    probabilities /= probabilities.sum()

    outcomes = np.random.choice(len(probabilities), size=shots, p=probabilities)
    n = len(qubit_order)
    bitstrings = [[(outcome >> (n - 1 - k)) & 1 for k in range(n)] for outcome in outcomes]
    return MeasurementResult(bitstrings)
```

We will track the expectation value of the logical observable $\overline{Z}_1 = Z_{q_0} Z_{q_1}$,
whose ideal value for $|1_L 0_L\rangle$ is $-1$.

```{code-cell} ipython3
def expectation_zl1(result: MeasurementResult) -> float:
    bits = result.asarray
    return float(np.mean((-1.0) ** (bits[:, 0] ^ bits[:, 1])))
```

+++

## Step 1: Ideal and unmitigated results

```{code-cell} ipython3
ideal_result = sample_result(circuit, noise_level=0.0)
ideal_value = expectation_zl1(ideal_result)

noisy_result = sample_result(circuit, noise_level=NOISE_LEVEL)
noisy_value = expectation_zl1(noisy_result)

print(f"Ideal <Z_L1>:        {ideal_value:+.4f}")
print(f"Noisy, unmitigated:  {noisy_value:+.4f}")
```

As expected, depolarizing noise pulls the noisy expectation value substantially towards zero.

+++

## Step 2: Error detection via post-selection

We now use {func}`mitiq.rem.post_select` to discard shots in which either syndrome ancilla
detected an error, i.e. measured a $1$. With our qubit ordering, the two ancilla outcomes are the
last two entries of each bitstring:

```{code-cell} ipython3
def no_error_detected(bits) -> bool:
    """Keep only shots where both syndrome qubits measured 0."""
    return bits[-2] == 0 and bits[-1] == 0


detected_result = post_select(noisy_result, no_error_detected)
detected_value = expectation_zl1(detected_result)
fraction_kept = detected_result.shots / noisy_result.shots

print(f"Error detection only: {detected_value:+.4f}  (kept {fraction_kept:.1%} of shots)")
```

Error detection alone already reduces the gap to the ideal value, simply by throwing away the
shots that we know are unreliable. The cost of this improvement is that more than half of our
shots are discarded — an overhead that grows quickly with the noise level and circuit size, since
*any* single-qubit error on the data qubits, the ancillas, or the syndrome-extraction gates
themselves can trigger a syndrome flip.

+++

## Step 3: Combining error detection with ZNE

Finally, we apply ZNE on top of error detection. At each noise-scale factor, we fold the circuit,
re-run it with the same physical noise level, post-select on the syndromes, and compute the
logical expectation value from the surviving shots. {func}`mitiq.zne.execute_with_zne` then
extrapolates these values to the zero-noise limit.

```{code-cell} ipython3
def executor(scaled_circuit: cirq.Circuit) -> float:
    result = sample_result(scaled_circuit, noise_level=NOISE_LEVEL)
    selected = post_select(result, no_error_detected)
    return expectation_zl1(selected)


factory = RichardsonFactory(scale_factors=[1.0, 3.0, 5.0])
combined_value = zne.execute_with_zne(
    circuit, executor, factory=factory, scale_noise=zne.scaling.fold_global
)

print(f"Error detection + ZNE: {combined_value:+.4f}")
```

+++

## Comparing all four approaches

```{code-cell} ipython3
labels = ["Ideal", "Noisy\n(unmitigated)", "Error\ndetection only", "Error detection\n+ ZNE"]
values = [ideal_value, noisy_value, detected_value, combined_value]
colors = ["#444444", "#d62728", "#1f77b4", "#2ca02c"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, values, color=colors)
ax.axhline(ideal_value, color="black", linestyle="--", linewidth=1)
ax.set_ylabel(r"$\langle \overline{Z}_1 \rangle$")
ax.set_title(r"Recovering $\langle \overline{Z}_1 \rangle$ for $|1_L 0_L\rangle$")
ax.set_ylim(-1.2, 0.2)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val - 0.07 if val < 0 else val + 0.03,
        f"{val:.3f}",
        ha="center",
        va="top" if val < 0 else "bottom",
        fontsize=9,
    )
plt.tight_layout()
plt.show()
```

Each additional layer of mitigation moves the result closer to the ideal value of $-1$: noise
alone destroys most of the signal, error detection by itself recovers part of it, and combining
error detection with ZNE recovers the rest (with a small overshoot, a common artifact of
Richardson extrapolation).

+++

## Discussion: the cost of discarding shots

The [[4,2,2]] code can only *detect* errors, not correct them, so every detected error means a
discarded shot. In this example, roughly half of all shots were discarded after error detection,
meaning that twice as many circuit executions are needed to reach the same statistical precision
as an undetected (but biased) experiment. This overhead grows with the size and depth of the
circuit being protected, and is compounded further once ZNE's noise-scaled circuits — which are
themselves more error-prone — are folded in.

In practice, this trade-off is often worthwhile: a modest increase in the number of shots can
remove a large systematic bias that no amount of additional sampling alone could fix.
Combining a lightweight error-detecting code with a software mitigation technique like ZNE is one
way to get a meaningful accuracy improvement on near-term hardware without the much larger qubit
overhead of a full error-*correcting* code.
