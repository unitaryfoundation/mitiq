---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```{tags} rem, zne, cirq, intermediate
```

# Composing techniques: error detection with the `[[4,2,2]]` code and ZNE

Error *detection* and error *mitigation* are complementary ways of coping with noise on near-term
hardware. An error-detecting code flags runs in which an error is known to have occurred so that
they can be thrown away, while a mitigation technique such as Zero-Noise Extrapolation (ZNE)
post-processes the *surviving* runs to estimate the noise-free expectation value. In this tutorial
we combine the two: we encode a logical state in the smallest quantum error **detecting** code, the
`[[4,2,2]]` code, discard the shots in which a syndrome measurement reveals an error using
`mitiq.rem.post_select`, and then apply [ZNE](../guide/zne.md) to what remains.

The `[[4,2,2]]` code encodes 2 logical qubits into 4 physical qubits. It can *detect* (but not
correct) any single-qubit error, because every single-qubit Pauli error anticommutes with at least
one of its two stabilizer generators, `XXXX` and `ZZZZ`, flipping the corresponding syndrome from
`+1` to `-1`. Its four logical codewords are

$$
\begin{aligned}
|0_L 0_L\rangle &= \tfrac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle), &
|0_L 1_L\rangle &= \tfrac{1}{\sqrt{2}}(|0011\rangle + |1100\rangle), \\
|1_L 0_L\rangle &= \tfrac{1}{\sqrt{2}}(|0101\rangle + |1010\rangle), &
|1_L 1_L\rangle &= \tfrac{1}{\sqrt{2}}(|0110\rangle + |1001\rangle).
\end{aligned}
$$

A useful reference for the encoding circuit and stabilizer structure is the
[Error Correction Zoo entry for this code](https://errorcorrectionzoo.org/c/stab_4_2_2); the
interplay between detection and mitigation is discussed in
[arXiv:2510.01181](https://arxiv.org/abs/2510.01181).

+++

## Setup

```{code-cell} ipython3
from functools import partial

import cirq
import matplotlib.pyplot as plt
import numpy as np

from mitiq import MeasurementResult, zne
from mitiq.rem import post_select
from mitiq.zne.inference import LinearFactory

# Four data qubits and two ancillas used to read out the ZZZZ and XXXX syndromes.
data = cirq.LineQubit.range(4)
anc_z, anc_x = cirq.LineQubit.range(4, 6)
```

## The logical circuit

We prepare the logical state $|0_L 0_L\rangle$, which is just a GHZ state on the four data qubits,
and then apply a logical operation. A convenient choice of logical operators for this code is

$$
\bar{X}_1 = X_1 X_3, \quad \bar{Z}_1 = Z_0 Z_1, \qquad
\bar{X}_2 = X_2 X_3, \quad \bar{Z}_2 = Z_0 Z_2,
$$

each a weight-two Pauli that commutes with both stabilizers. We apply the logical bit-flip
$\bar{X}_1 = X_1 X_3$, taking $|0_L 0_L\rangle \to |1_L 0_L\rangle$. Our observable is the logical
$\bar{Z}_1 = Z_0 Z_1$: it has the ideal value $\langle \bar{Z}_1 \rangle = -1$ on $|1_L 0_L\rangle$,
and because it is a product of $Z$ operators we can read it off directly from computational-basis
measurements of the data qubits.

```{code-cell} ipython3
def logical_circuit() -> cirq.Circuit:
    """Encode |0_L 0_L>, then apply the logical bit-flip Xbar_1 = X_1 X_3."""
    circuit = cirq.Circuit()
    # Encode |0_L 0_L> = (|0000> + |1111>)/sqrt(2).
    circuit.append(cirq.H(data[0]))
    circuit.append(cirq.CNOT(data[0], data[i]) for i in (1, 2, 3))
    # Logical X on the first logical qubit: |0_L 0_L> -> |1_L 0_L>.
    circuit.append([cirq.X(data[1]), cirq.X(data[3])])
    return circuit


print(logical_circuit())
```

## Syndrome measurement

To detect errors we measure the two stabilizers non-destructively with one ancilla each. The `ZZZZ`
parity is accumulated onto `anc_z` with a ladder of `CNOT`s controlled by the data qubits, and the
`XXXX` parity is read onto `anc_x` by conjugating the analogous ladder with Hadamards. On an
error-free state both ancillas return `0`; a detected error flips one of them to `1`.

```{code-cell} ipython3
def protected_circuit() -> cirq.Circuit:
    """Logical circuit followed by ZZZZ and XXXX syndrome extraction (no measurements yet)."""
    circuit = logical_circuit()
    # ZZZZ syndrome -> anc_z.
    circuit.append(cirq.CNOT(data[i], anc_z) for i in range(4))
    # XXXX syndrome -> anc_x.
    circuit.append(cirq.H(anc_x))
    circuit.append(cirq.CNOT(anc_x, data[i]) for i in range(4))
    circuit.append(cirq.H(anc_x))
    return circuit


base_circuit = protected_circuit()
```

## Noise model and executor

We use a local depolarizing noise model on a density-matrix simulator, so no real hardware is required.
The executor runs the circuit, appends measurements of the data qubits and of the two syndrome
ancillas, and returns the raw shots as a `MeasurementResult`. A fixed simulator seed keeps the
notebook reproducible.

```{code-cell} ipython3
def sample(circuit: cirq.Circuit, noise_level: float, repetitions: int = 20_000) -> MeasurementResult:
    """Run ``circuit`` under local depolarizing noise and return the measured bitstrings.

    Each bitstring is ``[d0, d1, d2, d3, s_z, s_x]``: the four data qubits followed by the two
    syndrome ancillas.
    """
    noisy = circuit.with_noise(cirq.depolarize(noise_level)) if noise_level > 0 else circuit.copy()
    noisy.append(cirq.measure(*data, key="data"))
    noisy.append(cirq.measure(anc_z, key="s_z"))
    noisy.append(cirq.measure(anc_x, key="s_x"))

    result = cirq.DensityMatrixSimulator(seed=7).run(noisy, repetitions=repetitions)
    bitstrings = np.column_stack(
        [result.measurements["data"], result.measurements["s_z"], result.measurements["s_x"]]
    )
    return MeasurementResult(bitstrings)


def logical_z1(result: MeasurementResult) -> float:
    """Estimate <Zbar_1> = <Z_0 Z_1> from the data bits of the surviving shots."""
    bits = np.array(result.result)
    if bits.shape[0] == 0:
        return float("nan")
    return float(np.mean(1 - 2 * ((bits[:, 0] + bits[:, 1]) % 2)))  # average of (-1)^(b0 + b1)


NOISE = 0.02  # ~2% depolarizing probability per gate
```

The ideal (noiseless) value is $-1$, while depolarizing noise pulls the estimate toward $0$:

```{code-cell} ipython3
ideal = logical_z1(sample(base_circuit, 0.0))
noisy = logical_z1(sample(base_circuit, NOISE))
print(f"Ideal   <Zbar_1> = {ideal:+.3f}")
print(f"Noisy   <Zbar_1> = {noisy:+.3f}")
```

## Error detection via post-selection

We now keep only the shots in which **both** syndrome ancillas read `0`, i.e. no error was detected.
`mitiq.rem.post_select` makes this a one-liner over the raw shots (the syndrome bits are the
last two entries of each bitstring).

```{code-cell} ipython3
raw = sample(base_circuit, NOISE)
kept = post_select(raw, lambda bits: bits[-2] == 0 and bits[-1] == 0)

detected = logical_z1(kept)
survival = len(kept.result) / len(raw.result)
print(f"Detected <Zbar_1> = {detected:+.3f}")
print(f"Fraction of shots kept: {survival:.0%}")
```

Discarding flagged shots moves the estimate back toward the ideal value. The price is a reduced
number of usable shots. Here roughly half are thrown away, which is a genuine overhead: to obtain
a target statistical precision one must take correspondingly more shots, and the survival fraction
shrinks as the circuit (and hence the error rate) grows.

## Adding zero-noise extrapolation

Post-selection removes *detected* errors, but undetected errors (for example, two-qubit errors that
preserve both stabilizers) survive. We mitigate the residual bias with ZNE applied to the
post-selected estimator. The executor below returns the post-selected $\langle \bar{Z}_1 \rangle$
as a function of the circuit, which is exactly the interface ZNE expects; ZNE folds the circuit to
amplify the noise and extrapolates back to the zero-noise limit.

```{code-cell} ipython3
def execute_with_detection(circuit: cirq.Circuit, noise_level: float = NOISE) -> float:
    """Post-selected estimate of <Zbar_1>, used as the executor for ZNE."""
    shots = sample(circuit, noise_level)
    kept = post_select(shots, lambda bits: bits[-2] == 0 and bits[-1] == 0)
    return logical_z1(kept)


detected_and_zne = zne.execute_with_zne(
    base_circuit,
    partial(execute_with_detection, noise_level=NOISE),
    factory=LinearFactory(scale_factors=[1, 2, 3]),
    scale_noise=zne.scaling.fold_global,
)
print(f"Detected + ZNE <Zbar_1> = {detected_and_zne:+.3f}")
```

## Comparing all four cases

```{code-cell} ipython3
labels = ["Noisy", "Error-detected", "Error-detected\n+ ZNE", "Ideal"]
values = [noisy, detected, detected_and_zne, ideal]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, values, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
ax.axhline(ideal, color="#1f77b4", linestyle="--", linewidth=1, label="ideal")
ax.set_ylabel(r"$\langle \bar{Z}_1 \rangle$")
ax.set_title(r"[[4,2,2]] error detection + ZNE (depolarizing $p=2\%$)")
ax.set_ylim(-1.1, 0.05)
fig.tight_layout()
plt.show()
```

Each technique moves the estimate closer to the ideal value of $-1$, and the combination outperforms
either one alone: error detection removes the shots with detectable errors, and ZNE then corrects
the residual bias from the errors that detection cannot catch. The two techniques are complementary,
since detection acts at the level of individual shots while ZNE acts on the aggregate expectation
value. This extra accuracy is not free: both the post-selection overhead and the noise-scaled
circuit evaluations consume additional shots.
