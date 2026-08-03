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

# What additional options are available when using PT?

Pauli Twirling in Mitiq targets CNOT and CZ gates via
{func}`.generate_pauli_twirl_variants`. This page covers how to add custom
noise channels to twirled circuits using {func}`.add_noise_to_two_qubit_gates`.


## Adding noise to twirled circuits with `add_noise_to_two_qubit_gates`

The function {func}`.add_noise_to_two_qubit_gates` inserts a noise
channel after every CNOT and CZ gate in a twirled circuit. This is
useful for simulating the effect of PT on circuits with specific noise
models.

Two noise channels are built in: `"bit-flip"` and `"depolarize"`.

```{code-cell} ipython3
import cirq
from cirq import LineQubit, Circuit, CNOT
from mitiq.pt import generate_pauli_twirl_variants, add_noise_to_two_qubit_gates

q0, q1 = LineQubit.range(2)
cnot_circuit = Circuit(CNOT(q0, q1))

twirled = generate_pauli_twirl_variants(cnot_circuit, num_circuits=1)[0]

noisy_twirled = add_noise_to_two_qubit_gates(twirled, "depolarize", p=0.05)
print("Twirled circuit with depolarizing noise:\n")
print(noisy_twirled)
```

### Extending the noise dictionary

To add a custom noise channel, update the `CIRQ_NOISE_OP` dictionary
before calling {func}`.generate_pauli_twirl_variants` with a `noise_name`.

```{code-cell} ipython3
from mitiq.pt import CIRQ_NOISE_OP

CIRQ_NOISE_OP["phase-flip"] = cirq.phase_flip

noisy_twirled_custom = add_noise_to_two_qubit_gates(
    twirled, "phase-flip", p=0.03
)
print("Twirled circuit with custom phase-flip noise:\n")
print(noisy_twirled_custom)
```

The same approach works for generating variants with built-in noise
via the `noise_name` argument:

```{code-cell} ipython3
twirled_with_noise = generate_pauli_twirl_variants(
    cnot_circuit, num_circuits=3, noise_name="phase-flip", p=0.03
)
print(f"Generated {len(twirled_with_noise)} twirled circuits with phase-flip noise.")
print(f"\nExample:\n{twirled_with_noise[0]}")
```

```{tip}
Any callable with signature `(float) -> cirq.Gate` can be added to the
dictionary. This covers any single-parameter Cirq noise channel.
```

## How the number of generated Pauli twirled circuits affects the outcome

The number of twirled circuits generated (controlled by `num_circuits`) determines how well the physical average approximates the ideal, intended Pauli channel. 

- **Small `num_circuits` (e.g., 1-10):** The variance in the expectation values will be quite high because the Pauli group hasn't been adequately sampled over. The resulting effective noise channel might still contain coherent properties.
- **Large `num_circuits` (e.g., 20-100+):** By the law of large numbers, the averaged results converge toward the exact stochastic Pauli channel. However, generating and evaluating too many circuits increases the simulation or execution cost. 

As a starting point, 20 to 50 twirl variants per expectation value often provides a reasonable balance between sufficient noise tailoring and execution overhead on physical hardware; the right number depends on circuit size, noise rate, and the precision required.

## Which noise channels are tailored by PT?

PT aims to convert coherent or generally asymmetric noise into purely stochastic Pauli noise. Any noise channel with off-diagonal terms in its Pauli Transfer Matrix (PTM) benefits from twirling, which zeroes out those off-diagonals. 

The following heatmaps provide a visual representation of how different original noise channels are transformed (tailored) after Pauli Twirling.
The code below generates these heatmaps by computing the Pauli Transfer Matrix (PTM) of each channel before and after twirling.

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
paulis = [I, X, Y, Z]
labels = ["I", "X", "Y", "Z"]

def ptm(channel_kraus):
    """Compute the 4x4 Pauli Transfer Matrix of a channel given its Kraus operators."""
    n = 4
    mat = np.zeros((n, n))
    for i, Pi in enumerate(paulis):
        for j, Pj in enumerate(paulis):
            val = 0.0
            for K in channel_kraus:
                KPj = K @ Pj
                KPjKd = KPj @ K.conj().T
                val += np.trace(Pi @ KPjKd).real
            mat[i, j] = val / 2.0
    return mat

def twirl_ptm(ptm_in):
    """Twirling a channel: keep only diagonal elements (exact result for Pauli twirling)."""
    return np.diag(np.diag(ptm_in))

def coherent_kraus(theta=0.3):
    K = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return [K]

def amplitude_damping_kraus(gamma=0.2):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return [K0, K1]

def depolarizing_kraus(p=0.1):
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z
    return [K0, K1, K2, K3]

channels = [
    ("Coherent over-rotation", coherent_kraus()),
    ("Amplitude damping", amplitude_damping_kraus()),
    ("Depolarizing", depolarizing_kraus()),
]

fig, axes = plt.subplots(len(channels), 2, figsize=(7, 9))
for row, (name, kraus) in enumerate(channels):
    p_before = ptm(kraus)
    p_after  = twirl_ptm(p_before)
    for col, (mat, title) in enumerate([(p_before, "Before PT"), (p_after, "After PT")]):
        ax = axes[row, col]
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_title(f"{name}\n{title}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("../img/pt_ptm_heatmaps.png", dpi=100, bbox_inches="tight")
plt.show()
```

```{figure} ../img/pt_ptm_heatmaps.png
---
width: 600px
name: pt-heatmaps
---
PTM heatmaps before and after PT. The off-diagonal block terms of Coherent over-rotations (top) and Amplitude Damping (middle) are perfectly zeroed out. Depolarizing noise (bottom), which is already stochastic Pauli noise, remains unchanged.
```

## Stacking PT with other QEM techniques

Because PT does not remove noise but rather structures it favorably (eliminating coherent worst-case scaling), it is almost always best used as a **preprocessing step** before applying a dedicated quantum error mitigation technique like [Zero-Noise Extrapolation (ZNE)](zne.md) or [Probabilistic Error Cancellation (PEC)](pec.md).

For example, ZNE relies on the assumption that noise scales predictably as the circuit depth increases. Coherent errors violate this assumption by scaling quadratically or constructively. By applying PT *at each noise scale level*, the noise becomes stochastic, guaranteeing a much smoother extrapolation curve. 

When stacking PT with a QEM technique:
1. Generate the base mitigating circuits (e.g. at various scale factors for ZNE).
2. Apply PT variants around the multi-qubit gates in *each* of the scaled circuits.
3. Compute the average expectation value for the PT variants at each scale level.
4. Feed those partially noise-tailored expectation values into the QEM inference engine.

```{figure} ../img/pt_qem_workflow.svg
---
width: 700px
name: pt-qem-workflow
---
Workflow demonstrating the stacking of PT before inference in a broader QEM pipeline.
```
