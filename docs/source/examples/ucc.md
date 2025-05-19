---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  name: python3
---

```{tags} ucc, zne, beginner
```

# Using the UCC Compiler with Mitiq’s Error Mitigation

In this tutorial, we combine circuit compilation from the [Unitary Compiler Collection (UCC)](https://ucc.readthedocs.io/) with [Mitiq](https://mitiq.readthedocs.io/)'s Zero Noise Extrapolation (ZNE) technique for error mitigation. We'll explore how compilation affects noise sensitivity and how mitigation can restore accuracy.

## 🧰 Setup

Before starting, install the required packages:

```bash
pip install mitiq
pip install ucc
pip install cirq
```

## 🔁 Workflow

We'll be completing the following steps:

1. Define a quantum circuit.
2. Compile it with UCC.
3. Simulate it under a simple noise model.
4. Apply error mitigation using ZNE.
5. Compare ideal, noisy, and mitigated expectation values.

---

## Step 1: Imports

```{code-cell} ipython3
import cirq

import mitiq
import ucc
```

## Step 2: Create a Testing Circuit

For testing purposes we use a circuit that simulates the dynamics of the [Heisenberg model](https://en.wikipedia.org/wiki/Quantum_Heisenberg_model) on a square lattice.
[This circuit](https://github.com/unitaryfoundation/ucc-bench/blob/bc3e88e9c564efdb9e5a7af7493a7e1811c8fbf9/benchmarks/circuits/benchpress/square_heisenberg_N9_basis_rz_rx_ry_cx.qasm) is one of the circuits UCC is benchmarked on in the [`ucc-bench`](https://github.com/unitaryfoundation/ucc-bench) repository.

```{code-cell} ipython3
from cirq.contrib.qasm_import import circuit_from_qasm

with open("resources/square_heisenberge.qasm") as f:
    qasm = f.read()

circuit = circuit_from_qasm(qasm)

print(circuit)
print("circuit depth:", len(circuit))
print(
    "two qubit gate count: ",
    len([op for op in circuit.all_operations() if len(op.qubits) == 2]),
)
```

## Step 3: Compile with UCC

UCC will optimize the circuit to reduce the total number of gates.
Since noise accrues with circuit depth, shorter circuits are inherently more robust to noise.
This example shows a dramatic reduction in the number of gates, but to see the impact of UCC on a variety of circuits see the [UCC benchmark repository](https://github.com/unitaryfoundation/ucc-bench).

```{code-cell} ipython3
compiled = ucc.compile(circuit)

print(compiled)
print("compiled depth:", len(compiled))
print(
    "two qubit gate count: ",
    len([op for op in compiled.all_operations() if len(op.qubits) == 2]),
)
```

## Step 4: Define a Noisy Simulator

The dominant source of noise on modern devices is associated with two-qubit gates.
For simplicity, we define a simulator that adds depolarizing noise after each two-qubit gate, but no noise after single-qubit gates.

```{code-cell} ipython3
def execute(circuit, noise_level=0.01):
    noisy_circuit = cirq.Circuit()
    for op in circuit.all_operations():
        noisy_circuit.append(op)
        if len(op.qubits) == 2:
            noisy_circuit.append(
                cirq.depolarize(p=noise_level, n_qubits=2)(*op.qubits)
            )
    result = cirq.DensityMatrixSimulator(seed=42).simulate(noisy_circuit)
    return result.final_density_matrix[0, 0].real  # Probability of |00⟩
```

### Baseline: Ideal vs Noisy

```{code-cell} ipython3
ideal_uncompiled = execute(circuit, noise_level=0.0)
noisy_uncompiled = execute(circuit)

ideal_compiled = execute(compiled, noise_level=0.0)
noisy_compiled = execute(compiled)
```

## Step 5: Apply ZNE

Here ZNE is applied using the default parameters (Richardson extrapolation with scale factors 1, 3, and 5, and random unitary folding for the noise scaling method).

```{code-cell} ipython3
mitigated_uncompiled = mitiq.zne.execute_with_zne(circuit, execute)
mitigated_compiled = mitiq.zne.execute_with_zne(compiled, execute)
```

## Step 6: Compare the Results

We can now visualize the 4 methods of execution:

| compiled\mitigated | ❌                 | ✅                     |
| ------------------ | ------------------ | ---------------------- |
| ❌                 | `noisy_uncompiled` | `mitigated_uncompiled` |
| ✅                 | `noisy_compiled`   | `mitigated_compiled`   |

```{code-cell} ipython3
:dropdown: true
:tags: [hide-input]

import matplotlib.pyplot as plt

categories = ["Uncompiled", "Compiled"]
noisy = [noisy_uncompiled, noisy_compiled]
mitigated = [
    mitigated_uncompiled - noisy_uncompiled,
    mitigated_compiled - noisy_compiled,
]  # Delta from noisy
ideal = [ideal_uncompiled, ideal_compiled]

fig, ax = plt.subplots(figsize=(6, 3))

y = range(len(categories))
bar_width = 0.55

ax.barh(y, noisy, height=bar_width, color="lightcoral", label="Noisy")

ax.barh(
    y,
    mitigated,
    left=noisy,
    height=bar_width,
    hatch="//",
    edgecolor="firebrick",
    facecolor="none",
    label="Mitigated",
)

ax.axvline(x=ideal_uncompiled, color="maroon", linestyle="--", linewidth=1)
ax.text(
    ideal_uncompiled,
    0.4,
    "Ideal",
    color="maroon",
    ha="right",
)

ax.set_yticks(y)
ax.set_yticklabels(categories)
ax.set_xlabel("Expectation Value")
ax.set_title("Impact of Mitigation on pre- and post-compiled circuit")

ax.legend()
plt.show()
```

```{warning}
The above plot is a single instance of a given circuit.
The behavior of compilation and mitigation acting beneficially is a common phenomenon, but not always guaranteed.
```

## Conclusions

- **Compilation** minimizes the potential for errors to accrue.
- **Error mitigation** remains effective on compiled circuits.
- Combined use of UCC and ZNE gives significantly better results under noise.

Try running this pipeline over many random circuits to get a better statistical picture.

## References

- [UCC GitHub](https://github.com/unitaryfoundation/ucc)
- [Mitiq Documentation](https://mitiq.readthedocs.io/)
- [ZNE Paper](https://arxiv.org/abs/1612.02058)
