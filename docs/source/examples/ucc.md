---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
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

We’ll:

1. Generate a quantum circuit using Mitiq's benchmarking library.
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

For testing purposes we create a random clifford+$T$ circuit.

```{code-cell} ipython3
random_circuit = mitiq.benchmarks.generate_random_clifford_t_circuit(
    num_qubits=2,
    num_oneq_cliffords=100,
    num_twoq_cliffords=50,
    num_t_gates=50,
    seed=90,
)

print(random_circuit)
print("circuit depth:", len(random_circuit))
```

## Step 3: Compile with UCC

UCC will optimize the circuit to reduce the total number of gates.
Since noise accrues with circuit depth, shorter circuits are inherently more robust to noise.
This example shows a dramatic reduction in the number of gates, but to see the impact of UCC on a variety of circuits see the [UCC benchmark repository](https://github.com/unitaryfoundation/ucc-bench).

```{code-cell} ipython3
compiled_circuit = ucc.compile(random_circuit)

print(compiled_circuit)
print("compiled depth:", len(compiled_circuit))
```

## Step 4: Define a Noisy Simulator

The dominant source of noise on modern devices is associated with two-qubit gates.
For simplicity, we define a simulator that adds depolarizing noise after each two-qubit gate, but no noise after single-qubit gates.

```{code-cell} ipython3
def execute(circuit, noise_level=0.05):
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
ideal_uncompiled = execute(random_circuit, noise_level=0.0)
noisy_uncompiled = execute(random_circuit)

ideal_compiled = execute(compiled_circuit, noise_level=0.0)
noisy_compiled = execute(compiled_circuit)
```

## Step 5: Apply ZNE

Here ZNE is applied using the default parameters (Richardson extrapolation with scale factors 1, 3, and 5, and random unitary folding for the noise scaling method).

```{code-cell} ipython3
mitigated_uncompiled = mitiq.zne.execute_with_zne(random_circuit, execute)
mitigated_compiled = mitiq.zne.execute_with_zne(compiled_circuit, execute)
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
bar_width = 0.6

ax.barh(y, noisy, height=bar_width, color="lightgray", label="Noisy")

ax.barh(
    y,
    mitigated,
    left=noisy,
    height=bar_width,
    hatch="///",
    edgecolor="black",
    facecolor="none",
    label="Mitigated",
)

ax.axvline(x=ideal_uncompiled, color="blue", linestyle="--", linewidth=1)
ax.text(
    ideal_uncompiled,
    0.4,
    "Ideal",
    color="blue",
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
The above plot is a single instance of a random circuit.
Despite setting a random seed, non-deterministic behavior in the simulator means results vary from run to run.
While mitigation on the uncompiled circuit is almost always an improvement, the same is unfortunately not true for the compiled circuit due to it's short depth.

From inspection, the compiled circuit usually contains $< 5$ CNOT gates, which is where error is accrued.
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
