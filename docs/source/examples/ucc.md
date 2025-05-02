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

For testing purposes we create a rotated randomized benchmarking circuit.
Randomized benchmarking circuits are circuits that ultimately perform the identity operation, and hence when passed to a compiler like `ucc` are reduced to identity operations.
To avoide this, a $R_Z$ gate is applied to the first qubit of the circuit in the circuits midpoint.
In the code below we fix a $\theta$ which stumps the compiler enough for an interesting tutorial.

```{code-cell} ipython3
num_qubits = 2
depth = 40

random_circuit = mitiq.benchmarks.generate_rotated_rb_circuits(
    num_qubits, depth, theta=0.234, seed=20
)[0]

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
def execute(circuit, noise_level=0.02):
    noisy_circuit = cirq.Circuit()
    for op in circuit.all_operations():
        noisy_circuit.append(op)
        if len(op.qubits) == 2:
            noisy_circuit.append(
                cirq.depolarize(p=noise_level, n_qubits=2)(*op.qubits)
            )
    result = cirq.DensityMatrixSimulator().simulate(noisy_circuit)
    return result.final_density_matrix[0, 0].real  # Probability of |000⟩
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

```{code-cell} ipython3
header = "{:<15} {:<10} {:<10}"
row = "{:<15} {:<10.4f} {:<10.4f}"

print(header.format("", "Uncompiled", "Compiled"))
print(row.format("Ideal", ideal_uncompiled, ideal_compiled))
print(row.format("Noisy", noisy_uncompiled, noisy_compiled))
print(row.format("Mitigated", mitigated_uncompiled, mitigated_compiled))
print(
    row.format(
        "Mitigated Error",
        abs(ideal_uncompiled - mitigated_uncompiled),
        abs(ideal_compiled - mitigated_compiled),
    )
)
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
