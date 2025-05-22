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

## What is VD?
Virtual distillation is an error mitigation technique based on the following paper: [VD article](https://arxiv.org/pdf/2011.07064). VD leverages $M$ copies of a state $\rho$ to suppress the error term. Virtual distillation describes the approximation of the error-free expectation value of an operator $O$ as:

$$
\langle O \rangle_{corrected} = \dfrac{Tr(O\rho^M)}{Tr(\rho^M)}
$$

As described in the paper, we make use of the following equality:

$$
Tr(O\rho^M) = Tr(O^{\textbf{i}}S^{(M)}\rho^{\otimes M})
$$

This equation allows us to use $M$ copies of $\rho$ instead of calculating $\rho^M$ directly.

# How do I use VD?

As with all techniques, Virtual Distillation (VD) is compatible with any frontend supported by Mitiq:

```python
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

For the purpose of this tutorial we will use `cirq`:
```{code-cell} ipython3
frontend = "cirq"
```

## Problem setup
The VD implementation requires:
1. A quantum circuit to apply error mitigation to
2. An executor function that runs the circuit and returns measurement results

Currently, VD only supports $M=2$ copies and measurements of the Pauli-Z observable on each qubit.

## Applying VD
Below we provide an example of applying VD:

```python
import cirq
from mitiq import vd, MeasurementResult

# Create a simple example circuit WITHOUT measurements
# VD will add measurements automatically
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1])
])

def noisy_executor(circuit, noise_level=0.01, shots=1001) -> MeasurementResult:
    circuit_to_run = circuit.with_noise(cirq.depolarize(noise_level))
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(circuit_to_run, repetitions=shots)
    bitstrings = np.column_stack(list(result.measurements.values()))
    qubit_indices = tuple(
            int(q[2:-1])  # Extract index from "q(index)" string
            for k in result.measurements.keys()
            for q in k.split(",")
    )
    return MeasurementResult(bitstrings, qubit_indices)

# Apply VD
mitigated_expectation_values = vd.execute_with_vd(circuit, noisy_executor)
print(f"Z expectation values for each qubit: {mitigated_expectation_values}")
```

The function returns a list of error-mitigated expectation values $\langle Z_i \rangle$ for each qubit $i$ in the original circuit.