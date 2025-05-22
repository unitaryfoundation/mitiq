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

# What additional options are available when using VD?

The function used for virtual distillation is `execute_with_vd()`. Based on the current implementation, the function signature is:

```python
from mitiq.vd import execute_with_vd

vd_values: List[float] = execute_with_vd(
    circuit,
    executor
)
```

## Required Parameters

- **circuit**: A `cirq.Circuit` object representing the quantum circuit to apply VD to
- **executor**: A callable function that takes a circuit and returns measurement results. The executor should return a `MeasurementResult` object containing bitstring measurement outcomes

## Current Limitations

- **Observable**: Only Pauli-Z measurements are supported (hardcoded in the implementation)
- **Number of copies**: Only `M=2` copies are currently supported
- **Shots**: Use an odd number of shots to prevent normalization issues where the denominator could become zero
- **Output**: Returns expectation values $\langle Z_i \rangle$ for each qubit $i$ in the original circuit

## Example Usage

```{code-cell} ipython3
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

The function automatically handles the circuit duplication, applies the diagonalizing gates, and processes the measurement results according to the VD protocol.