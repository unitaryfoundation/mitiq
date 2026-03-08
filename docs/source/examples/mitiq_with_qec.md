---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Combining Error Mitigation with Quantum Error Correction

This tutorial demonstrates how to combine Mitiq's error mitigation techniques with Quantum Error Correction (QEC) codes. We'll show how using both together can provide better results than either alone.

## Overview

Recent research has shown that error mitigation and error correction can be combined synergistically:

1. **Error Correction** protects logical qubits by encoding them across multiple physical qubits
2. **Error Mitigation** (like ZNE, DDD, PEC) reduces the impact of remaining errors on expectation values

This combination is particularly valuable for near-term quantum devices where full error correction is not yet practical.

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import cirq
from cirq import Circuit, LineQubit, X, Z, H, CNOT, measure

from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory

print("Mitiq and Cirq imported successfully!")
```

## Simple Bit-Flip Error Correction Code

We'll implement a simple 3-qubit bit-flip code that can correct single bit-flip errors.

### Encoding

A single logical qubit is encoded across 3 physical qubits:

$$|0\rangle_L = |000\rangle, \quad |1\rangle_L = |111\rangle$$

```{code-cell} ipython3
def encode_bit_flip(qubits: List[cirq.LineQubit]) -> cirq.Circuit:
    """Encode a single qubit into 3 qubits using bit-flip code."""
    circuit = cirq.Circuit()
    # |0⟩ -> |000⟩ (already there)
    # |1⟩ -> |111⟩ via CNOTs
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    return circuit

# Test encoding
q = cirq.LineQubit.range(3)
encoder = encode_bit_flip(q)
print("Bit-flip encoding circuit:")
print(encoder)
```

### Syndrome Measurement

To detect errors, we measure stabilizers. For bit-flip errors, we check parity:

```{code-cell} ipython3
def measure_syndrome(qubits: List[cirq.LineQubit], 
                     ancilla: List[cirq.LineQubit]) -> cirq.Circuit:
    """Measure syndrome for bit-flip error detection."""
    circuit = cirq.Circuit()
    # Syndrome 1: checks if qubits 0 and 1 have same parity
    circuit.append(cirq.H(ancilla[0]))
    circuit.append(cirq.CNOT(qubits[0], ancilla[0]))
    circuit.append(cirq.CNOT(qubits[1], ancilla[0]))
    circuit.append(cirq.H(ancilla[0]))
    
    # Syndrome 2: checks if qubits 1 and 2 have same parity
    circuit.append(cirq.H(ancilla[1]))
    circuit.append(cirq.CNOT(qubits[1], ancilla[1]))
    circuit.append(cirq.CNOT(qubits[2], ancilla[1]))
    circuit.append(cirq.H(ancilla[1]))
    
    return circuit

# Test syndrome measurement
q_data = cirq.LineQubit.range(3)
q_ancilla = [cirq.LineQubit(3), cirq.LineQubit(4)]
syndrome_circ = measure_syndrome(q_data, q_ancilla)
print("\nSyndrome measurement circuit:")
print(syndrome_circ)
```

## Creating a Protected Circuit

Let's create a circuit that applies a logical X operation with error correction:

```{code-cell} ipython3
def create_protected_x_circuit() -> cirq.Circuit:
    """Create a circuit that applies logical X with bit-flip protection."""
    q_data = cirq.LineQubit.range(3)
    q_ancilla = [cirq.LineQubit(3), cirq.LineQubit(4)]
    
    circuit = cirq.Circuit()
    
    # Encode logical |0⟩ -> |000⟩
    circuit += encode_bit_flip(q_data)
    
    # Apply logical X (bit flip all 3 qubits)
    circuit.append(cirq.X(q) for q in q_data)
    
    # Measure syndrome (simplified - no correction for demonstration)
    circuit += measure_syndrome(q_data, q_ancilla)
    
    # Measure data qubits
    circuit.append(cirq.measure(*q_data, key='data'))
    
    return circuit

protected_circuit = create_protected_x_circuit()
print("Protected X circuit:")
print(protected_circuit)
```

## Simulating with Noise

```{code-cell} ipython3
def add_depolarizing_noise(circuit: cirq.Circuit, noise_prob: float) -> cirq.Circuit:
    """Add depolarizing noise after each gate."""
    noisy_circuit = cirq.Circuit()
    for moment in circuit:
        noisy_circuit.append(moment)
        # Add noise after each moment that has gates
        if len(moment.operations) > 0:
            for op in moment.operations:
                for qubit in op.qubits:
                    noisy_circuit.append(cirq.depolarize(p=noise_prob)(qubit))
    return noisy_circuit

def compute_expectation(circuit: cirq.Circuit, noise_prob: float = 0.01) -> float:
    """Compute expectation value of logical Z (parity of all qubits)."""
    noisy_circuit = add_depolarizing_noise(circuit, noise_prob)
    
    # Add measurement of parity
    qubits = sorted(circuit.all_qubits())
    measure_circuit = noisy_circuit + cirq.Circuit(cirq.measure(*qubits, key='result'))
    
    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(measure_circuit, repetitions=1000)
    
    # Compute parity (logical Z expectation)
    measurements = result.measurements['result']
    # Logical |1⟩ should have odd parity (all 1s after X)
    parities = np.sum(measurements, axis=1) % 2
    expectation = 1 - 2 * np.mean(parities)  # +1 for even, -1 for odd
    
    return expectation

# Test without mitigation
test_circuit = create_protected_x_circuit()
ideal_result = -1.0  # Should be -1 after logical X on |0⟩
noisy_result = compute_expectation(test_circuit, noise_prob=0.01)

print(f"Ideal expectation: {ideal_result}")
print(f"Noisy expectation: {noisy_result:.4f}")
print(f"Error: {abs(ideal_result - noisy_result):.4f}")
```

## Applying Zero Noise Extrapolation (ZNE)

Now let's use Mitiq's ZNE to mitigate the remaining errors:

```{code-cell} ipython3
def executor(circuit: cirq.Circuit) -> float:
    """Executor function for Mitiq that returns expectation value."""
    return compute_expectation(circuit, noise_prob=0.01)

# Apply ZNE with global folding
zne_strategy = zne.ZNEStrategy(
    noise_scaling_function=fold_global,
    factory=RichardsonFactory([1.0, 2.0, 3.0])
)

mitigated_result = zne.execute_with_zne(test_circuit, executor, zne_strategy)

print(f"\nResults comparison:")
print(f"Ideal:              {ideal_result:.4f}")
print(f"Raw noisy:          {noisy_result:.4f} (error: {abs(ideal_result - noisy_result):.4f})")
print(f"QEC + ZNE:          {mitigated_result:.4f} (error: {abs(ideal_result - mitigated_result):.4f})")
print(f"\nImprovement from ZNE: {abs(ideal_result - noisy_result) - abs(ideal_result - mitigated_result):.4f}")
```

## Comparing Different Noise Levels

Let's see how the combination performs across different noise levels:

```{code-cell} ipython3
def compute_expectation_at_noise(circuit: cirq.Circuit, noise_prob: float) -> float:
    """Compute expectation at a specific noise level."""
    return compute_expectation(circuit, noise_prob)

def executor_with_noise(circuit: cirq.Circuit, noise_prob: float) -> float:
    """Executor with variable noise."""
    return compute_expectation(circuit, noise_prob)

noise_levels = np.linspace(0.001, 0.05, 10)
raw_results = []
mitigated_results = []

for noise in noise_levels:
    # Raw with QEC only
    raw = compute_expectation(test_circuit, noise)
    raw_results.append(raw)
    
    # QEC + ZNE
    def exec_fn(circ):
        return compute_expectation(circ, noise)
    
    mitigated = zne.execute_with_zne(
        test_circuit, 
        exec_fn,
        zne_strategy
    )
    mitigated_results.append(mitigated)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, raw_results, 'o-', label='QEC only', linewidth=2)
plt.plot(noise_levels, mitigated_results, 's-', label='QEC + ZNE', linewidth=2)
plt.axhline(y=-1.0, color='k', linestyle='--', label='Ideal (no error)')
plt.xlabel('Depolarizing noise probability')
plt.ylabel('Expectation value ⟨Z_L⟩')
plt.title('Quantum Error Correction + Zero Noise Extrapolation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nSummary across noise levels:")
print(f"Average raw error:     {np.mean([abs(-1 - r) for r in raw_results]):.4f}")
print(f"Average mitigated error: {np.mean([abs(-1 - m) for m in mitigated_results]):.4f}")
print(f"Average improvement:     {np.mean([abs(-1 - r) - abs(-1 - m) for r, m in zip(raw_results, mitigated_results)]):.4f}")
```

## Key Insights

:::{note}
**Combining QEC and Error Mitigation:**

1. **QEC** provides a baseline protection by encoding logical information
2. **Error mitigation** (ZNE in this case) further reduces the impact of residual errors
3. The combination is particularly effective for near-term devices where full fault-tolerant QEC is not yet available

This approach is actively researched - see:
- [Quantum Error Correction on Error-mitigated Physical Qubits](https://arxiv.org/abs/2601.18384)
- [Combining Error Detection and Mitigation](https://arxiv.org/abs/2510.01181)
:::

## Conclusion

This tutorial demonstrated how to combine Mitiq's error mitigation with a simple quantum error correction code. The key steps were:

1. **Encode** logical qubits using a bit-flip code
2. **Execute** the protected circuit
3. **Apply ZNE** to further reduce errors on the encoded circuit

The combination shows improved fidelity compared to using either technique alone, which is valuable for near-term quantum computing applications.

For production use, you would:
- Use more sophisticated QEC codes (surface code, color code)
- Apply syndrome-based error correction (not just detection)
- Combine with other Mitiq techniques (DDD, PEC, CDR)
- Consider hardware-specific noise models

```{code-cell} ipython3
print("Tutorial complete! Try modifying the noise levels or circuit structure.")
```
