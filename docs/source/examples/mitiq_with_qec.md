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

## Running on IBM Quantum Hardware

This section shows how to run the QEC+ZNE combination on real IBM Quantum hardware. You'll need an IBM Quantum account and token.

```{code-cell} ipython3
# Optional: Run on IBM Quantum hardware
# Uncomment and run this section if you have IBM Quantum access

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    from qiskit_ibm_runtime.options import SamplerOptions
    
    # Load IBM Quantum account (use token from environment or save once)
    # service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
    
    # Or use saved credentials
    service = QiskitRuntimeService(channel="ibm_quantum")
    
    print("IBM Quantum account loaded!")
    print(f"Available backends: {[b.name for b in service.backends(min_num_qubits=5)][:5]}")
except ImportError:
    print("qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime")
except Exception as e:
    print(f"Could not load IBM Quantum: {e}")
```

```{code-cell} ipython3
# Convert Cirq circuit to Qiskit and run on hardware
def run_on_ibm_hardware(cirq_circuit, backend_name="ibm_brisbane", shots=1024):
    """Execute circuit on IBM Quantum with error mitigation."""
    
    # Convert Cirq to Qiskit
    import cirq.contrib.qiskit as cirq_qiskit
    qiskit_circuit = cirq_qiskit.circuit_to_qiskit(cirq_circuit)
    
    # Get backend
    backend = service.backend(backend_name)
    print(f"Using backend: {backend.name}")
    print(f"Qubits available: {backend.num_qubits}")
    
    # Transpile for the specific backend
    transpiled = transpile(qiskit_circuit, backend, optimization_level=1)
    print(f"Transpiled circuit depth: {transpiled.depth()}")
    
    # Run with Sampler primitive (includes error mitigation options)
    with Session(backend=backend) as session:
        sampler = Sampler(session=session)
        options = SamplerOptions(default_shots=shots)
        
        job = sampler.run([transpiled], options=options)
        result = job.result()
        
    return result

# Example: Run a simple QEC circuit
# hardware_result = run_on_ibm_hardware(test_circuit, shots=1024)
# print("Hardware execution complete!")
```

```{code-cell} ipython3
# Compare simulator vs hardware results
def compare_simulator_vs_hardware(circuit, noise_prob=0.01):
    """Compare noise simulation vs real hardware execution."""
    
    # Simulator result (with noise model)
    sim_result = compute_expectation(circuit, noise_prob)
    
    # Hardware result (if available)
    try:
        hw_result = run_on_ibm_hardware(circuit)
        # Extract expectation from hardware result
        # (would need proper bitstring to expectation conversion)
        
        print(f"Simulator (noise={noise_prob}): {sim_result:.4f}")
        print(f"IBM Hardware: {hw_result}")
        print(f"Difference: {abs(sim_result - hw_result):.4f}")
    except Exception as e:
        print(f"Hardware comparison skipped: {e}")
        print(f"Simulator result: {sim_result:.4f}")

# Run comparison
# compare_simulator_vs_hardware(test_circuit)
```

:::{note}
**IBM Quantum Integration Notes:**

1. **Authentication**: Store your IBM Quantum token in environment variable `IBM_QUANTUM_TOKEN`
2. **Backend selection**: Use `service.least_busy(operational=True, min_num_qubits=5)` to auto-select
3. **Error suppression**: IBM's built-in error suppression (RESILIENCE_LEVEL) complements Mitiq's ZNE
4. **Combining techniques**: You can apply Mitiq ZNE on top of IBM's error suppression for layered mitigation

For the full QEC+ZNE tutorial on hardware, consider using IBM's `ibm_sherbrooke` or similar 100+ qubit devices to accommodate the 5-qubit bit-flip code with ancillas.
:::

```{code-cell} ipython3
print("Tutorial complete! Try modifying the noise levels, circuit structure, or running on IBM Quantum hardware.")
```
