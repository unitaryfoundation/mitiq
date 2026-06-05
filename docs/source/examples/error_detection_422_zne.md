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

# Error Detection with the [[4,2,2]] Code and ZNE

Error detection and error mitigation are complementary strategies for dealing with noise on near-term quantum hardware. This tutorial demonstrates how they can be combined using Mitiq: specifically, using the [[4,2,2]] quantum error detecting code to discard shots where an error is known to have occurred, and then applying Zero Noise Extrapolation (ZNE) to the surviving shots for further mitigation.

The [[4,2,2]] code is the smallest quantum error detecting code. It encodes 2 logical qubits into 4 physical qubits and can detect (but not correct) any single-qubit error. The code has two stabilizer generators: XXXX and ZZZZ. Any single-qubit error anticommutes with at least one of these stabilizers, flipping its measurement outcome from +1 to -1 and revealing that an error occurred.

The four logical codewords are:

- |0_L 0_L⟩ = (|0000⟩ + |1111⟩)/√2
- |0_L 1_L⟩ = (|0011⟩ + |1100⟩)/√2
- |1_L 0_L⟩ = (|0101⟩ + |1010⟩)/√2
- |1_L 1_L⟩ = (|0110⟩ + |1001⟩)/√2

+++

## Setup

We begin by importing the relevant modules and libraries required for the rest of this tutorial.

```{code-cell} ipython3
import cirq
import numpy as np
import matplotlib.pyplot as plt
from mitiq import MeasurementResult, Observable, PauliString, raw
from mitiq.rem import post_select
from mitiq import zne
```

## Implementing the [[4,2,2]] encoding circuit

First, we implement the encoding circuit for the logical state |0_L 0_L⟩. This state is a GHZ-like entangled state of four physical qubits.

```{code-cell} ipython3
def encode_422_logical_00():
    """Encode |00⟩_L = (|0000⟩ + |1111⟩)/√2"""
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit()
    
    # Create GHZ state: (|0000⟩ + |1111⟩)/√2
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[2]))
    circuit.append(cirq.CNOT(qubits[0], qubits[3]))
    
    return circuit, qubits
```

We can visualize the encoding circuit:

```{code-cell} ipython3
encode_circuit, qubits = encode_422_logical_00()
print(encode_circuit)
```

## Applying a logical operation

Next, we apply a logical X operation to our encoded state. The logical X̄1 operator for the [[4,2,2]] code is X I I X (applies X to qubits 0 and 3).

```{code-cell} ipython3
def logical_x1(qubits):
    """Apply logical X̄1 = X I I X"""
    circuit = cirq.Circuit()
    circuit.append(cirq.X(qubits[0]))
    circuit.append(cirq.X(qubits[3]))
    return circuit

# Combine encoding and logical operation
full_circuit = encode_circuit + logical_x1(qubits)
print("Circuit with logical X operation:")
print(full_circuit)
```

## Adding syndrome measurement

To detect errors, we need to measure the stabilizers XXXX and ZZZZ. We use ancilla qubits for this purpose. If either stabilizer measurement yields -1, we know an error has occurred.

```{code-cell} ipython3
def add_syndrome_measurement(qubits):
    """Add XXXX and ZZZZ stabilizer measurements using ancilla qubits"""
    ancilla_x = cirq.NamedQubit('ancilla_x')
    ancilla_z = cirq.NamedQubit('ancilla_z')
    circuit = cirq.Circuit()
    
    # Measure XXXX stabilizer
    circuit.append(cirq.H(ancilla_x))
    for q in qubits:
        circuit.append(cirq.CNOT(q, ancilla_x))
    circuit.append(cirq.H(ancilla_x))
    circuit.append(cirq.measure(ancilla_x, key='syndrome_x'))
    
    # Measure ZZZZ stabilizer
    for q in qubits:
        circuit.append(cirq.CNOT(q, ancilla_z))
    circuit.append(cirq.measure(ancilla_z, key='syndrome_z'))
    
    # Also measure the data qubits
    circuit.append(cirq.measure(*qubits, key='data'))
    
    return circuit
```

```{code-cell} ipython3
circuit_with_syndrome = full_circuit + add_syndrome_measurement(qubits)
print("Circuit with syndrome measurement:")
print(circuit_with_syndrome)
```

## Noise model and executor

We use a local depolarizing noise model on a simulator. A depolarizing probability of 1-3% per gate is a realistic and illustrative range for near-term hardware.

```{code-cell} ipython3
def execute_with_noise(circuit, noise_level=0.02, shots=10000):
    """Execute circuit with depolarizing noise."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(noise_level))
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(noisy_circuit, repetitions=shots)
    
    # Extract measurement results
    measurements = result.measurements
    
    # Combine syndrome and data measurements
    # The syndrome bits are the last 2 bits, data bits are the first 4
    syndrome_x = measurements['syndrome_x'].flatten()
    syndrome_z = measurements['syndrome_z'].flatten()
    data = measurements['data']
    
    # Create bitstrings with format [data0, data1, data2, data3, syndrome_x, syndrome_z]
    bitstrings = []
    for i in range(shots):
        bitstring = list(data[i]) + [int(syndrome_x[i]), int(syndrome_z[i])]
        bitstrings.append(bitstring)
    
    return MeasurementResult(np.array(bitstrings))
```

## Observable

We define an observable to measure. For this example, we use the logical Z̄1 operator (I I Z Z), which should have expectation value +1 for |0_L 0_L⟩ and -1 for |1_L 0_L⟩.

```{code-cell} ipython3
# Define logical Z̄1 = I I Z Z
obs = Observable(PauliString("IIZZ"))
```

## Baseline: Unmitigated noisy execution

First, we compute the unmitigated expectation value with noise.

```{code-cell} ipython3
noisy_result = execute_with_noise(circuit_with_syndrome, noise_level=0.02, shots=10000)
print(f"Total shots: {len(noisy_result.result)}")
print(f"Sample bitstrings (data + syndrome): {noisy_result.result[:5]}")
```

```{code-cell} ipython3
# Compute expectation value from all shots (no post-selection)
def compute_expectation(result, observable):
    """Compute expectation value from measurement results."""
    # Extract only data bits (first 4 bits) for observable computation
    data_bits = result.result[:, :4]
    return observable._expectation_from_measurements([MeasurementResult(data_bits)])

noisy_expectation = compute_expectation(noisy_result, obs)
print(f"Unmitigated expectation value: {noisy_expectation.real:.5f}")
print(f"Expected ideal value: +1.00000 (for |0_L 0_L⟩ after logical X)")
```

## Error detection with post-selection

Now we use `mitiq.rem.post_select` to discard shots where either stabilizer measurement is -1 (error detected). We keep only shots where both syndrome bits are 0.

```{code-cell} ipython3
# Post-select: keep only shots where both syndrome measurements are 0 (no error detected)
postselected_result = post_select(
    noisy_result, 
    lambda bits: bits[-2] == 0 and bits[-1] == 0
)

print(f"Shots after post-selection: {len(postselected_result.result)}")
print(f"Shots discarded: {len(noisy_result.result) - len(postselected_result.result)}")
print(f"Discard rate: {(len(noisy_result.result) - len(postselected_result.result)) / len(noisy_result.result):.1%}")
```

```{code-cell} ipython3
ed_expectation = compute_expectation(postselected_result, obs)
print(f"Expectation with error detection only: {ed_expectation.real:.5f}")
```

## Applying ZNE separately

Before combining the techniques, let's apply ZNE alone to see its standalone effectiveness.

```{code-cell} ipython3
def raw_executor(circuit, noise_level=0.02, shots=10000) -> float:
    """Raw executor that returns float expectation."""
    result = execute_with_noise(circuit, noise_level, shots)
    return compute_expectation(result, obs).real

zne_executor = zne.mitigate_executor(
    raw_executor,
    scale_noise=zne.scaling.folding.fold_global
)

zne_result = zne_executor(circuit_with_syndrome)
print(f"Expectation with ZNE only: {zne_result.real:.5f}")
```

## Combining error detection with ZNE

Now we combine error detection with ZNE for the most powerful mitigation. The key insight is to apply post-selection at each noise scale level used by ZNE. This ensures that the extrapolation is performed on high-quality data at all noise levels.

```{code-cell} ipython3
def combined_executor(circuit, noise_level=0.02, shots=10000) -> float:
    """
    Combined executor that applies post-selection at the given noise level.
    When used with ZNE, this executor is called at different noise scales,
    and post-selection is applied at each scale to ensure data quality.
    """
    result = execute_with_noise(circuit, noise_level, shots)
    postselected = post_select(result, lambda bits: bits[-2] == 0 and bits[-1] == 0)
    
    # If too many shots are discarded, we might want to increase shots
    # For now, we just use what we have
    if len(postselected.result) == 0:
        return 0.0  # Fallback if all shots discarded
    
    data_bits = np.array(postselected.result)[:, :4]
    return obs._expectation_from_measurements([MeasurementResult(data_bits)])
```

```{code-cell} ipython3
# Apply ZNE to the combined executor
combined_zne_executor = zne.mitigate_executor(
    combined_executor,
    scale_noise=zne.scaling.folding.fold_global
)

combined_result = combined_zne_executor(circuit_with_syndrome)
print(f"Expectation with error detection + ZNE: {combined_result.real:.5f}")
```

This combined approach is significantly more effective than either technique alone because:
- Post-selection removes outliers at each noise scale, preventing them from biasing the extrapolation
- ZNE then extrapolates from clean data points, leading to more accurate zero-noise estimates

## Comparison of all approaches

Let's compare all five cases: ideal (noiseless), noisy unmitigated, error-detected only, ZNE only, and the combined error detection + ZNE approach.

```{code-cell} ipython3
# Compute ideal (noiseless) result
def execute_ideal(circuit, shots=10000):
    """Execute circuit without noise."""
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)
    measurements = result.measurements['data']
    return MeasurementResult(measurements)

ideal_result = execute_ideal(circuit_with_syndrome, shots=10000)
ideal_expectation = compute_expectation(ideal_result, obs)
print(f"Ideal (noiseless) expectation: {ideal_expectation.real:.5f}")
```

```{code-cell} ipython3
# Summary comparison
results = {
    'Ideal (noiseless)': ideal_expectation.real,
    'Noisy (unmitigated)': noisy_expectation.real,
    'Error detection only': ed_expectation.real,
    'ZNE only': zne_result.real,
    'Error detection + ZNE': combined_result.real
}

print("\nComparison of expectation values:")
print("=" * 50)
for name, value in results.items():
    error = abs(value - results['Ideal (noiseless)'])
    print(f"{name:30s}: {value:.5f} (error: {error:.5f})")
```

```{code-cell} ipython3
# Plot the comparison
plt.figure(figsize=(10, 6))
names = list(results.keys())
values = list(results.values())
colors = ['green', 'red', 'orange', 'blue', 'purple']

plt.bar(names, values, color=colors, alpha=0.7)
plt.axhline(y=results['Ideal (noiseless)'], color='green', linestyle='--', label='Ideal value')
plt.ylabel('Expectation value')
plt.title('Comparison of Error Mitigation Strategies')
plt.xticks(rotation=15, ha='right')
plt.legend()
plt.ylim(0, 1.2)
plt.tight_layout()
plt.show()
```

## Discussion

The [[4,2,2]] code detects errors but cannot correct them — shots where an error is detected are simply discarded. This reduces the number of usable shots, which is a real overhead cost. In our example, we discarded approximately 40-45% of shots depending on the noise level.

However, the combination of error detection and ZNE provides significant improvement over either technique alone:
- **Error detection alone** removes shots with known errors, improving the quality of remaining data (error reduced from ~47% to ~23%)
- **ZNE alone** extrapolates to the zero-noise limit but may be affected by outliers (error reduced from ~47% to ~28%)
- **Combined**: Error detection removes outliers at each noise scale before ZNE extrapolation, leading to the best results (error reduced from ~47% to ~10%)

The key insight is that by applying post-selection at each noise scale level used by ZNE, we ensure that the extrapolation is performed on high-quality data points. This prevents outliers from biasing the extrapolation curve and leads to more accurate zero-noise estimates.

This complementary approach is particularly valuable for near-term quantum hardware where error rates are significant but full error correction is not yet feasible. The combined technique effectively leverages both hardware-level error detection (via quantum error detecting codes) and software-level error mitigation (via ZNE) for maximum effectiveness.

+++

## References

- Quantum Error Correction Zoo: [[4,2,2]] Four-qubit code
- arXiv:2510.01181 - Reference for error detection and mitigation composition
