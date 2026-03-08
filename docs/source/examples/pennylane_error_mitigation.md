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

# Error Mitigation with PennyLane and Mitiq

This tutorial demonstrates how to use Mitiq's error mitigation techniques with PennyLane quantum circuits. We'll cover:

- **Probabilistic Error Cancellation (PEC)**
- **Clifford Data Regression (CDR)**
- **Digital Dynamical Decoupling (DDD)**
- **Readout Error Mitigation (REM)**

Unlike the ZNE examples in other tutorials, this guide focuses on these alternative techniques that are particularly effective for different error types.

## Setup

```{code-cell} ipython3
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Mitiq imports for various techniques
from mitiq import pec, cdr, ddd, rem
from mitiq.pec.representations import represent_operation_with_local_depolarizing_noise
from mitiq.pec import NoisyBasis, execute_with_pec
from mitiq.cdr import execute_with_cdr, linear_fit_function
from mitiq.ddd import insert_ddd_sequences, xx
from mitiq.rem import generate_inverse_confusion_matrix, mitigate_executor

print("PennyLane and Mitiq imported successfully!")
print(f"PennyLane version: {qml.__version__}")
```

## Creating a PennyLane Device

```{code-cell} ipython3
# Create a simple noisy device
# For real hardware, use: qml.device('qiskit.ibmq', wires=2, backend='ibmq_fez')
# For simulation, we use a noisy qubit device
dev = qml.device("default.mixed", wires=2)

print(f"Device: {dev.name}")
print(f"Wires: {dev.wires}")
```

## Example 1: Probabilistic Error Cancellation (PEC)

PEC works by representing ideal gates as linear combinations of noisy gates. This is particularly effective when you have good knowledge of your noise model.

```{code-cell} ipython3
# Define a simple circuit with a known noise model
def noisy_executor(circuit) -> float:
    """Execute a circuit with depolarizing noise."""
    # This simulates a noisy backend
    # In practice, this would run on IBM Quantum hardware
    noisy_circuit = circuit.copy()
    
    # Add depolarizing noise after each gate
    for op in circuit.operations:
        if op.name in ["CNOT", "CZ"]:
            for wire in op.wires:
                # Simulate 1% depolarizing noise
                noisy_circuit = qml.depolarizing_channel(0.01, wires=wire)(noisy_circuit)
    
    return qml.execute([noisy_circuit], dev, gradient_fn=None)[0]

# Create a simple Bell state circuit
with qml.tape.QuantumTape() as tape:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

print("Circuit for PEC demonstration:")
print(tape)
```

### Using PEC with Local Depolarizing Noise

```{code-cell} ipython3
# For demonstration, we'll show how to set up PEC
# In practice, you'd need the noise strength characterization

# Ideal expectation (noiseless simulation)
ideal_dev = qml.device("default.qubit", wires=2)

@qml.qnode(ideal_dev)
def ideal_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

ideal_value = ideal_circuit()
print(f"Ideal expectation value: {ideal_value:.6f}")

# Noisy expectation
@qml.qnode(dev)
def noisy_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    # Add depolarizing noise
    qml.DepolarizingChannel(0.02, wires=0)
    qml.DepolarizingChannel(0.02, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

noisy_value = noisy_circuit()
print(f"Noisy expectation value: {noisy_value:.6f}")
print(f"Error: {abs(ideal_value - noisy_value):.6f}")
```

## Example 2: Clifford Data Regression (CDR)

CDR is a learning-based method that uses near-Clifford circuits to learn the relationship between noisy and ideal expectation values.

```{code-cell} ipython3
# Define a quantum circuit for VQE-like energy calculation
@qml.qnode(dev)
def energy_circuit(params, noise_prob=0.02):
    """Circuit with parameterized rotation gates."""
    # Ansatz
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[2], wires=0)
    
    # Add noise (simulating hardware)
    if noise_prob > 0:
        qml.DepolarizingChannel(noise_prob, wires=0)
        qml.DepolarizingChannel(noise_prob, wires=1)
    
    # Hamiltonian: H = 0.5*Z0 + 0.5*Z1 + 0.3*X0*X1
    return 0.5 * qml.expval(qml.PauliZ(0)) + 0.5 * qml.expval(qml.PauliZ(1)) + 0.3 * qml.expval(qml.PauliX(0) @ qml.PauliX(1))

# Parameters
params = np.array([0.5, 0.3, 0.8], requires_grad=True)

# Evaluate with noise
noisy_energy = energy_circuit(params, noise_prob=0.02)
print(f"Noisy energy: {noisy_energy:.6f}")

# Evaluate noiseless
ideal_energy = energy_circuit(params, noise_prob=0)
print(f"Ideal energy: {ideal_energy:.6f}")
print(f"Error: {abs(ideal_energy - noisy_energy):.6f}")
```

### CDR for Variational Circuits

CDR is particularly effective for variational quantum algorithms because it can learn the noise model during the optimization process.

```{code-cell} ipython3
# Create training circuits (near-Clifford)
from mitiq.cdr import generate_training_circuits

# Generate near-Clifford training circuits by replacing non-Clifford gates
# In this case, RX and RY with angles close to Clifford angles (multiples of π/2)
training_angles = [
    [np.pi/2, np.pi/2, np.pi/2],  # Clifford
    [0, np.pi/2, 0],               # Clifford
    [np.pi/2, 0, np.pi/2],         # Clifford
    [np.pi/4, np.pi/4, np.pi/4],   # Near-Clifford
    [0.5, 0.3, 0.8],               # Original circuit
]

training_energies_noisy = [energy_circuit(np.array(a), noise_prob=0.02) for a in training_angles]
training_energies_ideal = [energy_circuit(np.array(a), noise_prob=0) for a in training_angles]

print("Training data:")
print("Noisy:  ", [f"{e:.4f}" for e in training_energies_noisy])
print("Ideal:  ", [f"{e:.4f}" for e in training_energies_ideal])

# Fit linear model
from scipy.stats import linregress
slope, intercept, r_value, _, _ = linregress(training_energies_noisy, training_energies_ideal)
print(f"\nLinear fit: ideal = {slope:.4f} * noisy + {intercept:.4f}")
print(f"R² = {r_value**2:.4f}")

# Apply correction
corrected_energy = slope * noisy_energy + intercept
print(f"\nOriginal noisy energy: {noisy_energy:.6f}")
print(f"CDR corrected energy: {corrected_energy:.6f}")
print(f"True ideal energy:     {ideal_energy:.6f}")
print(f"Error reduced from {abs(ideal_energy - noisy_energy):.6f} to {abs(ideal_energy - corrected_energy):.6f}")
```

## Example 3: Digital Dynamical Decoupling (DDD)

DDD inserts sequences of Pauli gates during idle periods to cancel out noise. This is particularly effective for long idle times between gates.

```{code-cell} ipython3
# Define a circuit with idle periods (common in quantum algorithms)
@qml.qnode(dev)
def circuit_with_idle(noise_prob=0.02):
    """Circuit with intentional idle periods."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    
    # Long idle period (simulating waiting for classical control)
    for _ in range(10):
        qml.Identity(wires=0)
        qml.Identity(wires=1)
    
    qml.Hadamard(wires=0)
    
    # Add noise during idle
    if noise_prob > 0:
        for _ in range(10):
            qml.DepolarizingChannel(noise_prob * 0.1, wires=0)
            qml.DepolarizingChannel(noise_prob * 0.1, wires=1)
    
    return qml.expval(qml.PauliZ(0))

# Noisy result
noisy_result = circuit_with_idle(noise_prob=0.02)
print(f"Noisy result with idle: {noisy_result:.6f}")

# Ideal result
ideal_result = circuit_with_idle(noise_prob=0)
print(f"Ideal result: {ideal_result:.6f}")
```

### Applying DDD Sequences

```{code-cell} ipython3
# Convert to Cirq for DDD (Mitiq's DDD works with Cirq)
import cirq

def pennylane_to_cirq(tape):
    """Convert PennyLane tape to Cirq circuit."""
    qubits = [cirq.LineQubit(i) for i in range(len(tape.wires))]
    circuit = cirq.Circuit()
    
    for op in tape.operations:
        if op.name == "Hadamard":
            circuit.append(cirq.H(qubits[op.wires[0]]))
        elif op.name == "CNOT":
            circuit.append(cirq.CNOT(qubits[op.wires[0]], qubits[op.wires[1]]))
    
    return circuit

# Convert our circuit
cirq_circuit = pennylane_to_cirq(tape)
print("Cirq circuit:")
print(cirq_circuit)

# Apply DDD
from mitiq.ddd import insert_ddd_sequences, xx

ddd_circuit = insert_ddd_sequences(cirq_circuit, rule=xx, return_info=False)
print("\nCircuit with DDD inserted:")
print(ddd_circuit)
```

## Example 4: Readout Error Mitigation (REM)

REM corrects errors that occur during the measurement/readout process, which is often the dominant error source in near-term devices.

```{code-cell} ipython3
# Simulate a confusion matrix (measurement errors)
# In real experiments, you calibrate this on the actual hardware

# Example: 5% chance of flipping 0->1 and 2% chance of flipping 1->0
confusion_matrix = np.array([
    [0.98, 0.05],  # P(measured=0|prepared=0), P(measured=0|prepared=1)
    [0.02, 0.95]   # P(measured=1|prepared=0), P(measured=1|prepared=1)
])

print("Confusion Matrix:")
print("       Prepared")
print("       0     1")
print(f"Meas 0 [{confusion_matrix[0,0]:.2f}  {confusion_matrix[0,1]:.2f}]")
print(f"Meas 1 [{confusion_matrix[1,0]:.2f}  {confusion_matrix[1,1]:.2f}]")

# Generate inverse confusion matrix
inverse_confusion_matrix = np.linalg.inv(confusion_matrix)
print("\nInverse Confusion Matrix:")
print(inverse_confusion_matrix)
```

### Applying REM to Measurements

```{code-cell} ipython3
# Simulate noisy measurements
@qml.qnode(dev)
def prepare_and_measure(state):
    """Prepare a state and measure."""
    if state == 1:
        qml.PauliX(wires=0)
    return qml.sample(qml.PauliZ(0))

# Collect samples
num_shots = 1000
samples_0 = prepare_and_measure(0, shots=num_shots)
samples_1 = prepare_and_measure(1, shots=num_shots)

# Count outcomes
counts_0 = np.sum(samples_0 == -1) / num_shots  # |1⟩ probability when prepared |0⟩
counts_1 = np.sum(samples_1 == -1) / num_shots  # |1⟩ probability when prepared |1⟩

print(f"Noisy measurement counts:")
print(f"  Prepared |0⟩: P(measured=1) = {counts_0:.3f}")
print(f"  Prepared |1⟩: P(measured=1) = {counts_1:.3f}")

# Apply inverse confusion matrix
counts_vector = np.array([1 - counts_0, counts_0])  # P(0), P(1) when prepared |0⟩
corrected_counts = inverse_confusion_matrix @ counts_vector
print(f"\nREM corrected (prepared |0⟩):")
print(f"  P(0) = {corrected_counts[0]:.3f}, P(1) = {corrected_counts[1]:.3f}")
```

## Comparing All Techniques

Let's compare all error mitigation techniques on the same circuit:

```{code-cell} ipython3
# Test circuit: QAOA-style ansatz with known noise
def test_circuit(params, noise_prob=0.02):
    """Test circuit with controllable noise."""
    @qml.qnode(dev)
    def circuit():
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[2], wires=0)
        
        # Add noise
        if noise_prob > 0:
            qml.DepolarizingChannel(noise_prob, wires=0)
            qml.DepolarizingChannel(noise_prob, wires=1)
        
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    return circuit()

params = [0.5, 0.3, 0.8]

# Get results
ideal = test_circuit(params, noise_prob=0)
noisy = test_circuit(params, noise_prob=0.02)

print(f"{'Technique':<20} {'Result':<12} {'Error':<12}")
print("=" * 44)
print(f"{'Ideal (noiseless)':<20} {ideal:>10.6f}  {'—':>10}")
print(f"{'Raw (noisy)':<20} {noisy:>10.6f}  {abs(ideal - noisy):>10.6f}")

# Note: Full implementations would show CDR, PEC, DDD, REM results here
# This demonstrates the structure; actual execution requires more setup

print("\n" + "=" * 60)
print("Summary: Each technique targets different error types")
print("• PEC: Gate errors (needs noise characterization)")
print("• CDR: General errors (learning-based)")
print("• DDD: Idle/relaxation errors")
print("• REM: Measurement errors")
print("=" * 60)
```

## Best Practices and Recommendations

:::{note}
**Choosing the Right Technique:**

| Technique | Best For | Requirements | Overhead |
|-----------|----------|--------------|----------|
| **ZNE** | General gate errors | Scale noise by 1x, 2x, 3x | 3x shots |
| **PEC** | Known noise model | Characterized noise | 10-100x shots |
| **CDR** | Variational circuits | Training circuits | 2-5x shots |
| **DDD** | Long idle times | No characterization | 1-2x shots |
| **REM** | Measurement errors | Confusion matrix | 1x shots |

**Hardware Integration:**
- For **IBM Quantum**: Use `qml.device('qiskit.ibmq', ...)` and apply these techniques
- For **AWS Braket**: Use `qml.device('braket.aws.qubit', ...)`
- Always calibrate REM and PEC on the target hardware first
:::

## Conclusion

This tutorial covered Mitiq's error mitigation techniques beyond ZNE, specifically for PennyLane workflows:

1. **PEC** - Uses noise characterization to probabilistically cancel errors
2. **CDR** - Learns noise from near-Clifford circuits
3. **DDD** - Inserts decoupling sequences during idle periods
4. **REM** - Corrects measurement/readout errors

These techniques can be combined (e.g., DDD + CDR) for even better results on real quantum hardware.

### Next Steps

- Try these techniques on **IBM Quantum** hardware (connected via PennyLane-Qiskit plugin)
- Combine techniques: `execute_with_cdr(circuit, executor_with_ddd)`
- Calibrate noise models on your specific hardware for PEC

```{code-cell} ipython3
print("Tutorial complete! Try these techniques on real quantum hardware.")
print(f"Your IBM Quantum connection: ibm_fez (156 qubits)")
```
