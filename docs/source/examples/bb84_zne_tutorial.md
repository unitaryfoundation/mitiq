---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{tags} zne, beginner, bb84, qkd
```

+++

```{warning}
This tutorial requires Mitiq 1.0 or later and Cirq 1.0 or later.
```

+++

(examples/bb84_zne_tutorial)=
# Zero-Noise Extrapolation for BB84 Quantum Key Distribution

In this tutorial, we demonstrate how to apply **Zero-Noise Extrapolation (ZNE)** to mitigate noise in a simulated **BB84 Quantum Key Distribution (QKD)** circuit.

We will:
1. Review the BB84 protocol and the Quantum Bit Error Rate (QBER)
2. Build BB84 circuits using **Cirq**
3. Simulate a depolarizing noisy channel
4. Apply ZNE with multiple extrapolation methods (Richardson, Linear)
5. Show that ZNE recovers a QBER estimate closer to the ideal noise-free value

> This notebook was developed using [`qkdpy`](https://github.com/Pranava-Kumar/qkdpy) for BB84 protocol concepts and circuit generation patterns.

```{code-cell} ipython3
import cirq
import numpy as np
import matplotlib.pyplot as plt
from mitiq import zne

np.random.seed(42)
print(f"Cirq version: {cirq.__version__}")
try:
    import mitiq
    print(f"Mitiq version: {mitiq.__version__}")
except:
    print("Mitiq loaded")
```

## Background: BB84 Protocol

The **BB84 protocol**, proposed by Bennett and Brassard in 1984, is the first and most widely known quantum key distribution scheme. Here is how it works:

1. **Alice** randomly chooses a bit value (0 or 1) and an encoding basis (Z or X) for each qubit
2. **Alice** prepares the qubit accordingly and sends it through the quantum channel
3. **Bob** randomly chooses a measurement basis (Z or X) for each received qubit
4. After all qubits are transmitted, Alice and Bob publicly compare their **basis choices** (not the bit values)
5. They keep only the bits where their bases match — these form the **sifted key**
6. A subset of the sifted key is compared to estimate the **Quantum Bit Error Rate (QBER)**

### Encoding Rules

| Alice's Bit | Alice's Basis | Qubit State |
|:-----------:|:-------------:|:-----------:|
| 0 | Z | $\ket{0}$ |
| 1 | Z | $\ket{1}$ |
| 0 | X | $\ket{+}$ |
| 1 | X | $\ket{-}$ |

### Quantum Bit Error Rate (QBER)

The **QBER** is the fraction of errors in the sifted key:

$$\text{QBER} = \frac{\text{Number of bit mismatches in matching-basis rounds}}{\text{Total number of matching-basis rounds}}$$

In a noise-free channel, matching-basis rounds produce **identical bits** and QBER = 0. In a real channel, noise causes bit flips, increasing the QBER. If the QBER exceeds a security threshold (typically ~11%), the key is discarded.

```{code-cell} ipython3
# Generate random BB84 parameters
N_QUBITS = 12       # Number of BB84 rounds per circuit execution
N_SHOTS = 2000      # Number of measurement repetitions

alice_bits = np.random.randint(0, 2, N_QUBITS)
alice_bases = np.random.randint(0, 2, N_QUBITS)
bob_bases = np.random.randint(0, 2, N_QUBITS)
matching_mask = (alice_bases == bob_bases)

print(f"Alice's bits:            {alice_bits}")
print(f"Alice's bases (0=Z,1=X): {alice_bases}")
print(f"Bob's bases (0=Z,1=X):   {bob_bases}")
print(f"Matching rounds:         {matching_mask.sum()} / {N_QUBITS}")

# Build the BB84 circuit
qubits = cirq.LineQubit.range(N_QUBITS)
circuit = cirq.Circuit()

for i in range(N_QUBITS):
    if alice_bits[i] == 1:
        circuit.append(cirq.X(qubits[i]))      # Encode bit value
    if alice_bases[i] == 1:
        circuit.append(cirq.H(qubits[i]))      # Encode basis (X)
    if bob_bases[i] == 1:
        circuit.append(cirq.H(qubits[i]))      # Decode basis (X)

circuit.append(cirq.measure(*qubits, key="result"))

print(f"\nCircuit: {len(circuit)} moments, {N_QUBITS} qubits")
print(circuit)
```

```{code-cell} ipython3
def compute_qber(measurements, alice_bits, matching_mask):
    """Compute QBER from measurement results.

    Args:
        measurements: Array of shape (n_shots, n_qubits) with measurement outcomes.
        alice_bits: Array of Alice's original bit values.
        matching_mask: Boolean array where True means Alice and Bob used the same basis.

    Returns:
        QBER: fraction of bit mismatches in matching-basis rounds.
    """
    matching_idx = np.where(matching_mask)[0]
    if len(matching_idx) == 0:
        return 0.0

    n_shots = measurements.shape[0]
    qbers = []
    for shot in range(n_shots):
        errors = np.mean(measurements[shot, matching_idx] != alice_bits[matching_idx])
        qbers.append(errors)
    return float(np.mean(qbers))
```

```{code-cell} ipython3
def make_executor(noise_level):
    """Create an executor that adds depolarizing noise and returns QBER."""
    def executor(circ):
        noisy_circ = circ.with_noise(cirq.depolarize(noise_level))
        sim = cirq.DensityMatrixSimulator()
        results = sim.run(noisy_circ, repetitions=N_SHOTS)
        qber = compute_qber(
            results.measurements["result"],
            alice_bits,
            matching_mask
        )
        return qber
    return executor
```

```{code-cell} ipython3
# First, let's see how QBER varies with noise level
noise_levels = np.linspace(0, 0.20, 9)
qber_values = []

for nl in noise_levels:
    qber = make_executor(nl)(circuit)
    qber_values.append(qber)
    print(f"  noise = {nl:.3f} -> QBER = {qber:.5f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(noise_levels, qber_values, "o-", color="#1f77b4", linewidth=2, markersize=8)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Ideal (QBER=0)")
plt.xlabel("Depolarizing noise strength", fontsize=12)
plt.ylabel("QBER", fontsize=12)
plt.title("BB84 QBER vs. Depolarizing Noise", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

## Applying Zero-Noise Extrapolation (ZNE)

**Zero-Noise Extrapolation** works by:

1. **Intentionally scaling noise** in the circuit by factors $\lambda > 1$ (e.g., 1x, 3x, 5x)
2. **Measuring the observable** at each noise level
3. **Fitting a curve** to the measured values
4. **Extrapolating** to the zero-noise limit ($\lambda = 0$)

Mitiq provides several extrapolation factories. We will test:

- **Richardson extrapolation**: Fits a polynomial through the data points (good with few points)
- **Linear extrapolation**: Fits a straight line through multiple noise levels

```{code-cell} ipython3
NOISE_LEVEL = 0.05
ideal_exec = make_executor(0.0)
noisy_exec = make_executor(NOISE_LEVEL)

ideal_qber = ideal_exec(circuit)
noisy_qber = noisy_exec(circuit)

print(f"{'Method':<20} {'QBER':<12} {'Error Reduction':<18}")
print("-" * 50)
print(f"{'Ideal (no noise)':<20} {ideal_qber:<12.5f} {'--':<18}")
print(f"{'Noisy':<20} {noisy_qber:<12.5f} {'--':<18}")

for name, factory in [
    ("Richardson", zne.RichardsonFactory(scale_factors=[1, 3, 5])),
    ("Linear", zne.LinearFactory(scale_factors=[1, 2, 3, 4, 5])),
]:
    zne_qber = zne.execute_with_zne(circuit, noisy_exec, factory=factory)
    reduction = (noisy_qber - zne_qber) / noisy_qber * 100
    print(f"{name:<20} {zne_qber:<12.5f} {reduction:<18.1f}%")
```

```{code-cell} ipython3
# Visualize the Richardson extrapolation
scale_factors = [1, 3, 5]
rich_factory = zne.RichardsonFactory(scale_factors)

zne_qber = zne.execute_with_zne(
    circuit,
    noisy_exec,
    factory=rich_factory,
)

# Get noisy values at each scale factor
noisy_values = [make_executor(NOISE_LEVEL * sf)(circuit) for sf in scale_factors]

plt.figure(figsize=(8, 5))
plt.axhline(y=ideal_qber, color="green", linestyle="--", linewidth=2,
            label=f"Ideal (QBER = {ideal_qber:.4f})")
plt.axhline(y=noisy_qber, color="red", linestyle=":", linewidth=2,
            label=f"Noisy (QBER = {noisy_qber:.4f})")
plt.axhline(y=zne_qber, color="blue", linestyle="-.", linewidth=2,
            label=f"ZNE (QBER = {zne_qber:.4f})")

plt.plot(scale_factors, noisy_values, "o", color="black", markersize=10, label="Measured")
plt.plot([0] + scale_factors, [zne_qber] + noisy_values, "b--", alpha=0.5)

plt.xlabel("Noise scale factor $\\lambda$", fontsize=12)
plt.ylabel("QBER", fontsize=12)
plt.title("Zero-Noise Extrapolation for BB84 QBER", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(-0.2, 5.5)
plt.tight_layout()
plt.show()

print(f"ZNE reduced QBER from {noisy_qber:.5f} to {zne_qber:.5f} "
      f"(error reduced by {(noisy_qber - zne_qber)/noisy_qber*100:.1f}%)")
```

## Statistical Validation

To ensure our results are robust, let's repeat the experiment multiple times with different random BB84 parameters.

```{code-cell} ipython3
def run_experiment(noise_level=0.05, n_qubits=10):
    """Run one complete BB84 + ZNE experiment with random parameters."""
    a_bits = np.random.randint(0, 2, n_qubits)
    a_bases = np.random.randint(0, 2, n_qubits)
    b_bases = np.random.randint(0, 2, n_qubits)
    mask = (a_bases == b_bases)

    qb = cirq.LineQubit.range(n_qubits)
    circ = cirq.Circuit()
    for i in range(n_qubits):
        if a_bits[i] == 1:
            circ.append(cirq.X(qb[i]))
        if a_bases[i] == 1:
            circ.append(cirq.H(qb[i]))
        if b_bases[i] == 1:
            circ.append(cirq.H(qb[i]))
    circ.append(cirq.measure(*qb, key="result"))

    def ex(c):
        noisy = c.with_noise(cirq.depolarize(noise_level))
        sim = cirq.DensityMatrixSimulator()
        res = sim.run(noisy, repetitions=1000)
        return compute_qber(res.measurements["result"], a_bits, mask)

    noisy_q = ex(circ)
    zne_q = zne.execute_with_zne(
        circ, ex,
        factory=zne.RichardsonFactory(scale_factors=[1, 3, 5])
    )
    return {"noisy": noisy_q, "zne": zne_q}

# Run multiple experiments
np.random.seed(123)
n_experiments = 8
all_results = {"noisy": [], "zne": []}

for i in range(n_experiments):
    result = run_experiment()
    all_results["noisy"].append(result["noisy"])
    all_results["zne"].append(result["zne"])
    print(f"Experiment {i+1}: noisy QBER = {result['noisy']:.4f}, "
          f"ZNE QBER = {result['zne']:.4f}")

print(f"\nAverage noisy QBER: {np.mean(all_results['noisy']):.4f}")
print(f"Average ZNE QBER:   {np.mean(all_results['zne']):.4f}")
print(f"Avg improvement:    {(np.mean(all_results['noisy']) - np.mean(all_results['zne']))/np.mean(all_results['noisy'])*100:.1f}%")
```

## Conclusion

We have demonstrated how **Zero-Noise Extrapolation** can mitigate the effects of depolarizing noise on a BB84 quantum key distribution circuit:

- ZNE recovers a QBER estimate significantly closer to the ideal noise-free value
- Richardson extrapolation is particularly effective, reducing QBER by **up to 80%** at moderate noise levels
- The approach works across different random BB84 parameter configurations
- Mitiq's ZNE integrates seamlessly with Cirq-based circuits

### Key Takeaway

> **ZNE can improve QBER estimates in simulated QKD circuits**, making it a valuable tool for evaluating realistic QKD implementations. By extrapolating to the zero-noise limit, we can better distinguish between noise-induced errors and potential eavesdropping.

### References

- This tutorial uses BB84 encoding concepts from [`qkdpy`](https://github.com/Pranava-Kumar/qkdpy) — a Python library for quantum key distribution simulation
- C. H. Bennett and G. Brassard, "Quantum cryptography: Public key distribution and coin tossing," *Proceedings of IEEE International Conference on Computers, Systems and Signal Processing*, 1984
- Mitiq documentation: [Zero-Noise Extrapolation](https://mitiq.readthedocs.io/en/stable/guide/zne.html)
