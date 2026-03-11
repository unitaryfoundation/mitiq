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

# Benchmarking Circuits

Mitiq benchmarks error mitigation techniques by evaluating improvements in metrics such as state fidelity (the closeness of the mitigated quantum state to the ideal state), output probability distributions, and logical error rates. The benchmarking process involves running diverse circuit types—such as GHZ, Mirror, Quantum Volume, and Randomized Benchmarking circuits—and comparing mitigated results against ideal theoretical outcomes. Additionally, Mitiq evaluates the overhead associated with each error mitigation technique, such as the increase in circuit depth or the number of samples required, as seen in methods like Zero Noise Extrapolation (ZNE) and Probabilistic Error Cancellation (PEC).


## GHZ Circuits

The {func}`.generate_ghz_circuit` create the GHZ states that are highly sensitive to noise. A [GHZ (Greenberger–Horne–Zeilinger)](https://en.wikipedia.org/wiki/Greenberger-Horne-Zeilinger_state) state is a maximally entangled quantum state involving multiple qubits. Thus, they make it easy to test error rates in entanglement creation and preservation, which is central for many quantum algorithms.

```{code-cell} ipython3
from mitiq.benchmarks import generate_ghz_circuit

circuit = generate_ghz_circuit(n_qubits=7)

print(circuit)
```

## Mirror Circuits

The {func}`.generate_mirror_circuit`, as defined in {cite}`Proctor_2021_NatPhys`, involves running a quantum circuit forward and then “mirroring” it (applying the reverse operations). Ideally, this results in returning the system to the initial state, so they’re great for testing if the noise mitigation is effective in preserving information through complex sequences.

```{code-cell} ipython3
from mitiq.benchmarks import generate_mirror_circuit
import networkx as nx

topology = nx.complete_graph(7) # Provide appropriate topology
circuit, correct_bitstring = generate_mirror_circuit(nlayers=7, two_qubit_gate_prob=1.0, connectivity_graph=topology, return_type="cirq")

print(circuit)
```

## Quantum Volume Circuits

The {func}`.generate_quantum_volume_circuit`, as defined in {cite}`Cross_2019_Validating`, tests the maximum achievable "volume" or computational capacity of a quantum processor. Running these circuits with error mitigation tests if mitiq’s techniques improve the effective quantum volume.

```{code-cell} ipython3
from mitiq.benchmarks import generate_quantum_volume_circuit

circuit,_ = generate_quantum_volume_circuit(num_qubits=4, depth=7)

print(circuit)
```

## Mirror Quantum Volume Circuits

The {func}`.generate_mirror_qv_circuit`, as defined in {cite}`Amico_2023_arxiv`, is designed to test [Quantum Volume](https://en.wikipedia.org/wiki/Quantum_volume), a metric combining circuit depth, number of qubits, and fidelity. These circuits run a quantum circuit forward and then “mirroring” it to check whether error mitigation techniques help achieve higher effective quantum volumes on noisy devices.

```{code-cell} ipython3
from mitiq.benchmarks import generate_mirror_qv_circuit

circuit = generate_mirror_qv_circuit(num_qubits=7, depth=2)

print(circuit)
```

## Quantum Phase Estimation Circuits

The {func}`.generate_qpe_circuit`, as defined in [Quantum phase estimation algorithm](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm) is used to the measure eigenvalues of unitary operators. Since accurate phase estimation requires precise control over operations, these circuits test the mitigation techniques’ ability to handle small noise effects over multiple gate sequences.

```{code-cell} ipython3
from mitiq.benchmarks import generate_qpe_circuit

circuit = generate_qpe_circuit(evalue_reg=7)

print(circuit)
```

## Randomized Benchmarking Circuits

The {func}`.generate_rb_circuits` are sequences of random gates (generally Clifford gates), to estimate an average error rate. They’re standard in benchmarking for evaluating how well mitiq’s error mitigation reduces this error rate across different levels of noise.

```{code-cell} ipython3
from mitiq.benchmarks import generate_rb_circuits

circuits = generate_rb_circuits(n_qubits=1, num_cliffords=5)
circuit=circuits[0]

print(circuit)
```

## Rotated Randomized Benchmarking Circuits

The {func}`.generate_rotated_rb_circuits` are sequences of random gates similar to {func}`.generate_rb_circuits`, but with rotations added, that allows assessment of errors beyond just the standard Clifford gates. They’re useful to check how well Mitiq handles noise in scenarios with more diverse gates.

```{code-cell} ipython3
from mitiq.benchmarks import generate_rotated_rb_circuits

circuits = generate_rotated_rb_circuits(n_qubits=1, num_cliffords=5)
circuit=circuits[0]

print(circuit)
```

## Randomized Clifford+T Circuits

The {func}`.generate_random_clifford_t_circuit` add the T gate to the standard Clifford set, adding more complex operations to the random benchmarking. This type evaluates Mitiq’s performance with gate sets that go beyond the Clifford gates, crucial for fault-tolerant computing.

```{code-cell} ipython3
from mitiq.benchmarks import generate_random_clifford_t_circuit

circuit = generate_random_clifford_t_circuit(num_qubits=7, num_oneq_cliffords=2, num_twoq_cliffords=2, num_t_gates=2)

print(circuit)
```

## W State Circuits

The {func}`.generate_w_circuit` are entangled circuits that distribute the entanglement across qubits differently than GHZ states. Testing with W state circuits can help explore how well a device maintains distributed entanglement in noisy environments.

A generalized multipartite $N$-qubit W-state is defined in equation {math:numref}`w_state`:

$$
\ket{W_N} = \frac{1}{\sqrt{N}} \left( \ket{100 \dots 0} + \ket{010 \dots 0} + \dots + \ket{0 \dots 01}\right)
$$(w_state)

Such a $N$-qubit W-state circuit can be generated through {func}`.generate_w_circuit` as defined in
{cite}`Cruz_2019_Efficient`. The construction relies on an initial state $\ket{10 \dots 0}$ and a fundamental building block $B(p)$ such that

$$
B(p) \ket{00} = \ket{00} , \,
B(p) \ket{10} = \sqrt{p} \ket{10} + \sqrt{1-p} \ket{01}
$$

This building block comprises of a controlled $G(p)$ and an inverted CNOT where $0 < p < 1$.

$$
G(p) = \begin{pmatrix}
\sqrt{p} & -\sqrt{1-p} \\
\sqrt{1-p} & \sqrt{p}
\end{pmatrix}
$$


```{code-cell} ipython3
from mitiq.benchmarks import generate_w_circuit

circuit = generate_w_circuit(n_qubits=4)

print(circuit)
```
We can also verify the final state of the circuit is equivalent to $\ket{W_4}$.

$$
\ket{W_4} = \frac{1}{\sqrt{4}} \left( \ket{1000} + \ket{0100} + \ket{0010} +  \ket{0001}\right)
$$

```{code-cell} ipython3
import cirq 

w4_state_vector_transpose = (
        cirq.Simulator()
        .simulate(circuit, initial_state=1000)
        .final_state_vector)
```