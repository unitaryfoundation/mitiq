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
```{tags} rem, zne, pt, ddd, cirq, intermediate
```

# Composing techniques: Combining 4 Error Mitigation Techniques

Applying multiple error mitigation techniques can be beneficial for mitigating as much noise in quantum computers as possible. It is possible to apply more than one pair of techniques for error mitigation. 

In this example, we demonstrate how to apply Pauli Twirling (PT), Dynamical Decoupling (DDD), Readout Error Mitigation (REM) and Zero-noise extrapolation (ZNE) to a benchmarking circuit. More information on these techniques, including examples of how pairs of these techniques can be applied together, can be found in the corresponding sections of the user guide (linked
above).

+++

##Setup

We start by importing the relevant modules and libraries required for the rest of this tutorial.

```{code-cell} ipython3
import cirq
import random
import numpy as np
from mitiq.benchmarks import generate_rb_circuits
from mitiq import MeasurementResult, Observable, PauliString, raw
```

##Pauli Twirling (PT)
PT, by itself, is not considered an error mitigation technique. Rather, it converts coherent noise in the quantum circuit into incoherent. We can try applying this technique onto a GHZ circuit, like so:

```{code-cell} ipython3
# Define qubits
qubits = cirq.LineQubit.range(3)

# Build GHZ circuit: (|000⟩ + |111⟩)/√2
ghz_circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
)
```

```{code-cell} ipython3
ghz_circuit
```

```{code-cell} ipython3
def pauli_twirl_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    new_ops = []

    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.CNotPowGate):
            a, b = op.qubits
            # Sample random Paulis
            P1 = random.choice([cirq.I, cirq.X, cirq.Y, cirq.Z])
            P2 = random.choice([cirq.I, cirq.X, cirq.Y, cirq.Z])
            inv_P1 = P1**-1
            inv_P2 = P2**-1

            new_ops += [P1(a), P2(b), op, inv_P1(a), inv_P2(b)]
        else:
            new_ops.append(op)

    return cirq.Circuit(new_ops)
```


```{code-cell} ipython3
twirled_ghz = pauli_twirl_circuit(ghz_circuit)
print("Twirled GHZ Circuit:")
print(twirled_ghz)
```

## DDD
Dynnamical noise decoupling inserts idle gate sequences to suppress decoherence. This is best applied after PT to maintain structure.

We define an executor function that returns an expectation value:

```{code-cell} ipython3
def executor(circuit: cirq.Circuit) -> float:
    qubits = sorted(circuit.all_qubits())
    circuit_meas = circuit.copy()
    circuit_meas.append(cirq.measure(*qubits, key='m'))
    
    simulator = cirq.Simulator()
    result = simulator.run(circuit_meas, repetitions=1000)
    bitstrings = result.measurements['m']
    
    # Compute expectation of Z0 * Z1 = parity of first two qubits
    values = [(-1) ** (b[0] ^ b[1]) for b in bitstrings]
    expectation = sum(values) / len(values)
    
    return expectation
```
```{code-cell} ipython3

# No observables used

ddd_result = execute_with_ddd(
    circuit=twirled_ghz,
    executor=executor,
    observable=None,   
    rule=rules.xx,
    num_trials=1,
    full_output=True
)
```
```{code-cell} ipython3
ddd_result
```
```
(-1.0,
 {'ddd_value': -1.0,
  'ddd_trials': [-1.0],
  'circuits_with_ddd': [0: ───H───X───@───X───I───X───X───
              │
1: ───Z───────X───Z───X───@───X───
                          │
2: ───Y───I───X───X───I───X───Y───]})
```
```{code-cell} ipython3
ddd_circuit = ddd_result[1]['circuits_with_ddd'][0]
ddd_circuit.append(cirq.measure(*ddd_circuit.all_qubits(), key='m'))

```
```
ddd_circuit
```
```
0: ───H───X───@───X───I───X───X───M('m')───
              │                   │
1: ───Z───────X───Z───X───@───X───M────────
                          │       │
2: ───Y───I───X───X───I───X───Y───M────────
```

## Readout Error Mitigation (REM)

```{code-cell} ipython3
def execute(circuit: cirq.Circuit, noise_level: float = 0.002, p0: float = 0.05) -> MeasurementResult:
    """Execute a circuit with depolarizing noise and readout bit-flip errors on measured qubits."""
    measurements = circuit[-1]
    circuit = circuit[:-1]
    circuit = circuit.with_noise(cirq.depolarize(noise_level))

    measured_qubits = list(measurements.qubits)
    circuit.append(cirq.bit_flip(p0).on_each(measured_qubits))

    circuit.append(measurements)

    simulator = cirq.DensityMatrixSimulator()

    result = simulator.run(circuit, repetitions=10000)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)

```

```{code-cell} ipython3
qubits = [cirq.LineQubit(i) for i in range(3)]
cirq_ps = cirq.PauliString({qubits[0]: cirq.Z, qubits[2]: cirq.Z})  # Note: qubit 1 with identity omitted

spec = "".join(str(cirq_ps[q]) if q in cirq_ps else 'I' for q in qubits)
support = tuple(range(len(qubits)))

obs = Observable(PauliString(spec, support=support))
print(obs)

```
```{code-cell} ipython3
noisy = raw.execute(ddd_circuit, execute, obs)
```
```{code-cell} ipython3
from functools import partial

ideal = raw.execute(ddd_circuit, partial(execute, noise_level=0, p0=0), obs)
```

```{code-cell} ipython3
print("Unmitigated value:", "{:.12f}".format(noisy.real))
```

