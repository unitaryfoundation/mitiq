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

In this example, we demonstrate a comprehensive error mitigation pipeline using:

1. Pauli Twirling (PT), for tailoring noise from coherent to incoherent, 

2. Dynamical Decoupling (DDD), that inserts idle gate sequences for reducing interactions with the environment,

3. Readout Error Mitigation (REM), for classical correction post-measurement,

4. Zero-Noise Extrapolation (ZNE), for extrapolation to noise-free observable values.

More information on these techniques, including examples of how pairs of these techniques can be applied together, can be found in the corresponding sections of the user guide.

+++

##Setup

We start by importing some of the modules and libraries we need for this tutorial:

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

```
0: ───H───@───────
          │
1: ───────X───@───
              │
2: ───────────X───
```

```{code-cell} ipython3
import random

def pauli_twirl_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    new_ops = []

    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.CNotPowGate):
            a, b = op.qubits
            # Sample random Pauli gates
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

We are now ready to apply our noise models.

## DDD
Dynnamical noise decoupling inserts idle gate sequences to suppress decoherence. This is best applied after PT to maintain structure.

We define an executor function that returns an expectation value:

```{code-cell} ipython3
def executor(circuit: cirq.Circuit) -> float:
    # Add measurements to all qubits in the circuit
    qubits = sorted(circuit.all_qubits())
    circuit_meas = circuit.copy()
    circuit_meas.append(cirq.measure(*qubits, key='m'))
    
    # Run the circuit with sampling
    simulator = cirq.Simulator()
    result = simulator.run(circuit_meas, repetitions=1000)
    bitstrings = result.measurements['m']
    
    # Compute expectation of Z0 * Z1 = parity of first two qubits
    values = [(-1) ** (b[0] ^ b[1]) for b in bitstrings]
    expectation = sum(values) / len(values)
    
    return expectation
```
```{code-cell} ipython3
from mitiq.ddd import execute_with_ddd, rules

#No observables

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
  'circuits_with_ddd': [0: ───H───Y───@───Y───I───X───X───
              │
1: ───X───────X───X───I───@───I───
                          │
2: ───X───I───X───X───I───X───X───]})
```
```{code-cell} ipython3
ddd_circuit = ddd_result[1]['circuits_with_ddd'][0]
ddd_circuit.append(cirq.measure(*ddd_circuit.all_qubits(), key='m'))
```
```
ddd_circuit
```
```
0: ───H───Y───@───Y───I───X───X───M('m')───
              │                   │
1: ───X───────X───X───I───@───I───M────────
                          │       │
2: ───X───I───X───X───I───X───X───M────────
```

## Combining Readout Error Mitigation (REM) and Zero Noise Extrapolation (ZNE)

```{code-cell} ipython3
def execute(circuit: cirq.Circuit, noise_level: float = 0.002, p0: float = 0.05) -> MeasurementResult:
    measurements = circuit[-1]
    circuit = circuit[:-1]
    circuit = circuit.with_noise(cirq.depolarize(noise_level))
    circuit.append(cirq.bit_flip(p0).on_each(circuit.all_qubits()))
    circuit.append(measurements)

    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(circuit, repetitions=10000)

    # Extract bitstrings in correct order
    measurement_key = list(result.measurements.keys())[0]  # e.g., "m"
    bitstrings = result.measurements[measurement_key]
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

```
Unmitigated value: -0.959200000000
```

```{code-cell} ipython3
p0 = p1 = 0.05  # Your readout error probabilities

num_measured_qubits = 3
icm = rem.generate_inverse_confusion_matrix(num_measured_qubits, p0, p1)

rem_executor = rem.mitigate_executor(execute, inverse_confusion_matrix=icm)

combined_executor = zne.mitigate_executor(rem_executor, observable=obs, scale_noise=zne.scaling.folding.fold_global)

combined_result = combined_executor(ddd_circuit)
print("Mitigated value obtained with REM + ZNE:", "{:.12f}".format(combined_result.real))

```

```
Mitigated value obtained with REM + ZNE: -1.000000000000
```