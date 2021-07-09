---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Clifford data regression API

+++

This example shows how to use Clifford data regression (CDR) by means of a simple example.

```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import cirq
from mitiq import cdr
```

## Setup

+++

To use CDR, we call `cdr.execute_with_cdr` with four necessary "ingredients":

1. A quantum circuit to prepare a state $\rho$.
1. A quantum computer or noisy simulator to sample bitstrings from $\rho$.
1. One or more observables $O$ which specify what we wish to compute: $\text{Tr} [ \rho O ]$.
1. A near-Clifford (classical) circuit simulator.

+++

### (1) Circuit

+++

The circuit can be specified as any quantum circuit supported by Mitiq but **must be compiled into the gateset $\{ \sqrt{X}, Z, \text{CNOT}\}$.**

```{code-cell} ipython3
a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.rx(0.1).on(a),
    cirq.rx(-0.72).on(b),
    cirq.rz(0.4).on(a),
    cirq.rz(0.2).on(b),
    cirq.CNOT.on(a, b),
    cirq.rx(-0.1).on(b),
    cirq.rz(-0.23).on(a),
    cirq.CNOT.on(b, a),
    cirq.rx(-0.112).on(a),
    cirq.measure(a, b, key="z"),
)
circuit
```

### (2) Executor

+++

The executor inputs a circuit and returns a dictionary or counter of computational basis measurements. **The return type must be `Dict[int, int]` or `Counter[int]`.** 

Typically this function will send the circuit to a quantum computer and wait for the results. Here for sake of example we use a noisy simulator.

```{code-cell} ipython3
from typing import Counter


def sample_bitstrings(circ: cirq.Circuit, shots: int = 1000, noise: float = 0.01) -> Counter[int]:
    # Add depolarizing noise to emulate a noisy quantum processor!
    circuit = circ.with_noise(cirq.depolarize(noise))
    
    return cirq.DensityMatrixSimulator().run(circuit, repetitions=shots).histogram(key="z")
```

An example of calling this function is shown below.

```{code-cell} ipython3
sample_bitstrings(circuit)
```

### (3) Observable(s)

+++

The observables $O$ indicate what we wish to compute via $\text{Tr} [ \rho O ]$ and **must be specified as "diagonal" (one-dimensional) NumPy arrays.**

```{code-cell} ipython3
# Observable(s) to measure.
z = np.diag([1, -1])
obs = np.diag(np.kron(z, z))
```

### (4) (Near-clifford) Simulator

+++

The CDR method creates a set of "training circuits" which are related to the input circuit and are efficiently simulable. These circuits are simulated on a classical (noiseless) simulator to collect data for regression. **The simulator return type must be the same as the executor return type.**

To use CDR at scale, an efficient near-Clifford circuit simulator must be specified. In this example, the circuit is small enough to use any classical simulator.

```{code-cell} ipython3
def sample_bitstrings_simulator(circuit, shots: int = 1000) -> Counter:
    return sample_bitstrings(circuit, shots=shots, noise=0.0)
```

## Results

+++

Now we can run CDR. We first compute the noiseless result then the noisy result to compare to the mitigated result from CDR.

+++

### The noiseless result

```{code-cell} ipython3
cdr.execute.calculate_observable(
    state_or_measurements=sample_bitstrings_simulator(circuit),
    observable=obs,
)
```

### The noisy result

```{code-cell} ipython3
cdr.execute.calculate_observable(
    state_or_measurements=sample_bitstrings(circuit),
    observable=obs,
)
```

### The mitigated result

```{code-cell} ipython3
cdr.execute_with_cdr(
    circuit=circuit,
    executor=sample_bitstrings,
    observables=[obs],
    simulator=sample_bitstrings_simulator,
)
```

## Additional options

+++

In addition to the four necessary arguments shown above, there are additional parameters in CDR.

+++

### Training circuits

+++

One option is how many circuits are in the training set (default is 10). This can be changed as follows.

```{code-cell} ipython3
cdr.execute_with_cdr(
    circuit=circuit,
    executor=sample_bitstrings,
    observables=[obs],
    simulator=sample_bitstrings_simulator,
    num_training_circuits=5,
)
```

Another option is which fit function to use for regresstion (default is `cdr.linear_fit_function`).

+++

### Fit function

```{code-cell} ipython3
cdr.execute_with_cdr(
    circuit=circuit,
    executor=sample_bitstrings,
    observables=[obs],
    simulator=sample_bitstrings_simulator,
    fit_function=cdr.linear_fit_function_no_intercept,
)
```

### Variable noise CDR

+++

The circuit + training circuits can also be run at different noise scale factors to implement [Variable noise Clifford data regression](https://arxiv.org/abs/2011.01157).

```{code-cell} ipython3
from mitiq.zne import scaling

cdr.execute_with_cdr(
    circuit=circuit,
    executor=sample_bitstrings,
    observables=[obs],
    simulator=sample_bitstrings_simulator,
    scale_factors=(1, 3),
    scale_noise=scaling.fold_gates_at_random,
)
```
