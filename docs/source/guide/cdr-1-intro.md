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

# How do I use CDR?

Here we show how to use CDR by means of a simple example.

```{code-cell} ipython3
import numpy as np
import warnings
warnings.simplefilter("ignore", np.exceptions.ComplexWarning)

import cirq
from mitiq import cdr, Observable, PauliString
```

## Problem setup

To use CDR, we call {func}`.cdr.execute_with_cdr` with four "ingredients":

1. A quantum circuit to prepare a state $\rho$.
1. A quantum computer or noisy simulator to return a {class}`mitiq.QuantumResult` from $\rho$.
1. An observable $O$ which specifies what we wish to compute via $\mathrm{Tr} ( \rho O )$.
1. A near-Clifford (classical) circuit simulator.

### 1. Define a quantum circuit

The quantum circuit can be specified as any quantum circuit supported by Mitiq but
**must be compiled into a gateset in which the only non-Clifford gates are
single-qubit rotations around the $Z$ axis: $R_Z(\theta)$**. For example:

$$\{ \sqrt{X}, R_Z(\theta), \text{CNOT}\},$$
$$\{{R_X(\pi/2)}, R_Z(\theta), \text{CZ}\},$$
$$\{H, S, R_Z(\theta), \text{CNOT}\},$$
$$ \dots$$

In the next cell we define (as an example) a quantum circuit which contains some
Clifford gates and some non-Clifford $R_Z(\theta)$ rotations.

```{code-cell} ipython3
a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H.on(a),  # Clifford
    cirq.H.on(b),  # Clifford
    cirq.rz(1.75).on(a),
    cirq.rz(2.31).on(b),
    cirq.CNOT.on(a, b),  # Clifford
    cirq.rz(-1.17).on(b),
    cirq.rz(3.23).on(a),
    cirq.rx(np.pi / 2).on(a),  # Clifford
    cirq.rx(np.pi / 2).on(b),  # Clifford
)

# CDR works better if the circuit is not too short. So we increase its depth.
circuit = 5 * circuit
```

### 2. Define an executor

We define an [executor](executors.md) function which inputs a circuit and returns a {class}`.QuantumResult`.
Here for sake of example we use a simulator that adds single-qubit depolarizing noise after each layer and returns the final density matrix.

```{code-cell} ipython3
from mitiq.interface.mitiq_cirq import compute_density_matrix

compute_density_matrix(circuit).round(3)
```

### 3. Observable

As an example, assume that we wish to compute the expectation value $\mathrm{Tr} [ \rho O ]$ of the following observable $O$:

```{code-cell} ipython3
obs = Observable(PauliString("ZZ"), PauliString("X", coeff=-1.75))
print(obs)
```

```{warning}
To apply CDR the {class}`.Observable` must be Hermitian (i.e. it must always produce real expectation values).
```

You can read more about the {class}`.Observable` class in the [documentation](observables.md).

### 4. (Near-Clifford) Simulator

The CDR method creates a set of "training circuits" which are related to the input circuit and are efficiently simulable.
These circuits are simulated on a classical (noiseless) simulator to collect data for regression.
The simulator should also return a `QuantumResult`.

To use CDR at scale, an efficient near-Clifford circuit simulator must be specified. Two examples of efficient near-Clifford circuit simulators are [Qrack with the stabilizer hybrid (Clifford+Rz)](https://github.com/vm6502q/pyqrack-jupyter/blob/main/Clifford_RZ.ipynb)  and [IBM's extended stabilizer simulator](https://qiskit.github.io/qiskit-aer/tutorials/6_extended_stabilizer_tutorial.html).

In this example, the circuit is small enough to use any classical simulator, and we use the same density matrix simulator as above but without noise.

```{code-cell} ipython3
def simulate(circuit: cirq.Circuit) -> np.ndarray:
    return compute_density_matrix(circuit, noise_level=(0.0,))

simulate(circuit).round(3)
```

## Run CDR

Now we can run CDR.
We first compute the noiseless result then the noisy result to compare to the mitigated result from CDR.

### The ideal result

```{code-cell} ipython3
ideal_expval = obs.expectation(circuit, simulate).real
print(f"ideal expectation value: {ideal_expval:.2f}")
```

### The noisy result

We now generate the noisy result.
Note that {func}`.compute_density_matrix` function by default runs a simulation with noise.

```{code-cell} ipython3
unmitigated_expval = obs.expectation(circuit, compute_density_matrix).real
print(f"unmitigated expectation value: {unmitigated_expval:.2f}")
```

### The mitigated result

```{code-cell} ipython3
mitigated_expval = cdr.execute_with_cdr(
    circuit,
    compute_density_matrix,
    observable=obs,
    simulator=simulate,
    random_state=0,
).real
print(f"mitigated expectation value: {mitigated_expval:.2f}")
```

```{code-cell} ipython3
unmitigated_error = abs(unmitigated_expval - ideal_expval)
mitigated_error = abs(mitigated_expval - ideal_expval)

print(f"Unmitigated Error:           {unmitigated_error:.2f}")
print(f"Mitigated Error:             {mitigated_error:.2f}")

improvement_factor = unmitigated_error / mitigated_error
print(f"Improvement factor with CDR: {improvement_factor:.2f}")
```
