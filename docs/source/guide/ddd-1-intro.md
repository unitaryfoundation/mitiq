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

# How do I use DDD?
DDD works in two main stages: generate noise-scaled circuits by inserting DDD sequences, and combining the resulting measurements post-execution.

The section [Apply DDD](#apply-ddd) applies the protocol in a single step, and then in the section [Step by step application of DDD](#step-by-step-application-of-ddd), we’ll show how you can apply the technique stepwise.

This workflow can be executed by a single call to {func}`.execute_with_ddd`.
If more control is needed over the protocol, Mitiq provides {func}`.ddd.construct_circuits` and {func}`.ddd.combine_results` to handle the first and second steps respectively.

As with all techniques, DDD is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```


## Problem setup
We first define the circuit of interest. In this example, the circuit has a
slack window with a length of 4 (in the sense that 4 single-qubit gates can fit in that window).

```{code-cell} ipython3
from cirq import LineQubit, Circuit, rx, rz, CNOT

a, b = LineQubit.range(2)
circuit = Circuit(
    rx(0.1).on(a),
    rx(0.1).on(a),
    rz(0.4).on(a),
    rx(-0.72).on(a),
    rz(0.2).on(a),
    rx(-0.8).on(b),
    CNOT.on(a, b),
)

print(circuit)
```

Next we define a simple executor function which inputs a circuit, executes
the circuit on a noisy simulator, and returns the probability of the ground
state. See the [Executors](executors.md) section for more information on
how to define more advanced executors.

```{code-cell} ipython3
import numpy as np
from cirq import DensityMatrixSimulator, amplitude_damp
from mitiq.interface import convert_to_mitiq

def execute(circuit, noise_level=0.1):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with amplitude damping noise.
    """
    # Replace with code based on your frontend and backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    noisy_circuit = mitiq_circuit.with_noise(amplitude_damp(gamma=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

The [executor](executors.md) can be used to evaluate noisy (unmitigated)
expectation values.

```{code-cell} ipython3
# Compute the expectation value of the |0><0| observable.
noisy_value = execute(circuit)
ideal_value = execute(circuit, noise_level=0.0)
print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.3}")
```

## Select the DDD sequences to be applied
We now import a DDD _rule_ from Mitiq, i. e., a function that generates DDD sequences of different length.
In this example, we opt for YY sequences (pairs of Pauli Y operations).
```{code-cell} ipython3
from mitiq import ddd

rule = ddd.rules.yy
```

## Apply DDD
Digital dynamical decoupling can be easily implemented with the function
{func}`.execute_with_ddd()`.

```{code-cell} ipython3
mitigated_result = ddd.execute_with_ddd(
    circuit=circuit,
    executor=execute,
    rule=rule,
)
```

```{code-cell} ipython3
print(f"Error with mitigation (DDD): {abs(ideal_value - mitigated_result) :.3}")
```

Here we observe that the application of DDD reduces the estimation error when compared
to the unmitigated result.

```{admonition} Note:
DDD is designed to mitigate noise that has a finite correlation time. For the
simple Markovian noise simulated in this example, DDD can still have a
non-trivial effect on the final error, but it is not always a positive effect.
For example, one can check that by changing the parameters of the input circuit,
the error with DDD is sometimes larger than the unmitigated error.
```

## Step by step application of DDD

In this section we demonstrate the use of {func}`.ddd.construct_circuits` for those who might want to generate circuits with DDD sequences inserted, and have more control over the protocol.

### Generating circuits with DDD sequences

Here we will generate a list of circuits with DDD sequences inserted, which will later be passed to the executor. The number of circuits generated can be checked using the `len` function.

```{code-cell} ipython3
circuits_with_ddd = ddd.construct_circuits(circuit=circuit, rule=rule, num_trials=10)

print(f"Number of sample circuits:    {len(circuits_with_ddd)}")
print(circuits_with_ddd[0])
```

Now that we have many circuits, we can inspect them (or even change them if desired).
We can then execute the circuits and store the results in a list, which can be used by the {func}`.ddd.combine_results` to get a combined result.

### Combine the results

We will now get the combined result of the list of circuits generated.

```{code-cell} ipython3
results = [execute(circuit) for circuit in circuits_with_ddd]
combined_result = ddd.combine_results(results)

print(f"Error with single-step DDD: {abs(ideal_value - mitigated_result) :.5f}")
print(f"Error with multi-step DDD:  {abs(ideal_value - combined_result) :.5f}")
```

As you can see above, the multi-step DDD gives the same the error as the single step DDD error using the function {func}`.execute_with_ddd`.

+++

The section
[What additional options are available when using DDD?](ddd-3-options.md)
contains information on more advanced ways of applying DDD with Mitiq.
