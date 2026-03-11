---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How do I use PEC?

To apply PEC, you must know how to represent *ideal operations* as linear combinations of *noisy implementable operations*.
This information is exploited by Mitiq to remove the bias error from noisy expectation values.

- By *ideal operations*, we mean noiseless unitary gates applied to specific qubits.
- By *noisy implementable operations*, we mean those noisy gates that can be applied to specific qubits by a real backend.

Since real backends are not perfect, the noisy implementable operations do not have an ideal unitary effect and typically correspond
to non-unitary quantum channels.

**Note:** In this documentation we often interchange the terms *gate* and *operation* for simplicity. However, in the source code of Mitiq,
we follow the notation in which an operation is a gate applied to specific qubits, while a gate represents an abstract (qubit-independent)
operation.

+++

PEC works in two main stages: generate noise-scaled circuits via inserting quasi-probability representations of noisy gates, and combining the resulting measurements post-execution.

The section [Run PEC](#run-pec) applies the protocol in a single step, and then in the section [Step by step application of PEC](#step-by-step-application-of-pec), we'll show how you can apply the technique stepwise.

This workflow can be executed by a single call to {func}`.execute_with_pec`.
If more control is needed over the protocol, Mitiq provides {func}`.pec.construct_circuits` and {func}`.pec.combine_results` to handle the first and second steps respectively.

As with all techniques, PEC is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

In the next cell you can select the frontend used in this tutorial. For example:

```{code-cell} ipython3
frontend = "qiskit"
```

## Problem setup

+++

We first define the circuit of interest. For simplicity, in this example we use
a single-qubit randomized-benchmarking circuit whose ideal (noiseless) execution is equivalent to the
identity operation.

```{code-cell} ipython3
from mitiq import benchmarks

circuit = benchmarks.generate_rb_circuits(
  n_qubits=1, num_cliffords=2, return_type=frontend,
)[0]

print(circuit)
```

In the code cell above, we used the function {func}`~mitiq.benchmarks.randomized_benchmarking.generate_rb_circuits()`, in which `num_cliffords` is the
number of random Clifford group elements and is proportional to the depth of the circuit.

+++

We now define an [executor](executors.md) which simulates a backend with depolarizing noise and computes the survival probability of the $|00\rangle$ state.

+++

**Note:** The body of the following function is supposed to be written by the user and depends on the specific
frontend and backend. Here, for simplicity, we first convert the input circuit to the Mitiq internal
representation (Cirq) and then simulate it.

```{code-cell} ipython3
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

def execute(circuit, noise_level=0.01):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    # Replace with code based on your frontend and backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    # We add a simple noise model to simulate a noisy backend.
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

The function `execute` can be used to evaluate noisy (unmitigated) expectation values.

```{code-cell} ipython3
# Compute the expectation value of the |0⟩⟨0| observable.
noisy_value = execute(circuit)
ideal_value = execute(circuit, noise_level=0.0)
print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.5f}")
```

We now show how to use PEC to reduce this error.

+++

## Applying probabilistic error cancellation (PEC)

+++

### Represent ideal gates as linear combinations of noisy gates

+++

To apply PEC, we need to represent some or all of the operations of the ideal `circuit` as linear combinations
of noisy implementable operations. Such linear combinations are also known as quasi-probability representations (see [What is the theory behind PEC?](pec-5-theory.md)) and, in Mitiq, they correspond to {class}`.OperationRepresentation` objects. More information on the {class}`.OperationRepresentation` class can be found in the section [What additional options are available in PEC?](pec-3-options.md).

But how can we obtain the quasi-probability representations that are appropriate for a given backend?
There are two main alternative scenarios.

- **Case 1:** The noise of the backend can be approximated by a simple noise model, such that quasi-probability representations can be analytically computed {cite}`Temme_2017_PRL, Sun_2021_PRAppl, Takagi_2020_PRR`. For example, this is possible for depolarizing or amplitude damping noise.

- **Case 2:** The noise of the backend is too complex and cannot be approximated by a simple noise model.

Depending on the previous two cases, the method to obtain quasi-probability representations is different.

- **Method for case 1:** A simple noise model (e.g. depolarizing or amplitude damping) is typically characterized by a single `noise_level`
parameter (or a few parameters) that can be experimentally estimated. Possible techniques for estimating the noise level are randomized-benchmarking
experiments or calibration experiments. Often, gate error probabilities are reported by hardware vendors and can be used to obtain a good guess
for the `noise_level` without running any experiments. Given the noise model and the `noise_level`, one can apply known analytical expressions to
compute the quasi-probability representations of arbitrary gates {cite}`Temme_2017_PRL, Takagi_2020_PRR`.

- **Method for case 2:** Assuming an over-simplified noise model may be a bad approximation. In this case, the suggested approach is to
perform the complete process tomography of a basis set of implementable noisy operations (e.g. the native gate set of the backend).
One could also use *gate set tomography* (GST), a noise characterization technique which is robust to state-preparation and measurement errors.
Given the superoperators of the noisy implementable operations, one can obtain the quasi-probability representations as solutions of
numerical optimization problems {cite}`Temme_2017_PRL, Sun_2021_PRAppl`. In Mitiq, this is possible through the
{func}`.find_optimal_representation()` function as shown the section [What additional options are available in PEC?](pec-3-options.md).

+++

In this example, the `execute` function simulates the effect of depolarizing noise of some fixed `noise_level` acting after each gate. Therefore, we are in the first scenario defined above and we can generate a list of   {class}`.OperationRepresentation` objects associated to all the gates of the `circuit` as follows:

```{code-cell} ipython3
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise

noise_level = 0.01
reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level)
print(f"{len(reps)} OperationRepresentation objects produced, assuming {100 * noise_level}% depolarizing noise.")
```

As an example, we print the first {class}`.OperationRepresentation` in `reps`, showing how an ideal gate can be expressed as a linear combination of noisy gates:

```{code-cell} ipython3
print(reps[0])
```

**IMPORTANT:** For PEC to work properly, the noise model and the `noise_level`
associated to the `execute` function must correspond to those used to compute the
{class}`.OperationRepresentation` objects.

+++

### Run PEC

+++

Once the necessary {class}`.OperationRepresentation`s have been defined,
probabilistic error cancellation can be applied through the {func}`.execute_with_pec` function as follows:

```{code-cell} ipython3
from mitiq import pec

pec_value = pec.execute_with_pec(circuit, execute, representations=reps)

print(f"Error without PEC: {abs(ideal_value - noisy_value) :.5f}")
print(f"Error with PEC:    {abs(ideal_value - pec_value) :.5f}")
```

As printed above, PEC reduced the error compared to the unmitigated case.

## Step by step application of PEC

This section demonstrates the use of the {func}`.pec.construct_circuits` for those who want to generate and see a list of sampled circuits based on the given quasi-probability representaions.

### Generate Sample Circuits
We will now generate a list of sampled circuits. Note that the number of sampled circuits generated depends on the input provided, which can be seen using the function `len`.

```{code-cell} ipython3
sampled_circuits = pec.construct_circuits(circuit, representations=reps)

print(f"Number of sample circuits:    {len(sampled_circuits)}")
print(sampled_circuits[0])
```

Now that we have many circuits, we can inspect them (or even change them if desired).
We can then execute the circuits and store the results in a list, which can be used by the {func}`.pec.combine_results` to get a combined result.

### Combine the results

We will now get the combined result of the list of circuits generated.

```{code-cell} ipython3
results = [execute(circuit) for circuit in sampled_circuits]

combined_result = pec.combine_results([results], 1 , [1,1,1])

print(f"Error with single-step PEC: {abs(ideal_value - pec_value) :.5f}")
print(f"Error with multi-step PEC:    {abs(ideal_value - combined_result) :.5f}")
```

As you can see above, the multi-step PEC gives the same the error as the single step PEC error using the function {func}`.execute_with_pec`.