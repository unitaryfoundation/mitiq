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

# What happens when I use PEC?

When applying PEC with Mitiq, the workflow of quantum and classical data is represented in the figure below.

![pec_figure](../img/pec_workflow2_steps.png)

- The user provides a `QPROGRAM`, (i.e. a quantum circuit defined via any of the supported frontends).
- Mitiq probabilistically generates a list of auxiliary circuits. This step depends on the quasi-probability representations provided by the user.
- The generated circuits are executed via a user-defined [Executor](executors.md).
- Mitiq infers an unbiased estimate of the ideal expectation value from a classical post-processing of the measured data.
- The error mitigated expectation value is returned to the user.

The general workflow of PEC is similar to the workflow of ZNE. The main difference is that in ZNE the auxiliary
circuits are obtained by noise scaling, while
in PEC they are probabilistically generated. As a consequence, the final inference step is different too.
The inference step of PEC is based on the Monte Carlo estimation protocol discussed in [What is the theory behind PEC?](pec-5-theory.md)

+++

As shown in the Section [How do I use PEC?](pec-1-intro.md), the standard way of applying PEC in Mitiq is based
on the function {func}`.execute_with_pec()`. However, one may be interested in applying PEC with a lower-level of
abstraction for reasons of efficiency or for customization proposes. 
In particular, it could be useful to explicitly split the PEC process into the two main steps shown in the figure above:

- Step 1: Probabilistic generation of all the auxiliary circuits;
- Step 2: Inference of the ideal expectation value from the noisy execution of the auxiliary circuits. 

+++

## Step 1: Probabilistic generation of all the auxiliary circuits

+++

### Define the circuit of interest and the quasi-probability representations

+++

We define the circuit of interest as we did in [How do I use PEC?](pec-1-intro.md)

```{code-cell} ipython3
import mitiq
from mitiq import benchmarks

frontend = "qiskit"  # Supported: "cirq", "qiskit", "pyquil", "braket", "pennylane".

circuit = benchmarks.generate_rb_circuits(
  n_qubits=1, num_cliffords=2, return_type = frontend,
)[0]
print(circuit)
```

We assume local depolarizing noise and we define the list of {class}`.OperationRepresentation`s (one for each gate)
as we did in [How do I use PEC?](pec-1-intro.md).

```{code-cell} ipython3
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise

noise_level = 0.01
reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level)
print(f"{len(reps)} OperationRepresentation objects produced, assuming {noise_level :.2%} depolarizing noise.")
```

Each element of `reps` is an {class}`.OperationRepresentation` object that represents an ideal gate
$\mathcal G_i$ of `circuit` as a linear combination of noisy operations $\{\mathcal O_{i, \alpha} \}$:

$$ \mathcal G_i = \sum_\alpha \eta_{i, \alpha} \mathcal O_{i, \alpha}, \quad \sum_\alpha \eta_{i, \alpha}
= 1, \quad \sum_\alpha |\eta_{i,\alpha}|=\gamma_i,$$

where $\{\eta_{i, \alpha}\}$ is a real quasi-probability distribution with one-norm $\gamma_i$.

For example, the first {class}`.OperationRepresentation` in reps is:

```{code-cell} ipython3
print(reps[0])
```

### Probabilistic generation of all the auxiliary circuits

+++

According to the [theory of PEC](pec-5-theory.md), to probabilistically generate an auxiliary circuit one should:
- Define an empty `sampled_circuit` to be populated with probabilistic operations $\mathcal O_{j,\alpha}$;
- Define an empty `gate_sign_list` to be populated with the sign values ${\rm sgn}(\eta_{j,\alpha})$;
- Start a loop over the ideal operations of `circuit`;
- For each gate of `circuit`, search for the corresponding {class}`.OperationRepresentation` in `reps`;
- Sample a noisy gate from the quasi-probability distribution of the ideal gate using `rep[j].sample()` as
   shown in [What additional options are available in PEC?](pec-3-options.md);
- Append the sampled gate to `sampled_circuit` and the corresponding sign to `gate_sign_list`;
- When the loop is completed, return `sampled_circuit` and the associated `sampled_sign` (which is the product of all
   the elements of `gate_sign_list`).

Instead of manually applying all the previous steps, one can call the Mitiq function {func}`.sample_circuit()` to probabilistically generate an integer number `num_samples` of auxiliary circuits.

```{code-cell} ipython3
from mitiq import pec

sampled_circuits, sampled_signs, one_norm = pec.sample_circuit(circuit, reps, num_samples=5, random_state=30)
for circuit in sampled_circuits:
    print(circuit)
print("Signs:", sampled_signs)
print("One-norm:", one_norm)
```

The above result contains:
1. The list of `num_samples` auxiliary circuits that have been probabilistically generated according to the quasi-probability distributions in `reps`.
2. the list of the sampled signs ($\pm 1$) associated to each auxiliary circuit.
3. The one-norm $\gamma$ of the global quasi-probability distribution of the full circuit.

+++

Let us increase the value of `num_samples` and unpack the result in three different variables:

```{code-cell} ipython3
sampled_circuits, sampled_signs, one_norm = pec.sample_circuit(
  circuit, reps, num_samples=200,
)
```

We are now ready for the second step of PEC, i.e., the inference
of the ideal expectation value from the noisy execution of the `sampled_circuits`.

+++

## Step 2: Inference of the ideal expectation value from the noisy execution of the auxiliary circuits

+++

### Noisy execution of all the auxiliary circuits

+++

To execute the auxiliary circuits we define the same `execute` function that we used in the Section ["How do I use PEC?"](pec-1-intro.md).
Actually, since we have a list of circuits to execute, we define a *batched* version of the function, i.e., a function that takes a list of
circuits as input and returns a list of expectation values. More details on batched executors can be found in the ["Executors"](pec-1-intro.md) section.

```{code-cell} ipython3
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

def batched_execute(circuits: list[mitiq.QPROGRAM], noise_level: float=0.01) -> list[float]:
    """Returns [Tr[ρ_1 |0⟩⟨0|], Tr[ρ_2 |0⟩⟨0|]... ] where ρ_j is the state prepared by the
    j_th circuit in the input argument "circuits".
    """
    # Replace with code based on your frontend and backend, possibly using a batched execution.
    expectation_values = []
    for circuit in circuits:
        mitiq_circuit, _ = convert_to_mitiq(circuit)
        noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
        rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
        expectation_values.append(rho[0, 0].real)
    return expectation_values
```

We also initialize a Mitiq {class}`.Executor` object. This step is optional but recommended to store/access executed circuits and results.

```{code-cell} ipython3
executor = mitiq.Executor(batched_execute, max_batch_size=100)
```
We execute all the auxiliary circuits generated in the previous section to obtain a list of noisy expectation values.

```{code-cell} ipython3
noisy_expectation_values = executor.evaluate(
    sampled_circuits, 
    force_run_all=False,  # Set True if shot noise is present in quantum results.
)

# Unique noisy expectation values associated to unique circuits in sampled_circuits
executor.quantum_results
```

Since expectation values are evaluated from the simulation of an exact density matrix, we executed equal circuits
only once by setting `force_run_all=False`.

**Note:** *If shot noise is present, one should instead use `force_run_all=True` to ensure the statistical
independence of the quantum results.*

```{code-cell} ipython3
print(f"{len(noisy_expectation_values)} noisy expectation values efficiently evaluated by executing only {len(executor.quantum_results)} unique circuits.")
```

Equivalently, but less efficiently, we could have done as follows:

```{code-cell} ipython3
noisy_expectation_values_direct = batched_execute(sampled_circuits)

assert noisy_expectation_values_direct == noisy_expectation_values
```

### Estimate the ideal expectation value from the noisy results

+++

According to the [theory of PEC](pec-5-theory.md), the ideal expectation value can be estimated as an average of
the noisy auxiliary expectation values, after scaling them by the corresponding `sampled_signs` and by the `one_norm` coefficient
(obtained in the previous section).

```{code-cell} ipython3
# Scale noisy results by one-norm coefficient and by sampled signs
unbiased_samples = [
  one_norm * value * sign for value, sign in zip(noisy_expectation_values, sampled_signs)
]

# Estimate ideal expectation value
pec_value = sum(unbiased_samples) / len(unbiased_samples)

unmitigated_value = executor.evaluate(circuit)[0]

print(f"Expectation value without error mitigation:    {unmitigated_value}")
print(f"Expectation value with error mitigation (PEC): {pec_value}")
```
To obtain a better PEC estimation (with reduced statistical fluctuations) one should increase the number
of probabilistically generated circuits, i.e., one should increase the argument `num_samples` when calling 
the {func}`.sample_circuit()` function.
