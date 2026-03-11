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

# How do I use LRE?

LRE works in two main stages: generate noise-scaled circuits via layerwise scaling, and apply inference to resulting measurements post-execution.

This workflow can be executed by a single call to {func}`.execute_with_lre`.
If more control is needed over the protocol, Mitiq provides {func}`.multivariate_layer_scaling` and {func}`.multivariate_richardson_coefficients` to handle the first and second steps respectively.

## Problem Setup

To demonstrate the use of LRE, we'll first define a quantum circuit, and a method of executing circuits for demonstration purposes.

For simplicity, we define a circuit whose unitary compiles to the identity operation.
Here we will use a randomized benchmarking circuit on a single qubit, visualized below.

```{code-cell} ipython3
from mitiq import benchmarks


circuit = benchmarks.generate_rb_circuits(n_qubits=1, num_cliffords=3)[0]

print(circuit)
```

We define an [executor](executors.md) which simulates the input circuit subjected to depolarizing noise, and returns the probability of measuring the ground state.
By altering the value for `noise_level`, ideal and noisy expectation values can be obtained.

```{code-cell} ipython3
from cirq import DensityMatrixSimulator, depolarize


def execute(circuit, noise_level=0.025):
    noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

Compare the noisy and ideal expectation values:

```{code-cell} ipython3
noisy = execute(circuit)
ideal = execute(circuit, noise_level=0.0)
print(f"Error without mitigation: {abs(ideal - noisy) :.5f}")
```

## Apply LRE directly

With the circuit and executor defined, we just need to choose the polynomial extrapolation degree as well as the fold multiplier.

```{code-cell} ipython3
from mitiq.lre import execute_with_lre


degree = 2
fold_multiplier = 3

mitigated = execute_with_lre(
    circuit,
    execute,
    degree=degree,
    fold_multiplier=fold_multiplier,
)


print(f"Error with mitigation (LRE): {abs(ideal - mitigated):.{3}}")
```

As you can see, the technique is extremely simple to apply, and no knowledge of the hardware/simulator noise is required.

## Step by step application of LRE

In this section we demonstrate the use of {func}`.multivariate_layer_scaling` and {func}`.multivariate_richardson_coefficients` for those who might want to inspect the intermediary circuits, and have more control over the protocol.

### Create noise-scaled circuits

We start by creating a number of noise-scaled circuits which we will pass to the executor.

```{code-cell} ipython3
from mitiq.lre.multivariate_scaling import multivariate_layer_scaling


noise_scaled_circuits = multivariate_layer_scaling(circuit, degree, fold_multiplier)
num_scaled_circuits = len(noise_scaled_circuits)

print(f"total number of noise-scaled circuits for LRE = {num_scaled_circuits}")
print(
    f"Average circuit depth = {sum(len(circuit) for circuit in noise_scaled_circuits) / num_scaled_circuits}"
)
```

As you can see, the noise scaled circuits are on average much longer than the original circuit.
An example noise-scaled circuit is shown below.

```{code-cell} ipython3
noise_scaled_circuits[3]
```

With the many noise-scaled circuits in hand, we can run them through our executor to obtain the expectation values.

```{code-cell} ipython3
noise_scaled_exp_values = [
    execute(circuit) for circuit in noise_scaled_circuits
]
```

### Classical inference

The penultimate step here is to fetch the coefficients we'll use to combine the noisy data we obtained above.
The astute reader will note that we haven't defined or used a `degree` or `fold_multiplier` parameter, and this is where they are both needed.

```{code-cell} ipython3
from mitiq.lre.inference import multivariate_richardson_coefficients


coefficients = multivariate_richardson_coefficients(
    circuit,
    fold_multiplier=fold_multiplier,
    degree=degree,
)
```

Each noise scaled circuit has a coefficient of linear combination and a noisy expectation value associated with it.

### Combine the results

```{code-cell} ipython3
mitigated = sum(
    exp_val * coeff
    for exp_val, coeff in zip(noise_scaled_exp_values, coefficients)
)
print(f"Error with mitigation (LRE): {abs(ideal - mitigated):.{3}}")
```

As you can see we again see a nice improvement in the accuracy using a two stage application of LRE.
