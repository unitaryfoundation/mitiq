---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Comparing LRE and ZNE

Both LRE and ZNE work in two main stages: generate noise-scaled circuits via scaling, and apply inference to resulting measurements post-execution.

This workflow can be executed by a single call to `execute_with_lre` or `execute_with_zne`.

For resource estimation, Mitiq provides `multivariate_layer_scaling` to inspect the circuits that are to be executed.

For ZNE we have access to the scaled circuits using the function `scaled_circuits`.

## Problem Setup

For this demonstration, we'll first define a quantum circuit, and a method of executing circuits for demonstration purposes.

Here we will use the rotated randomized benchmarking circuits on a single qubit and generate 50 random such circuits.

```{code-cell} ipython3
from mitiq.benchmarks import generate_rotated_rb_circuits

circuits = generate_rotated_rb_circuits(
    n_qubits=1, num_cliffords=3, theta=0.7, trials=50, seed=4
)

print(circuits[0])
```

We define an [executor](../guide/executors.md) which simulates the input circuit subjected to depolarizing noise, and returns the probability of measuring the ground state.

By altering the value for `noise_level`, ideal and noisy expectation values can be obtained.

```{code-cell} ipython3
from cirq import DensityMatrixSimulator, depolarize


def execute(circuit, noise_level=0.025):
    noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

Let's compute the expectation values with and without noise.

```{code-cell} ipython3
# Collect ideal and noisy values (probability of measuring 0 across all circuits).
noisy_values = []
ideal_values = []
for circuit in circuits:
    noisy_values.append(execute(circuit))
    ideal_values.append(execute(circuit, noise_level=0.0))
```

```{code-cell} ipython3
import numpy as np

# The theoretical value for the probability of measuring 0 when taking
# an average over all the rotated rb circuits.
p = lambda theta: 1 - (2 / 3) * np.sin(theta / 2) ** 2

print(f"Average error for noisy values: {abs(np.mean(noisy_values) - p(0.7))}")
print(f"Average error for ideal values: {abs(np.mean(ideal_values) - p(0.7))}")
```

For the ideal values we still see a small error, because we are only taking the average over 50 rotated randomized benchmarking circuits, so there will be noise due to randomness.

The ideal value, defined in the funcion `p`, is attained when computing this average over all the rotated randomized benchmarking circuits.

If you increase the number of circuits, you will find that the average error for ideal values tends to zero.

## Apply LRE and ZNE directly

With the circuit and executor defined, we just need to choose the polynomial extrapolation degree as well as the fold multiplier.

For ZNE we use the default values for the scale factors.

```{code-cell} ipython3
from mitiq.lre import execute_with_lre
from mitiq.zne import execute_with_zne

degree = 2
fold_multiplier = 3

# Collect mitigated values (probability of measuring 0 across all circuits) using LRE and ZNE.
mitigated_values_lre = []
mitigated_values_zne = []

for circuit in circuits:
    mitigated_lre = execute_with_lre(
        circuit, execute, degree=degree, fold_multiplier=fold_multiplier
    )
    mitigated_values_lre.append(mitigated_lre)
    mitigated_zne = execute_with_zne(circuit, execute)
    mitigated_values_zne.append(mitigated_zne)
```

```{code-cell} ipython3
error_lre = abs(np.mean(mitigated_values_lre) - p(0.7))
error_zne = abs(np.mean(mitigated_values_zne) - p(0.7))

print(f"Average error of mitigated values using LRE: {error_lre}")
print(f"Average error of mitigated values using ZNE: {error_zne}")
```

## Resource estimation

We now compare the resources required (number of circuits to run and circuit depth) to run these protocols to have a fair comparison. First we do this for LRE.

```{code-cell} ipython3
from mitiq.lre.multivariate_scaling import multivariate_layer_scaling

avg_num_scaled_circuits_lre = 0.0
avg_depth_scaled_circuits_lre = 0.0

for circuit in circuits:
    noise_scaled_circuits_lre = multivariate_layer_scaling(
        circuit, degree, fold_multiplier
    )
    num_scaled_circuits_lre = len(noise_scaled_circuits_lre)

    avg_num_scaled_circuits_lre += num_scaled_circuits_lre / len(circuits)
    avg_depth_scaled_circuits_lre += (
        (1 / len(circuits))
        * sum(len(circuit) for circuit in noise_scaled_circuits_lre)
        / num_scaled_circuits_lre
    )

print(
    f"Average number of noise-scaled circuits for LRE = {avg_num_scaled_circuits_lre}"
)
print(f"Average circuit depth = {avg_depth_scaled_circuits_lre}")
```

Next for ZNE.

```{code-cell} ipython3
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne import construct_circuits

scale_factors = [1.0, 2.0, 3.0]

avg_num_scaled_circuits_zne = 0.0
avg_depth_scaled_circuits_zne = 0.0

for circuit in circuits:
    noise_scaled_circuits_zne = construct_circuits(
        circuit=circuit,
        scale_factors=[1.0, 2.0, 3.0],
        scale_method=fold_gates_at_random,
    )
    num_scaled_circuits_zne = len(noise_scaled_circuits_zne)

    avg_num_scaled_circuits_zne += num_scaled_circuits_zne / len(circuits)
    avg_depth_scaled_circuits_zne += (
        (1 / len(circuits))
        * sum(len(circuit) for circuit in noise_scaled_circuits_zne)
        / num_scaled_circuits_zne
    )

print(
    f"Average number of noise-scaled circuits for ZNE = {avg_num_scaled_circuits_zne}"
)
print(f"Average circuit depth = {avg_depth_scaled_circuits_zne}")
```

```{code-cell} ipython3
print(f"Error improvement LRE over ZNE: {error_zne/error_lre}")
print(
    f"Ratio number of circuits required for LRE vs ZNE: {avg_num_scaled_circuits_lre/avg_num_scaled_circuits_zne}"
)
```

## Conclusion

With an additional cost of many circuits---in this case, around 17 times more---LRE achieves a notable improvement, reducing the error rate by approximately threefold.

Although our current tests were limited in the number of circuits, which means these results carry some uncertainty, the potential of LRE is clear.

There’s exciting promise and further research will help us better understand the balance between performance gains and resource requirements.
