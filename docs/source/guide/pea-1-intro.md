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

# How do I use PEA?

Probabilistic error amplification (PEA) estimates expectation values by probabilistically sampling noise-amplified circuits at several noise levels and extrapolating back to the zero-noise limit.

In Mitiq, this can be executed in a single call to {func}`.execute_with_pea` or step-by-step using {func}`mitiq.pea.pea.construct_circuits` and {func}`mitiq.pea.pea.combine_results`.

As with all techniques, PEA is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

## Problem setup

We first define the circuit of interest. For simplicity, in this example we use a randomized-benchmarking circuit whose ideal execution is equivalent to the identity operation.

```{code-cell} ipython3
from mitiq import benchmarks

circuit = benchmarks.generate_rb_circuits(
  n_qubits=1, num_cliffords=2, return_type="cirq",
)[0]

print(circuit)
```

```{note}
PEA also requires that the circuit can be decomposed into one- and two-qubit operations.
If your circuit contains larger operations, pre-compile it to a one- and two-qubit gate set
before applying PEA.
```

As an example, below we define a simple executor function which inputs a circuit, executes it on a noisy simulator, and returns the probability of the ground state.
See the [Executors](executors.md) section for more information on how to define more advanced executors.

```{code-cell} ipython3
import numpy as np

from cirq import DensityMatrixSimulator, depolarize

from mitiq.interface import convert_to_mitiq


def execute(circuit, noise_level=0.01):
    """Returns Tr[rho |0><0|] where rho is the state prepared by the circuit
    executed with depolarizing noise.
    """
    # Replace with code based on your frontend/backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

The [executor](executors.md) can be used to evaluate noisy (unmitigated) expectation values.

```{code-cell} ipython3
# Compute the expectation value of the |0><0| observable.
noisy_value = execute(circuit)
ideal_value = execute(circuit, noise_level=0.0)
print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.3f}")
```

## Apply PEA

PEA can be implemented with a single call to
{func}`.execute_with_pea`.

```{code-cell} ipython3
from mitiq import pea
from mitiq.zne.inference import LinearFactory

scale_factors = [1.0, 1.6, 2.4]
mitigated_result = pea.execute_with_pea(
    circuit,
    execute,
    scale_factors=scale_factors,
    noise_model="local_depolarizing",
    epsilon=0.01,
    extrapolation_method=LinearFactory.extrapolate,
    random_state=1,
)
```

```{code-cell} ipython3
print(f"Error with PEA: {abs(ideal_value - mitigated_result):.3f}")
```

Here we observe that the application of PEA reduces the estimation error when compared to the unmitigated result.
In the example above, both the noise amplification and extrapolation steps were taken behind the scenes thanks to the default options of {func}`.execute_with_pea`.
In the following pages, we describe additional options and show how to apply the two steps of PEA independently.

## Two-stage PEA workflow

If you want more control over the process, you can split PEA into two stages:

1. Construct noise-amplified circuits and execute them.
2. Combine the results and extrapolate to the zero-noise limit.

```{code-cell} ipython3
from mitiq import Executor, pea
from mitiq.zne.inference import LinearFactory

scale_factors = [1.0, 1.6, 2.4]

scaled_circuits, scaled_signs, scaled_norms = pea.construct_circuits(
    circuit,
    scale_factors=scale_factors,
    noise_model="local_depolarizing",
    epsilon=0.01,
    precision=0.1,
    random_state=1,
)

executor = Executor(execute)
scaled_results = [executor.evaluate(sc) for sc in scaled_circuits]

pea_value = pea.combine_results(
    scale_factors,
    scaled_results,
    scaled_norms,
    scaled_signs,
    extrapolation_method=LinearFactory.extrapolate,
)
```

```{code-cell} ipython3
print(f"Error with PEA (two-stage): {abs(ideal_value - pea_value):.3f}")
```