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

# How do I use ZNE?

ZNE is an easy to use technique which can be used in a single step
(for those who are in a hurry), or two steps (for those who want more control over the process). Both steps characterizing ZNE — noise scaling and extrapolation — can be easily applied with Mitiq. The
corresponding sub-modules are
[`mitiq.zne.scaling`](https://mitiq.readthedocs.io/en/latest/apidoc.html#module-mitiq.zne.scaling.folding)
and
[`mitiq.zne.inference`](https://mitiq.readthedocs.io/en/latest/apidoc.html#module-mitiq.zne.inference).

As with all techniques, ZNE is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

In the next cell you can select the frontend used in this tutorial. For example:

```{code-cell} ipython3
frontend = "cirq"
```

## Problem setup

We first define the circuit of interest. For simplicity, in this example we use
a randomized-benchmarking circuit whose ideal execution is equivalent to the
identity operation.

```{code-cell} ipython3
from mitiq import benchmarks

circuit = benchmarks.generate_rb_circuits(
  n_qubits=1, num_cliffords=2, return_type = frontend,
)[0]

print(circuit)
```

As an example, below we define a simple executor function
which inputs a circuit, executes it on a noisy simulator, and returns the probability
of the ground state.
See the [Executors](executors.md) section for more information on how to
define more advanced executors.

```{code-cell} ipython3
import numpy as np
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

def execute(circuit, noise_level=0.01):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    # Replace with code based on your frontend and backend.
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
print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.5f}")
```

## Apply ZNE

Zero-noise extrapolation can be easily implemented with the function
[`mitiq.zne.zne.execute_with_zne()`](https://mitiq.readthedocs.io/en/latest/apidoc.html#mitiq.zne.zne.execute_with_zne).

```{code-cell} ipython3
from mitiq import zne

mitigated_result = zne.execute_with_zne(circuit, execute)
```

```{code-cell} ipython3
print(f"Error with mitigation (ZNE): {abs(ideal_value - mitigated_result):.{3}}")
```

Here we observe that the application of ZNE reduces the estimation error when compared
to the unmitigated result.
In the example above, both the noise _scaling_ and _inference_ steps were taken behinds the scenes thanks to
the default options of {func}`.execute_with_zne` where the default noise scaling method is {func}`.fold_gates_at_random()`
and the executor computes the average expectation value once. Below we provide more details about these two aspects of ZNE
in Mitiq.

+++

## Select a noise scaling method

In Mitiq, one can select a noise scaling method via _noise scaling functions_.
A noise scaling function takes a circuit and a real scale factor as two inputs and
returns a new circuit. The returned circuit is equivalent to the input one (if executed on a noiseless backend),
but is more sensitive to noise when executed on a real noisy backend. In practice, by applying a noise
scaling function before the execution of a circuit, one can indirectly scale up the effect of noise. The noise scaling
function can either increase the total circuit execution time by inserting unitaries or increase the wait times in the
middle of a circuit execution. These two methods are _unitary folding_ and _identity scaling_ respectively.
See the section [What additional options are available in ZNE?](zne-3-options.md) for more details.

For example, the default function {func}`.fold_gates_at_random()` applies the _unitary folding_ map $G \rightarrow G G^\dagger G$
to a random subset of gates of the input circuit.
The folded circuit is more sensitive to gate errors since it has a number of gates approximately
equal to `scale_factor * n`, where `n` is the number of gates in the input circuit.

```{code-cell} ipython3
from mitiq.zne.scaling import fold_gates_at_random

folded = fold_gates_at_random(circuit, scale_factor=2.)
print("Folded circuit:", folded, sep="\n")
```

## Select a noise extrapolation method

Define a {class}`.Factory` object to select the noise extrapolation method and the noise scale
factors. The section
[What additional options are available when using ZNE?](zne-3-options.md) also
contains more details on the available noise extrapolation methods in Mitiq.

For example, a linear extrapolation method with scale factors 1 and 2, can be
initialized as follows:

```{code-cell} ipython3
from mitiq.zne.inference import LinearFactory

linear_fac = LinearFactory(scale_factors=[1.0, 2.0])
```

We can use the defined Factory and chosen noise scaling method to apply ZNE.

```{code-cell} ipython3
from mitiq import zne

mitigated_result = zne.execute_with_zne(
     circuit, execute, factory=linear_fac, scale_noise=fold_gates_at_random,
)
```

```{code-cell} ipython3
print(f"Error with mitigation (ZNE): {abs(ideal_value - mitigated_result):.{3}}")
```

The section [What additional options are available when using ZNE?](zne-3-options.md)
contains more information on both noise scaling and noise extrapolation methods in Mitiq.

## Two-stage application of ZNE

If you want to have more control and insight into how ZNE works, there are two functions available within `mitiq.zne` that allow circuit generation {func}`.zne.construct_circuits`, and expectation value calculation {func}`.zne.combine_results` respectively.
The high-level workflow is as follows.

1. Circuit definition
2. Circuit generation according to ZNE parameters
3. Circuit execution
4. Expectation value estimate via results from 3.

We'll use the circuit, and execution function as defined above, so step 1 is taken care of.
For step 2, we'll generate the noise scaled circuits as follows.


```{code-cell} ipython3
from mitiq.zne import construct_circuits, combine_results
from mitiq.zne.scaling import fold_gates_at_random

scale_factors = [1.0, 2.0, 3.0]

folded_circuits = construct_circuits(
    circuit=circuit,
    scale_factors=[1.0, 2.0, 3.0],
    scale_method=fold_gates_at_random,
)

folded_circuits
```

With the circuits in hand, we can move on to stage 3: circuit execution.
If a batched executor is available, then all circuits can be run at the same time, otherwise they will be run in series as demonstrated here.

```{code-cell} ipython3
results = [execute(circuit) for circuit in folded_circuits]

print(results)
```

```{warning}
The ordering of `results` must correspond to that of `folded_circuits` in order for the next stage to work correctly.
```

For the last step, we can calculate $\langle O\rangle_\text{ZNE}$ (step 4) be selecting an extrapolation technique, and passing it to {func}`.zne.combine_results`.

```{code-cell} ipython3
from mitiq.zne.inference import RichardsonFactory

extrapolation_method = RichardsonFactory(scale_factors=scale_factors).extrapolate
two_stage_zne_result = combine_results(
    scale_factors, results, extrapolation_method
)

print(f'Unmitigated value: {execute(circuit)}')
print(f'Mitigated value: {two_stage_zne_result}')
```

The two stage approach to ZNE allows for the inspection of the intermediary circuits that will be run, ahead of runtime.
This ensures an opportunity to check the computational cost of the technique before it is run, unlike when using {func}`.execute_with_zne`.
