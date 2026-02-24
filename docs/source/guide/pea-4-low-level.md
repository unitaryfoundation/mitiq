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

# What happens when I use PEA?

PEA in Mitiq is divided into two steps: probabilistic sampling of noise-amplified circuits, and extrapolation of the corresponding expectation values.
The workflow is shown in the figure below.

```{figure} ../img/pea_workflow.png
---
width: 500
name: figpea
---
The diagram shows the workflow of the probabilistic error amplification (PEA) technique in Mitiq.
```

**The first step** involves generating and executing noise-amplified circuits.
  - The user provides a ``QPROGRAM`` (a circuit from a supported frontend).
  - Mitiq generates lists of probabilistically sampled circuits for each
    noise scale factor using a noise model and baseline noise level.
  - Each sampled circuit is executed and its noisy expectation value recorded.

**The second step** involves combining the sampled results at each scale
factor and extrapolating to the zero-noise limit using a ZNE-style inference
method.

As demonstrated in [How do I use PEA?](pea-1-intro.md), the function
{func}`.execute_with_pea` applies both steps behind the scenes.
In the next sections, we show how to apply each step independently.

## First step: generating and executing noise-amplified circuits

### Problem setup
We define a circuit and an executor, as shown in [How do I use PEA?](pea-1-intro.md).

```{code-cell} ipython3
from mitiq import benchmarks
from cirq import DensityMatrixSimulator, depolarize

circuit = benchmarks.generate_rb_circuits(
  n_qubits=2, num_cliffords=3, return_type="cirq",
)[0]

print(circuit)

def execute(circuit, noise_level=0.01):
    """Returns Tr[rho |0><0|] where rho is the state prepared by the circuit
    executed with depolarizing noise.
    """
    noisy_circuit = circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real
```

### Sample noise-amplified circuits

```{code-cell} ipython3
from mitiq import pea

scale_factors = [1, 1.6, 2.4]
scaled_circuits, scaled_signs, scaled_norms = pea.construct_circuits(
    circuit,
    scale_factors=scale_factors,
    noise_model="local_depolarizing",
    epsilon=0.01,
    precision=0.2,
    random_state=1,
)
```

### Execute sampled circuits

```{code-cell} ipython3
from mitiq import Executor

executor = Executor(execute)
scaled_results = [executor.evaluate(sc) for sc in scaled_circuits]
```

## Second step: combining results and extrapolating

```{code-cell} ipython3
from mitiq.zne.inference import LinearFactory

pea_value = pea.combine_results(
    scale_factors,
    scaled_results,
    scaled_norms,
    scaled_signs,
    extrapolation_method=LinearFactory.extrapolate,
)
raw_value = executor.evaluate(circuit)[0]
ideal_value = executor.evaluate(circuit, noise_level=0)[0]

print(f"noisy error: {abs(raw_value - ideal_value):.3f}")
print(f"PEA error:   {abs(pea_value - ideal_value):.3f}")
```

```{attention}
Due to randomness in the PEA sampling protocol the PEA error is not always gauranteed to be smaller than the noisy error.
```

The two steps shown above are what {func}`.execute_with_pea` performs internally.
You can also request additional diagnostics by setting ``full_output=True`` in {func}`.execute_with_pea`, which returns a dictionary containing raw PEA data.
