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

# What happens when I use PT?

The workflow of Pauli Twirling (PT) in Mitiq is represented in the figure below.

```{figure} ../img/pt_workflow.svg
---
width: 700px
name: pt-workflow-overview-2
---
Workflow of the PT technique in Mitiq.
```

- The user provides a `QPROGRAM` (i.e. a quantum circuit defined via any of the supported [frontends](frontends-backends.md)).
- Mitiq generates multiple twirled variants of the circuit, each with random Pauli gates inserted around CNOT and CZ gates.
- Each twirled variant is executed via a user-defined [Executor](executors.md).
- The expectation values are averaged to obtain the noise-tailored result.

With respect to the workflows of other error-mitigation techniques (e.g. [ZNE](zne-4-low-level.md) or [PEC](pec-4-low-level.md)),
PT generates multiple randomly-twirled circuits and averages their results.
There is no complex inference step --- the final result is a simple average.

```{note}
The `num_circuits` parameter in {func}`.generate_pauli_twirl_variants` controls how many
twirled variants are generated. More variants lead to better convergence of the
Pauli-twirled noise channel.
```

As shown in [How do I use PT?](pt-1-intro.md), {func}`.generate_pauli_twirl_variants` generates
the twirled circuits. In the next sections, we show how to apply PT at a lower level:

- Twirling CZ and CNOT gates in the circuit
- Executing the modified circuits
- Estimating the expectation value by averaging over randomized twirling circuits

## Twirling CZ and CNOT gates in the circuit
To twirl about particular gates, we need the Pauli group for those gates. These groups are stored as lookup tables, in {attr}`mitiq.pt.pt.CNOT_twirling_gates` and {attr}`mitiq.pt.pt.CZ_twirling_gates`, so that we can randomly select a tuple from the group. Now we're ready to twirl our gates.

First let's define our circuit:
```{code-cell} ipython3
from cirq import LineQubit, Circuit, CZ, CNOT

a, b, c, d  = LineQubit.range(4)
circuit = Circuit(
    CNOT.on(a, b),
    CZ.on(b, c),
    CNOT.on(c, d),
)

print(circuit)
```
Now, we can see what happens when we apply the PT functions, through {func}`.twirl_CNOT_gates()` and the subsequent {func}`.twirl_CZ_gates()`
```{code-cell} ipython3
from mitiq import pt

circuit_to_twirl = circuit.copy()
CNOT_twirled_circuits = pt.twirl_CNOT_gates(circuit_to_twirl, num_circuits=10)
twirled_circuits = [
    pt.twirl_CZ_gates(c, num_circuits=1)[0] for c in CNOT_twirled_circuits
]
print("Twirling just the CNOT gates: \n", CNOT_twirled_circuits[0], "\n")
print("Twirling both CNOT and CZ gates: \n" ,twirled_circuits[0])
```
We see that we return lists of the randomly twirled circuits, and so we must take a simple average over their expectation values.

## Executing the modified circuits

Now that we have our twirled circuits, let's simulate some noise and execute those circuits, using the {class}`mitiq.Executor` to collect the results.
```{code-cell} ipython3
from cirq import DensityMatrixSimulator, amplitude_damp
from mitiq import Executor


def execute(circuit, noise_level=0.003):
    """Returns Tr[ρ |00..⟩⟨00..|] where ρ is the state prepared by the circuit
    executed with amplitude damping noise.
    """
    noisy_circuit = circuit.with_noise(amplitude_damp(noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real

executor = Executor(execute)
expvals = executor.evaluate(twirled_circuits)
```


## Estimate expectation value by averaging over randomized twirling circuits
Pauli Twirling doesn't require running the circuit at different noise levels or with different noise models. It applies a randomized sequence of Pauli operations within the same quantum circuit and averages the results to reduce the effect of the noise.

```{code-cell} ipython3
import numpy as np

average = float(np.average(expvals))
print(f"Average expectation value over {len(twirled_circuits)} twirled circuits: {average:.6f}")
```

Keep in mind, this code is for illustration and that the noise level, type of noise (here amplitude damping), and the observable need to be adapted to the specific experiment.

If executed on a noiseless backend, a given twirled circuit and the original `circuit` are equivalent.
On a real backend, they have a different sensitivity to noise. The core idea of the PT technique is that
averaging over twirled circuits tailors the noise into stochastic Pauli channels, such that a simple average over results
will return a noise-tailored result.

As a final remark, we stress that the low-level procedure shown above is exactly what
{func}`.generate_pauli_twirl_variants` does behind the scenes --- it calls {func}`.twirl_CNOT_gates`
and {func}`.twirl_CZ_gates` on each circuit. Let's verify this:

```{code-cell} ipython3
from mitiq.pt import generate_pauli_twirl_variants

high_level_circuits = generate_pauli_twirl_variants(circuit, num_circuits=10)
high_level_expvals = executor.evaluate(high_level_circuits)
high_level_average = float(np.average(high_level_expvals))

print(f"Low-level average:  {average:.6f}")
print(f"High-level average: {high_level_average:.6f}")
```
