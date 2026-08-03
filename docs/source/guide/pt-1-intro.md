---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How do I use PT?

As with all techniques, PT is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

In this first section, we see how to use PT in Mitiq, starting from a circuit of interest.

+++

## Problem setup
We first define the circuit, which in this example contains Hadamard (H), CNOT, and CZ gates.

```{code-cell} ipython3
from cirq import LineQubit, Circuit, CZ, CNOT, H

q0, q1, q2, q3  = LineQubit.range(4)
circuit = Circuit(
    H(q0),
    CNOT.on(q0, q1),
    CZ.on(q1, q2),
    CNOT.on(q2, q3),
)

print(circuit)
```

Next we define a simple executor function which inputs a circuit, executes
the circuit on a noisy simulator, and returns the probability of the ground
state. See the [Executors](executors.md) section for more information on
how to define more advanced executors.

During execution by the simulator, a coherent error is introduced by applying a 
rotation around the X-axis (Rx gate) to each output of any 2-qubit gate in the circuit of interest.

For the sake of this example executed by a simulator, we set the noise level to be proportional to the angle of the Rx rotation.

This noise model is well-suited to highlight the effect of Pauli Twirling,
which is a technique that transforms coherent noise into incoherent noise.

```{code-cell} ipython3
from numpy import pi
from cirq import CircuitOperation, CXPowGate, CZPowGate, DensityMatrixSimulator, Rx
from cirq.devices.noise_model import GateSubstitutionNoiseModel

def get_noise_model(noise_level: float) -> GateSubstitutionNoiseModel:
    """Substitute each CZ and CNOT gate in the circuit
    with the gate itself followed by an Rx rotation on the output qubits.
    """
    rads = pi / 2 * noise_level
    def noisy_c_gate(op):
        if isinstance(op.gate, (CZPowGate, CXPowGate)):
            return CircuitOperation(
                Circuit(
                    op.gate.on(*op.qubits), 
                    Rx(rads=rads).on_each(op.qubits),
                ).freeze())
        return op

    return GateSubstitutionNoiseModel(noisy_c_gate)

def execute(circuit: Circuit, noise_level: float):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit."""
    return (
        DensityMatrixSimulator(noise=get_noise_model(noise_level=noise_level))
        .simulate(circuit)
        .final_density_matrix[0, 0]
        .real
    )
```

The [executor](executors.md) can be used to evaluate noisy (unmitigated)
expectation values.

```{code-cell} ipython3
# Set the intensity of the noise
NOISE_LEVEL = 0.1

# Compute the expectation value of the |0><0| observable
# in both the noiseless and the noisy setup
ideal_value = execute(circuit, noise_level=0.0)
noisy_value = execute(circuit, noise_level=NOISE_LEVEL)

print(f"Error without twirling: {abs(ideal_value - noisy_value) :.3}")
```

## Apply PT
PT can be applied by first generating twirled variants of the circuit with the function
{func}`.generate_pauli_twirl_variants` from the `mitiq.pt` module, 
and then averaging over the results obtained by executing those variants.

The more variants are generated and averaged over, the more visible the results of PT are.
In this example we generate 5 twirled variants of the circuit, by setting the  `num_circuits` 
argument of the function {func}`.generate_pauli_twirl_variants` (default value is 10.)

```{code-cell} ipython3
from functools import partial
import numpy as np
from mitiq import Executor
from mitiq.pt import generate_pauli_twirl_variants

# Generate twirled circuits
NUM_TWIRLED_VARIANTS = 5
twirled_circuits = generate_pauli_twirl_variants(circuit, num_circuits=NUM_TWIRLED_VARIANTS)
# Average results executed over twirled circuits
pt_vals = Executor(partial(execute, noise_level=NOISE_LEVEL)).evaluate(twirled_circuits)
mitigated_result = np.average(pt_vals)

print(f"Error with twirling: {abs(ideal_value - mitigated_result) :.3}")
```

The idea behind Pauli Twirling is that it leaves the effective logical circuit unchanged, while tailoring the noise into stochastic Pauli errors.

```{admonition} Note:
Pauli Twirling is designed to transform noise, such as the coherent noise simulated in the example above,
but it should not be expected to always have a positive effect. In this sense, it is more of a noise tailoring technique, 
designed to be composed with other techniques rather than an error mitigation technique in itself.
```

+++

PT can also be stacked with dedicated quantum error mitigation techniques like ZNE or PEC.
See [Stacking PT with other QEM techniques](pt-3-options.md#stacking-pt-with-other-qem-techniques)
for a worked example and the
[ZNE with Pauli Twirling example](../examples/pt_zne.md) for a full tutorial.

The section
[What additional options are available when using PT?](pt-3-options.md)
contains information on more advanced ways of applying PT with Mitiq.
