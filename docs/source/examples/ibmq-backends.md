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

```{tags} qiskit, zne, basic
```

# Error mitigation on IBMQ backends with Qiskit


This tutorial shows an example of how to mitigate noise on IBMQ backends.

## Settings

```{code-cell} ipython3
import qiskit
from qiskit_aer import QasmSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from mitiq import zne
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise
```

**Note:** If `USE_REAL_HARDWARE` is set to `False`, a classically simulated noisy backend is used instead of a real quantum computer.

```{code-cell} ipython3
USE_REAL_HARDWARE = False
```

## Setup: Defining a circuit

+++

For simplicity, we'll use a random single-qubit circuit with ten gates that compiles to the identity, defined below.

```{code-cell} ipython3
qreg, creg = qiskit.QuantumRegister(1), qiskit.ClassicalRegister(1)
circuit = qiskit.QuantumCircuit(qreg, creg)
for _ in range(10):
    circuit.x(qreg)
circuit.measure(qreg, creg)
print(circuit)
```

We will use the probability of the ground state as our observable to mitigate, the expectation value of which should
evaluate to one in the noiseless setting.


## High-level usage


To use Mitiq with just a few lines of code, we simply need to define a function which inputs a circuit and outputs
the expectation value to mitigate. This function will:

1. [Optionally] Add measurement(s) to the circuit.
2. Run the circuit.
3. Convert from raw measurement statistics (or a different output format) to an expectation value.

We define this function in the following code block. Because we are using IBMQ backends, we first load our account.

+++

**Note:** Using an IBM quantum computer requires a valid IBMQ account. See <https://quantum-computing.ibm.com/>
for instructions to create an account, save credentials, and see online quantum computers.

```{code-cell} ipython3
if QiskitRuntimeService.saved_accounts() and USE_REAL_HARDWARE:
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    noise_model = False
else:
    # Simulate the circuit with noise
    noise_model = initialized_depolarizing_noise(noise_level=0.02)
    # Default to a simulator
    backend = QasmSimulator(noise_model=noise_model)


def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 8192) -> float:
    """Returns the expectation value to be mitigated.

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """
    # Transpile the circuit so it can be properly run
    pm = generate_preset_pass_manager(
        backend=backend,
        basis_gates=noise_model.basis_gates if noise_model else None,
        optimization_level=0, # Important to preserve folded gates.
    )

    exec_circuit = pm.run(circuit)

    if not isinstance(exec_circuit, list):
        exec_circuit = [exec_circuit]

    # Run the circuit
    sampler = Sampler(backend)
    
    job = sampler.run(exec_circuit, shots=shots)

    # Convert from raw measurement counts to the expectation value
    counts = job.result()[0].join_data().get_counts()
    if counts.get("0") is None:
        expectation_value = 0.
    else:
        expectation_value = counts.get("0") / shots
    return expectation_value
```

At this point, the circuit can be executed to return a mitigated expectation value by running {func}`zne.execute_with_zne`,
as follows.

```{code-cell} ipython3
unmitigated = ibmq_executor(circuit)
mitigated = zne.execute_with_zne(circuit, ibmq_executor)
print(f"Unmitigated result {unmitigated:.3f}")
print(f"Mitigated result {mitigated:.3f}")
```

As long as a circuit and a function for executing the circuit are defined, the {func}`zne.execute_with_zne` function can
be called as above to return zero-noise extrapolated expectation value(s).


## Options


Different options for noise scaling and extrapolation can be passed into the {func}`zne.execute_with_zne` function.
By default, noise is scaled by locally folding gates at random, and the default extrapolation is Richardson.

To specify a different extrapolation technique, we can pass a different {class}`.Factory` object to {func}`zne.execute_with_zne`. The
following code block shows an example of using linear extrapolation with five different (noise) scale factors.

```{code-cell} ipython3
linear_factory = zne.inference.LinearFactory(scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0])
mitigated = zne.execute_with_zne(circuit, ibmq_executor, factory=linear_factory)
print(f"Mitigated result {mitigated:.3f}")
```

To specify a different noise scaling method, we can pass a different function for the argument ``scale_noise``. This
function should input a circuit and scale factor and return a circuit. The following code block shows an example of
scaling noise by global folding (instead of local folding, the default behavior for
{func}`zne.execute_with_zne`).

```{code-cell} ipython3
mitigated = zne.execute_with_zne(circuit, ibmq_executor, scale_noise=zne.scaling.fold_global)
print(f"Mitigated result {mitigated:.3f}")
```

Any different combination of noise scaling and extrapolation technique can be passed as arguments to
{func}`zne.execute_with_zne`.


## Lower-level usage

Here, we give more detailed usage of the Mitiq library which mimics what happens in the call to
{func}`zne.execute_with_zne` in the previous example. In addition to showing more of the Mitiq library, this
example explains the code in the previous section in more detail.

First, we define factors to scale the circuit length by and fold the circuit using the ``fold_gates_at_random``
local folding method.

```{code-cell} ipython3
scale_factors = [1., 1.5, 2., 2.5, 3.]
folded_circuits = [
        zne.scaling.fold_gates_at_random(circuit, scale)
        for scale in scale_factors
]

# Check that the circuit depth is (approximately) scaled as expected
for j, c in enumerate(folded_circuits):
    print(f"Number of gates of folded circuit {j} scaled by: {len(c) / len(circuit):.3f}")
```

For a noiseless simulation, the expectation of this observable should be 1.0 because our circuit compiles to the identity.
For a noisy simulation, the value will be smaller than one. Because folding introduces more gates and thus more noise,
the expectation value will decrease as the length (scale factor) of the folded circuits increase. By fitting this to
a curve, we can extrapolate to the zero-noise limit and obtain a better estimate.

Below we execute the folded circuits using the ``backend`` defined at the start of this example.

```{code-cell} ipython3
shots = 8192

# Transpile the circuit so it can be properly run
pm = generate_preset_pass_manager(
    backend=backend,
    basis_gates=noise_model.basis_gates if noise_model else None,
    optimization_level=0, # Important to preserve folded gates.
)
    
exec_circuit = pm.run(folded_circuits)

if not isinstance(exec_circuit, list):
        exec_circuit = [exec_circuit]

# Run the circuit
sampler = Sampler(backend)
job = sampler.run(exec_circuit, shots=shots)
```

**Note:** We set the ``optimization_level=0`` to prevent any compilation by Qiskit transpilers.


Once the job has finished executing, we can convert the raw measurement statistics to observable values by running the
following code block.
<!-- #endregion -->

```{code-cell} ipython3
all_counts = [job.result()[i].join_data().get_counts() for i in range(len(folded_circuits))]
expectation_values = [counts.get("0") / shots for counts in all_counts]
print(f"Expectation values:\n{expectation_values}")
```

We can now see the unmitigated observable value by printing the first element of ``expectation_values``. (This value
corresponds to a circuit with scale factor one, i.e., the original circuit.)

```{code-cell} ipython3
print("Unmitigated expectation value:", round(expectation_values[0], 3))
```

Now we can use the static ``extrapolate`` method of {class}`zne.inference.Factory` objects to extrapolate to the zero-noise limit. Below we use an exponential fit and print out the extrapolated zero-noise value.

```{code-cell} ipython3
zero_noise_value = zne.ExpFactory.extrapolate(scale_factors, expectation_values, asymptote=0.5)
print(f"Extrapolated zero-noise value:", round(zero_noise_value, 3))
```
