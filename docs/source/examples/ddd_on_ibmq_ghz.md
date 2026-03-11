---
jupytext:
  formats: md:myst
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
```{tags} ddd, qiskit, basic
```

# Digital dynamical decoupling (DDD) with Qiskit on GHZ Circuits

In this notebook DDD is applied to improve the success rate of the computation on a real hardware backend. 
A similar approach can be taken on a simulated backend, by setting the ``USE_REAL_HARDWARE`` option to ``False``
and specifying a simulated backend from `qiskit.providers.fake_provider`, which includes a noise model that approximates the noise of the
real device.

In DDD, sequences of gates are applied to slack windows, i.e. single-qubit idle windows, in a quantum circuit. 
Applying such sequences can reduce the coupling between the qubits and the environment, mitigating the effects of noise.
While the DDD module includes some built-in sequences, the user may choose to define others best suited to their application.
For more information on DDD, see the section [DDD section of the user guide](../guide/ddd.md).

+++

## Setup

We begin by importing the relevant modules and libraries that we will require
for the rest of this tutorial.

```{code-cell} ipython3
from collections.abc import Callable
import numpy as np
from matplotlib import pyplot as plt

import qiskit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from mitiq.interface.mitiq_qiskit import to_qiskit
from mitiq import ddd, QPROGRAM
from mitiq.ddd import insert_ddd_sequences
```

## Define DDD rules
We now use Mitiq's DDD _rule_, i. e., a function that generates DDD sequences of different length.
In this example, we test the performance of repeated I (default built into `get_circuit` below) and repeated IXIX, repeated XX, and XX sequences from Mitiq.

```{code-cell} ipython3
import cirq

def rep_ixix_rule(window_length: int) -> Callable[[int], QPROGRAM]:
    return ddd.rules.repeated_rule(
        window_length, [cirq.I, cirq.X, cirq.I, cirq.X]
    )

def rep_xx_rule(window_length: int) -> Callable[[int], QPROGRAM]:
    return ddd.rules.repeated_rule(window_length, [cirq.X, cirq.X])

# Set DDD sequences to test.
rules = [rep_ixix_rule, rep_xx_rule, ddd.rules.xx]

# Test the sequence insertion
for rule in rules:
    print(rule(10))
```

## Set parameters for the experiment

```{code-cell} ipython3
# Total number of shots to use.
shots = 10000

# Qubits to use on the experiment.
num_qubits = 2

# Test at multiple depths.
depths = [10, 30, 50, 100]
```

## Define the circuit

We use Greenberger-Horne-Zeilinger (GHZ) circuits to benchmark the performance of the device.
GHZ circuits are designed such that only two bitstrings $|00...0 \rangle$ and $|11...1 \rangle$
should be sampled, with $P_0 = P_1 = 0.5$.
As noted in *Mooney et al. (2021)* {cite}`Mooney_2021`, when GHZ circuits are run on a device, any other measured bitstrings are due to noise.
In this example the GHZ sequence is applied first, followed by a long idle window of identity gates and finally the inverse of the GHZ
sequence.
Therefore $P_0 = 1$, and the frequency of the $|0 \rangle$ bitstring is our target metric (in this example we only measure the first qubit).

```{code-cell} ipython3
def get_circuit(depth: int):
    """Returns a circuit composed of a GHZ sequence, idle windows,
    and finally an inverse GHZ sequence.

    Args:
        depth: The depth of the idle window in the circuit.
    """
    circuit = qiskit.QuantumCircuit(num_qubits, num_qubits)
    circuit.h(0)
    circuit.cx(0, 1)
    for _ in range(depth):
        circuit.id(0)
        circuit.id(1)
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit
```

Test the circuit output for depth 4, unmitigated

```{code-cell} ipython3
ibm_circ = get_circuit(4)
print(ibm_circ)
```

Test the circuit output for depth 4, with IX sequences inserted

```{code-cell} ipython3
ixix_circ = insert_ddd_sequences(ibm_circ, rep_ixix_rule)
print(ixix_circ)
```

## Define the executor

Now that we have a circuit, we define the `execute` function which inputs a circuit and returns an expectation value -
here, the frequency of sampling the correct bitstring.

```{code-cell} ipython3
USE_REAL_HARDWARE = True
correct_bitstring=[0]
```

```{code-cell} ipython3
:tags: [remove-cell]

# hidden settings to allow efficient docs build
USE_REAL_HARDWARE = False
depths = [2, 4, 6, 8]
correct_bitstring=[0, 0]
```

```{code-cell} ipython3
if USE_REAL_HARDWARE:
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
else:
    from qiskit_ibm_runtime.fake_provider import FakeLimaV2 as FakeLima
    backend = FakeLima()


def ibm_executor(
    circuit: qiskit.QuantumCircuit,
    shots: int,
    correct_bitstring: list[int],
    noisy: bool = True,
) -> float:
    """Executes the input circuit(s) and returns ⟨A⟩, where 
    A = |correct_bitstring⟩⟨correct_bitstring| for each circuit.

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the
            expectation value.
        correct_bitstring: Bitstring the circuit is expected to return, in the
            absence of noise.
    """
    if noisy:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=0,
        )
        transpiled = pm.run(circuit)

        if not isinstance(transpiled, list):
            transpiled = [transpiled]

        sampler = Sampler(backend)
        job = sampler.run(transpiled, shots=shots)
        all_counts = job.result()[0].join_data().get_counts()
    else:
        ideal_backend = AerSimulator()
        job = ideal_backend.run(circuit, optimization_level=0, shots=shots)
        all_counts = job.result().get_counts()

    # Convert from raw measurement counts to the expectation value
    prob_zero = all_counts.get("".join(map(str, correct_bitstring)), 0.0) / shots
    return prob_zero
```

## Run circuits with and without DDD

```{code-cell} ipython3
:tags: [remove-output]

data = []
for depth in depths:
    circuit = get_circuit(depth)
    noisy_value = ibm_executor(
            circuit, shots=shots, correct_bitstring=correct_bitstring
    )
    data.append((depth, "unmitigated", noisy_value))
    for rule in rules:
        ddd_circuit = insert_ddd_sequences(circuit, rule)
        ddd_value = ibm_executor(
            ddd_circuit, shots=shots, correct_bitstring=correct_bitstring
        )
        data.append((depth, rule.__name__, ddd_value))
```

Now we can visualize the results.

```{code-cell} ipython3
:tags: [remove-output]

# Plot unmitigated
x, y = [], []
for res in data:
    if res[1] == "unmitigated":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="Unmitigated")

# Plot xx
x, y = [], []
for res in data:
    if res[1] == "rep_xx_rule":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="rep_xx_rule")

# Plot ixix
x, y = [], []
for res in data:
    if res[1] == "rep_ixix_rule":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="rep_ixix_rule")

# Plot xx
x, y = [], []
for res in data:
    if res[1] == "xx":
        x.append(res[0])
        y.append(res[2])
plt.plot(x, y, "--*", label="xx")


plt.legend()
```

```{figure} ../_thumbnails/ddd_qiskit_ghz_plot.png
---

name: ddd-qiskit-ghz-plot-ibmq
---
Plot of the unmitigated and DDD-mitigated expectation values obtained from executing the corresponding circuits.
```

+++

We can see that DDD improves the expectation value at each circuit depth, and the repeated XX sequence is the best at mitigating the errors
occurring during idle windows.
