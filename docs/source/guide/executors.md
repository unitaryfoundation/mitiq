---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Executors

+++

Error mitigation methods can involve running many circuits. The `mitiq.Executor` class is a tool for efficiently running many circuits and storing the results.

```{code-cell} ipython3
from mitiq import Executor, Observable, PauliString, QPROGRAM, QuantumResult
```

## The input function

+++

To instantiate an `Executor`, provide a function which either:

1. Inputs a `mitiq.QPROGRAM` and outputs a `mitiq.QuantumResult`.
2. Inputs a sequence of `mitiq.QPROGRAM`s and outputs a sequence of `mitiq.QuantumResult`s.

```{warning}
To avoid confusion and invalid results, the executor function must be [annotated](https://peps.python.org/pep-3107/) to tell Mitiq which type of `QuantumResult` it returns. Functions without annotations are assumed to return `float`s.
```

A `QPROGRAM` is "something which a quantum computer inputs" and a `QuantumResult` is "something which a quantum computer outputs." The latter is canonically a bitstring for real quantum hardware, but can be other objects for testing, e.g. a density matrix.

```{code-cell} ipython3
print(QPROGRAM)
```

```{code-cell} ipython3
print(QuantumResult)
```

## Creating an `Executor`

+++

The function `mitiq_cirq.compute_density_matrix` inputs a Cirq circuit and returns a density matrix as an `np.ndarray`.

```{code-cell} ipython3
import inspect

from mitiq.interface import mitiq_cirq

print(inspect.getfullargspec(mitiq_cirq.compute_density_matrix).annotations["return"])
```

We can instantiate an `Executor` with it as follows.

```{code-cell} ipython3
executor = Executor(mitiq_cirq.compute_density_matrix)
```

## Running circuits

+++

When first created, the executor hasn't been called yet and has no executed circuits and no computed results in memory.

```{code-cell} ipython3
print("Calls to executor:", executor.calls_to_executor)
print("\nExecuted circuits:\n", *executor.executed_circuits, sep="\n")
print("\nQuantum results:\n", *executor.quantum_results, sep="\n")
```

To run a circuit or sequence of circuits, use the `Executor.evaluate` method.

```{code-cell} ipython3
import cirq

q = cirq.LineQubit(0)
circuit = cirq.Circuit(cirq.H.on(q))

obs = Observable(PauliString("Z"))

results = executor.evaluate(circuit, obs)
print("Results:", results)
```

The `executor` has now been called and has results in memory. Note that `mitiq_cirq.compute_density_matrix` simulates the circuit with noise by default, so the resulting state (density matrix) is noisy.

```{code-cell} ipython3
print("Calls to executor:", executor.calls_to_executor)
print("\nExecuted circuits:\n", *executor.executed_circuits, sep="\n")
print("\nQuantum results:\n", *executor.quantum_results, sep="\n")
```

The interface for running a sequence of circuits is the same.

```{code-cell} ipython3
circuits = [cirq.Circuit(pauli.on(q)) for pauli in (cirq.X, cirq.Y, cirq.Z)]

results = executor.evaluate(circuits, obs)
print("Results:", results)
```

In addition to the results of running these circuits we have the full history.

```{code-cell} ipython3
print("Calls to executor:", executor.calls_to_executor)
print("\nExecuted circuits:\n", *executor.executed_circuits, sep="\n")
print("\nQuantum results:\n", *executor.quantum_results, sep="\n")
```

### Batched execution

+++

Notice in the above output that the executor has been called once for each circuit it has executed. This is because `mitiq_cirq.compute_density_matrix` inputs one circuit and outputs one `QuantumResult`.

Several quantum computing services allow running a sequence, or "batch," of circuits at once.
This is especially important for error mitigation: techniques such as ZNE, PEC, and LRE typically generate **many** noise-scaled or sampled circuits, and batching reduces the number of backend round-trips (and often wall-clock time) without changing the quantum resource counts themselves.
See also [Resource Requirements](resource-requirements.md).

To define a batched executor, the function must:

1. Accept a **sequence** of circuits (e.g. `list[QPROGRAM]`), and
2. Be **annotated** to return a sequence of results (`list[T]`, `Sequence[T]`, `tuple[T]`, or `Iterable[T]` where `T` is a `QuantumResult`).

Mitiq detects batching from that return-type annotation. Functions without a batched annotation are treated as serial even if they could accept a list.

Here is a density-matrix example:

```{code-cell} ipython3
import numpy as np


def batch_compute_density_matrix(circuits: list[cirq.Circuit]) -> list[np.ndarray]:
    return [mitiq_cirq.compute_density_matrix(circuit) for circuit in circuits]


batched_executor = Executor(batch_compute_density_matrix, max_batch_size=10)
```

You can check if Mitiq detected the ability to batch as follows.

```{code-cell} ipython3
batched_executor.can_batch
```

Now when running a batch of circuits, the executor will be called as few times as possible.

```{code-cell} ipython3
circuits = [cirq.Circuit(pauli.on(q)) for pauli in (cirq.X, cirq.Y, cirq.Z)]

results = batched_executor.evaluate(circuits, obs)

print("Results:", results)
print("\nCalls to executor:", batched_executor.calls_to_executor)
print("\nExecuted circuits:\n", *batched_executor.executed_circuits, sep="\n")
print("\nQuantum results:\n", *batched_executor.quantum_results, sep="\n")
```

#### Float-valued batched executors

Hardware jobs often return expectation values (floats) rather than density matrices.
A batched float executor looks like this:

```{code-cell} ipython3
def serial_float_executor(circuit: cirq.Circuit) -> float:
    """Serial executor: one circuit → one float."""
    rho = mitiq_cirq.compute_density_matrix(circuit)
    # Population of |0⟩ as a simple float-valued result.
    return float(np.real(rho[0, 0]))


def batched_float_executor(circuits: list[cirq.Circuit]) -> list[float]:
    """Batched executor: many circuits → many floats in one call."""
    # For demonstration only: this just loops the serial call under the
    # hood, so it isn't actually batched. Replace the body with your
    # provider's batch/job API to get real batching.
    return [serial_float_executor(c) for c in circuits]


serial_ex = Executor(serial_float_executor)
batched_ex = Executor(batched_float_executor, max_batch_size=50)

sample = [cirq.Circuit(pauli.on(q)) for pauli in (cirq.X, cirq.Y, cirq.Z)]
_ = serial_ex.evaluate(sample)
_ = batched_ex.evaluate(sample)

print("Serial calls: ", serial_ex.calls_to_executor)   # one call per circuit
print("Batched calls:", batched_ex.calls_to_executor)  # one call for the batch
print("can_batch:    ", serial_ex.can_batch, batched_ex.can_batch)
```

```{tip}
Replace the body of `batched_float_executor` with your provider's batch/job API
(for example, submitting a list of circuits in a single QPU job). Mitiq only
requires that the function accept a list of circuits and return a list of results
of equal length, in the same order.
```

#### `max_batch_size`

The optional `max_batch_size` argument caps how many circuits are sent in a single call.
If you evaluate more circuits than `max_batch_size`, Mitiq splits them into multiple batches:

```{code-cell} ipython3
small_batch_executor = Executor(batched_float_executor, max_batch_size=2)
many_circuits = [cirq.Circuit(cirq.X(q)) for _ in range(5)]
_ = small_batch_executor.evaluate(many_circuits)
print("Calls with max_batch_size=2:", small_batch_executor.calls_to_executor)
```

Choose `max_batch_size` based on your backend's limits (job size, payload size, or queue policy).

## Using `Executor`s in error mitigation techniques

+++

You can provide a function or an `Executor` to the `executor` argument of error mitigation techniques, but **providing an `Executor` is strongly recommended** for seeing the history of results.

```{code-cell} ipython3
from mitiq import zne
```

```{code-cell} ipython3
batched_executor = Executor(batch_compute_density_matrix, max_batch_size=10)

zne_value = zne.execute_with_zne(
    cirq.Circuit(cirq.H.on(q) for _ in range(6)), 
    executor=batched_executor, 
    observable=obs
)
print(f"ZNE value: {zne_value :g}")
```

```{code-cell} ipython3
print("Calls to executor:", batched_executor.calls_to_executor)
print("\nExecuted circuits:\n", *batched_executor.executed_circuits, sep="\n")
print("\nQuantum results:\n", *batched_executor.quantum_results, sep="\n")
```

With a batched executor, all noise-scaled circuits for a non-adaptive method
(for example a fixed set of ZNE scale factors) can be submitted together,
which is the usual way to use Mitiq efficiently on cloud hardware
{cite}`Russo_2022_Testing`.

## Defining an `Executor` that returns measurement outcomes (bitstrings)

+++

In the previous examples we have shown executors that return the density matrix of the final state. This is possible only for classical simulations.
The typical result of a real quantum computation is instead a list of bitstrings corresponding to the ("0" or "1") outcomes obtained when measuring each qubit in the computational basis.
In Mitiq this type of quantum backend is captured by an {class}`.Executor` that returns a {class}`.MeasurementResult` object.

For example, here is an example of a Cirq executor function that returns raw measurement outcomes:

```{code-cell} ipython3
from mitiq import MeasurementResult

def noisy_sampler(circuit, noise_level=0.1, shots=1000) -> MeasurementResult:
    circuit_to_run = circuit.with_noise(cirq.depolarize(noise_level))
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(circuit_to_run, repetitions=shots)
    bitstrings = np.column_stack(list(result.measurements.values()))
    qubit_indices = tuple(
            int(q[2:-1])  # Extract index from "q(index)" string
            for k in result.measurements.keys()
            for q in k.split(",")
    )
    return MeasurementResult(bitstrings, qubit_indices)
```

```{code-cell} ipython3
# Circuit with measurements to test the noisy_sampler function
circuit_with_measurements = circuit.copy()
circuit_with_measurements.append(cirq.measure(*circuit.all_qubits()))

print("Circuit to execute:", circuit_with_measurements)
noisy_sampler(circuit_with_measurements)
```

The rest of the Mitiq workflow is the same as in the case of a density matrix executor. For example:

```{code-cell} ipython3
executor = Executor(noisy_sampler)
obs = Observable(PauliString("X"))
results = executor.evaluate(circuit, obs)

print("Results:", results)
print("Calls to executor:", executor.calls_to_executor)
print("\nExecuted circuits:\n", *executor.executed_circuits, sep="\n")
print("\nQuantum results:\n", *executor.quantum_results, sep="\n")
```
