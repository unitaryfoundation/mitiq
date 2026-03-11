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

(guide/observables/observables)=
# Observables

+++

The `mitiq.Observable` class is a way to represent an observable as a linear combination of Pauli strings. This class can be used to compute expectation values which are mitigated by techniques in Mitiq.

```{code-cell} ipython3
from mitiq import Executor, Observable, PauliString
```

## Creating observables

+++

To create an observable, specify the `mitiq.PauliString`s that form the observable.

```{code-cell} ipython3
pauli1 = PauliString("ZZ", coeff=-1.21)
pauli2 = PauliString("X", support=(1,))
pauli3 = PauliString("ZX", coeff=3.2)

obs = Observable(pauli1, pauli2, pauli3)
print(obs)
```

## Basic properties and operations

+++

See the `Observable`s support and number of qubits as follows:

```{code-cell} ipython3
print(f"Observable acts (non-trivially) on {obs.nqubits} qubit(s) indexed {obs.qubit_indices}.")
```

The `PauliString`s of an observable are split into groups which can be measured simultaneously via single-qubit basis rotations and measurements.

```{code-cell} ipython3
print(f"Observable has {obs.nterms} `PauliString`(s) partitioned into {obs.ngroups} group(s).", end="\n\n")

for i, group in enumerate(obs.groups, start=1):
    print(f"Group {i}:", group)
```

You can (re-)partition the groups by calling `Observable.partition`.

```{code-cell} ipython3
obs.partition(seed=0)
```

Partitioning methods are generally randomized algorithms. For deterministic behavior, supply a seed. You can specify the groups manually as follows.

```{code-cell} ipython3
from mitiq.observable.pauli import PauliStringCollection

group1 = PauliStringCollection(pauli1)
group2 = PauliStringCollection(pauli2, pauli3)

obs = Observable.from_pauli_string_collections(group1, group2)
```

To see the (potentially very large) matrix representation of the observable:

```{code-cell} ipython3
obs.matrix()
```

You can explicitly specify the qubits to include in the matrix as follows.

```{code-cell} ipython3
obs.matrix(qubit_indices=[0, 1, 2])
```

Identity matrices are inserted on qubits outside of the observable's support.

+++

## Computing expectation values

+++

The main purpose of observables in Mitiq is computing expectation values (to perform error mitigation on). To do so, specify a state-preparation circuit as any `mitiq.QPROGRAM`. Here will we use Cirq.

```{code-cell} ipython3
import cirq
from mitiq.interface import mitiq_cirq
```

```{code-cell} ipython3
circuit = cirq.testing.random_circuit(
    cirq.LineQubit.range(obs.nqubits), n_moments=5, op_density=1, random_state=2
)
circuit
```

The `Observable.measure_in` method returns a set of circuits with single-qubit measurements to run on hardware for computing the expectation value.

```{code-cell} ipython3
circuits = obs.measure_in(circuit)

for c in circuits:
    print(c, end="\n\n")
```

To compute the expectation value, use the `Observable.expectation` method which an executor.

```{code-cell} ipython3
obs.expectation(circuit, execute=mitiq_cirq.sample_bitstrings)
```

## Using observables in error mitigation techniques

+++

In error mitigation techniques, you can provide an observable to specify the expectation value to mitigate.

```{warning}
As note in the [executor documentation](./executors.md#the-input-function), the executor must be annotated with the appropriate type hinting for the return type. Additionally, when specifying an `Observable`, you must ensure that the return type of the executor function is `MeasurementResultLike` or `DensityMatrixLike`.
```

```{code-cell} ipython3
from mitiq import zne
```

```{code-cell} ipython3
executor = Executor(mitiq_cirq.compute_density_matrix)

zne_value = zne.execute_with_zne(circuit, executor, obs)
print(f"ZNE value: {zne_value :g}")
```

If you do not provide an observable, the `executor` must compute and return the expectation value to mitigate.
