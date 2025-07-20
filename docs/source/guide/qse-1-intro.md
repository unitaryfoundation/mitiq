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

# How do I use QSE?
Here we show how to use QSE by means of a simple example.

## Problem setup
To use QSE, we call `qse.execute_with_qse()` with five “ingredients”:

- A quantum circuit to prepare a state
- A quantum computer or noisy simulator which returns a {class}`.QuantumResult`
- An observable we wish to compute the error mitigated expectation value
- A list of "check operators" which can be stabilizers, symmetries, or anything else.
- A "code Hamiltonian" which defines the state with the least amount of errors

## 1. Define a quantum circuit
The quantum circuit can be specified as any quantum circuit supported by Mitiq.

In the next cell we define (as an example) a quantum circuit that prepares the logical 0 state in the [[5,1,3]] code.
For simplicity, we use Gram Schmidt orthogonalization to form a unitary matrix that we use as a gate to prepare our state.

```{code-cell} ipython3
def prepare_logical_0_state_for_5_1_3_code():
    """
    To simplify the testing logic. We hardcode the logical 0 and logical 1
    states of the [[5,1,3]] code, copied from:
    https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code
    We then use Gram-Schmidt orthogonalization to fill up the rest of the
    matrix with orthonormal vectors.
    Following this we construct a circuit that has this matrix as its gate.
    """

    def gram_schmidt(
        orthogonal_vecs: list[np.ndarray],
    ) -> np.ndarray:
        # normalize input
        orthonormalVecs = [
            vec / np.sqrt(np.vdot(vec, vec)) for vec in orthogonal_vecs
        ]
        dim = np.shape(orthogonal_vecs[0])[0]  # get dim of vector space
        for i in range(dim - len(orthogonal_vecs)):
            new_vec = np.zeros(dim)
            new_vec[i] = 1  # construct ith basis vector
            projs = sum(
                [
                    np.vdot(new_vec, cached_vec) * cached_vec
                    for cached_vec in orthonormalVecs
                ]
            )  # sum of projections of new vec with all existing vecs
            new_vec -= projs
            orthonormalVecs.append(
                new_vec / np.sqrt(np.vdot(new_vec, new_vec))
            )
        return np.reshape(orthonormalVecs, (32, 32)).T

    logical_0_state = np.zeros(32)
    for z in ["00000", "10010", "01001", "10100", "01010", "00101"]:
        logical_0_state[int(z, 2)] = 1 / 4
    for z in [
        "11011",
        "00110",
        "11000",
        "11101",
        "00011",
        "11110",
        "01111",
        "10001",
        "01100",
        "10111",
    ]:
        logical_0_state[int(z, 2)] = -1 / 4

    logical_1_state = np.zeros(32)
    for z in ["11111", "01101", "10110", "01011", "10101", "11010"]:
        logical_1_state[int(z, 2)] = 1 / 4
    for z in [
        "00100",
        "11001",
        "00111",
        "00010",
        "11100",
        "00001",
        "10000",
        "01110",
        "10011",
        "01000",
    ]:
        logical_1_state[int(z, 2)] = -1 / 4

    # Fill up the rest of the matrix with orthonormal vectors
    matrix = gram_schmidt([logical_0_state, logical_1_state])
    circuit = cirq.Circuit()
    g = cirq.MatrixGate(matrix)
    qubits = cirq.LineQubit.range(5)
    circuit.append(g(*qubits))

    return circuit
```

## 2. Define an executor
We define an [executor](executors.md) function which inputs a circuit and returns a `QuantumResult`.
Here for sake of example we use a simulator that adds single-qubit depolarizing noise after each moment and returns the final density matrix.

```{code-cell} ipython3
import numpy as np
import cirq
from mitiq import QPROGRAM, Observable, PauliString, qse
from mitiq.interface import convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix


def execute_with_depolarized_noise(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0],
        noise_model_function=cirq.depolarize,
        noise_level=(0.01,),
    )
```

## 3. Observable
We define an observable in the code subspace: $O = ZZZZZ$.
As an example, assume that we wish to compute the expectation value $\mathrm{tr}[\rho O]$ of the observable $O$.

```{code-cell} ipython3
def get_observable_in_code_space(observable: list[cirq.PauliString]):
    FIVE_I = PauliString("IIIII")
    projector_onto_code_space = [
        PauliString("XZZXI"),
        PauliString("IXZZX"),
        PauliString("XIXZZ"),
        PauliString("ZXIXZ"),
    ]

    observable_in_code_space = Observable(FIVE_I)
    all_paulis = projector_onto_code_space + [observable]
    for g in all_paulis:
        observable_in_code_space *= 0.5 * Observable(FIVE_I, g)
    return observable_in_code_space

```

## 4. Check Operators and Code Hamiltonian

We then assign the check operators as a list of pauli strings, and the code Hamiltonian as an Observable for the [[5,1,3]] code.
The check operators of the [[5,1,3]] code are simply the expansion of the code’s 4 generators: $[XZZXI, IXZZX, XIXZZ, ZXIXZ]$

```{code-cell} ipython3
def get_5_1_3_code_check_operators_and_code_hamiltonian() -> tuple:
    """
    Returns the check operators and code Hamiltonian for the [[5,1,3]] code
    The check operators are computed from the stabilizer generators:
    (1+G1)(1+G2)(1+G3)(1+G4)  G = [XZZXI, IXZZX, XIXZZ, ZXIXZ]
    source: https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code
    """
    Ms = [
        "YIYXX",
        "ZIZYY",
        "IXZZX",
        "ZXIXZ",
        "YYZIZ",
        "XYIYX",
        "YZIZY",
        "ZZXIX",
        "XZZXI",
        "ZYYZI",
        "IYXXY",
        "IZYYZ",
        "YXXYI",
        "XXYIY",
        "XIXZZ",
        "IIIII",
    ]
    Ms_as_pauliStrings = [
        PauliString(M, coeff=1, support=range(5)) for M in Ms
    ]
    negative_Ms_as_pauliStrings = [
        PauliString(M, coeff=-1, support=range(5)) for M in Ms
    ]
    Hc = Observable(*negative_Ms_as_pauliStrings)
    return Ms_as_pauliStrings, Hc
```

## Run QSE

With everything defined, we can now run QSE in full.

```{code-cell} ipython3
def demo_qse():
  circuit = prepare_logical_0_state_for_5_1_3_code()
  observable = get_observable_in_code_space(PauliString("ZZZZZ"))
  check_operators, code_hamiltonian = get_5_1_3_code_check_operators_and_code_hamiltonian()
  return qse.execute_with_qse(
        circuit,
        execute_with_depolarized_noise,
        check_operators,
        code_hamiltonian,
        observable,
    )
```

### The mitigated result
```{code-cell} ipython3
print(demo_qse())
```

### The noisy result
```{code-cell} ipython3
circuit = prepare_logical_0_state_for_5_1_3_code()
observable = get_observable_in_code_space(PauliString("ZZZZZ"))
print(observable.expectation(circuit, execute_with_depolarized_noise))
```

### The noiseless result
```{code-cell} ipython3
def execute_no_noise(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0], noise_level=(0,)
    )

print(observable.expectation(circuit, execute_no_noise))
```