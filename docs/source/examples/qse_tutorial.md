---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{tags} qse, cirq, intermediate
```

# Quantum subspace expansion (QSE) with the [[5,1,3]] code

This tutorial demonstrates quantum subspace expansion (QSE) as an error
mitigation technique, following the [[5,1,3]] code example from McClean *et
al.* {cite}`McClean_2020_NatComm`. We prepare a logical $\lvert\overline{0}\rangle$
state, apply single-qubit depolarizing noise, and recover the logical
$\overline{Z}$ expectation value with `mitiq.qse`.

Unlike techniques that only post-process a scalar expectation value, QSE builds
a small subspace around the noisy state using check operators (here, elements of
the stabilizer group) and solves a projected problem inside that subspace. For
background, see the [QSE user guide](../guide/qse.md).

## Setup

```{code-cell} ipython3
import warnings

import cirq
import matplotlib.pyplot as plt
import numpy as np

from mitiq import Observable, PauliString, QPROGRAM, qse
from mitiq.interface import convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 14})
%matplotlib inline
```

## Prepare a logical state of the [[5,1,3]] code

The perfect [[5,1,3]] code encodes one logical qubit in five physical qubits
with distance 3. Its stabilizer generators are

$$
S = \{XZZXI,\ IXZZX,\ XIXZZ,\ ZXIXZ\},
$$

and the logical Pauli operators are
$\overline{X} = XXXXX$ and $\overline{Z} = ZZZZZ$.

We prepare $\lvert\overline{0}\rangle$ with a single unitary that maps
computational $\lvert 0\rangle^{\otimes 5}$ onto the logical-zero codeword
(the state amplitudes are taken from the standard [[5,1,3]] encoding).

```{code-cell} ipython3
def prepare_logical_0_state_for_5_1_3_code() -> cirq.Circuit:
    """Prepare |0-bar> of the [[5,1,3]] code as a Cirq circuit."""

    def gram_schmidt(orthogonal_vecs: list[np.ndarray]) -> np.ndarray:
        orthonormal_vecs = [
            vec / np.sqrt(np.vdot(vec, vec)) for vec in orthogonal_vecs
        ]
        dim = np.shape(orthogonal_vecs[0])[0]
        for i in range(dim - len(orthogonal_vecs)):
            new_vec = np.zeros(dim)
            new_vec[i] = 1
            projs = sum(
                np.vdot(new_vec, cached_vec) * cached_vec
                for cached_vec in orthonormal_vecs
            )
            new_vec -= projs
            orthonormal_vecs.append(
                new_vec / np.sqrt(np.vdot(new_vec, new_vec))
            )
        return np.reshape(orthonormal_vecs, (32, 32)).T

    logical_0_state = np.zeros(32)
    # Amplitudes +1/4
    np.put(logical_0_state, [0, 18, 9, 20, 10, 5], 1 / 4)
    # Amplitudes -1/4
    np.put(logical_0_state, [27, 6, 24, 29, 3, 30, 15, 17, 12, 23], -1 / 4)

    logical_1_state = np.zeros(32)
    np.put(logical_1_state, [31, 13, 22, 11, 21, 26], 1 / 4)
    np.put(logical_1_state, [4, 25, 7, 2, 28, 1, 16, 14, 19, 8], -1 / 4)

    matrix = gram_schmidt([logical_0_state, logical_1_state])
    qubits = cirq.LineQubit.range(5)
    return cirq.Circuit(cirq.MatrixGate(matrix)(*qubits))


circuit = prepare_logical_0_state_for_5_1_3_code()
print(circuit)
```

## Check operators and code Hamiltonian

QSE expands around the noisy state using a set of check operators $M_i$.
When the state lives in a stabilizer code, a natural choice is the stabilizer
group (or a generating subset). Mitiq then minimizes the code Hamiltonian

$$
H_c = -\sum_i M_i
$$

inside that subspace to find the least-errored projected state.

For the [[5,1,3]] code we expand the product of generators
$(I+G_1)(I+G_2)(I+G_3)(I+G_4)$ into the full set of 16 group elements and use
those as check operators — corresponding to the highest level of the hierarchy
studied in McClean *et al.*

```{code-cell} ipython3
def get_5_1_3_code_check_operators_and_code_hamiltonian() -> tuple[
    list[PauliString], Observable
]:
    """Check operators and code Hamiltonian for the [[5,1,3]] code."""
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
    check_operators = [PauliString(M, coeff=1) for M in Ms]
    code_hamiltonian = Observable(
        *[PauliString(M, coeff=-1) for M in Ms]
    )
    return check_operators, code_hamiltonian


check_operators, code_hamiltonian = (
    get_5_1_3_code_check_operators_and_code_hamiltonian()
)
print(f"{len(check_operators)} check operators")
```

## Observable and noisy executor

We estimate the logical operator $\overline{Z} = ZZZZZ$. For the prepared
$\lvert\overline{0}\rangle$ state the ideal expectation value is $+1$.

The executor returns a density matrix after applying an uncorrelated
single-qubit depolarizing channel of strength $p$ on every qubit, matching the
noise model used in the reference paper.

```{code-cell} ipython3
observable = Observable(PauliString("ZZZZZ"))


def make_executor(noise_level: float):
    """Return a density-matrix executor with depolarizing noise of strength p."""

    def execute(circuit: QPROGRAM) -> np.ndarray:
        return compute_density_matrix(
            convert_to_mitiq(circuit)[0],
            noise_model_function=cirq.depolarize,
            noise_level=(noise_level,),
        )

    return execute


ideal_executor = make_executor(0.0)
ideal_value = float(np.real(observable.expectation(circuit, ideal_executor)))
print(f"Ideal <Z-bar> = {ideal_value:+.6f}")
```

## Mitigate a single noisy expectation value

At a modest noise rate, compare the unmitigated estimate with the QSE-mitigated
one. `execute_with_qse` measures the matrix elements needed to build the
subspace projector, solves the small generalized eigenvalue problem classically,
and returns the mitigated expectation value $\mathrm{tr}[P\rho P A] /
\mathrm{tr}[P\rho P]$.

```{code-cell} ipython3
noise_level = 0.02
noisy_executor = make_executor(noise_level)

unmitigated = float(np.real(observable.expectation(circuit, noisy_executor)))
mitigated = float(
    np.real(
        qse.execute_with_qse(
            circuit,
            noisy_executor,
            check_operators,
            code_hamiltonian,
            observable,
        )
    )
)

print(f"Noise strength p = {noise_level}")
print(f"Unmitigated <Z-bar> = {unmitigated:+.6f}")
print(f"QSE-mitigated <Z-bar> = {mitigated:+.6f}")
print(f"Ideal               = {ideal_value:+.6f}")
print(f"|error| unmitigated = {abs(unmitigated - ideal_value):.6f}")
print(f"|error| QSE         = {abs(mitigated - ideal_value):.6f}")
```

QSE recovers nearly the ideal logical expectation: the check operators define a
subspace containing the code space, and the code-Hamiltonian minimization
selects the projection that removes the bulk of the depolarizing error.

## Performance across noise strengths

We now sweep the depolarizing probability $p$ and plot the absolute error in
$\langle\overline{Z}\rangle$. This mirrors the [[5,1,3]] numerical experiment in
McClean *et al.*, where full stabilizer expansion strongly suppresses logical
error under the same uncorrelated depolarizing channel.

```{code-cell} ipython3
noise_levels = [0.0, 0.01, 0.02, 0.04, 0.06]
unmitigated_vals: list[float] = []
mitigated_vals: list[float] = []

for p in noise_levels:
    executor = make_executor(p)
    unmitigated_vals.append(
        float(np.real(observable.expectation(circuit, executor)))
    )
    if p == 0.0:
        # At zero noise the state is already in the code space.
        mitigated_vals.append(ideal_value)
    else:
        mitigated_vals.append(
            float(
                np.real(
                    qse.execute_with_qse(
                        circuit,
                        executor,
                        check_operators,
                        code_hamiltonian,
                        observable,
                    )
                )
            )
        )

for p, u, m in zip(noise_levels, unmitigated_vals, mitigated_vals):
    print(
        f"p={p:.2f}: unmitigated={u:+.6f}, QSE={m:+.6f}, "
        f"|err|_U={abs(u - ideal_value):.2e}, |err|_QSE={abs(m - ideal_value):.2e}"
    )
```

```{code-cell} ipython3
unmitigated_error = [abs(v - ideal_value) for v in unmitigated_vals]
mitigated_error = [abs(v - ideal_value) for v in mitigated_vals]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(
    noise_levels,
    unmitigated_error,
    "o-",
    label="Unmitigated",
    color="#d62728",
)
ax.plot(
    noise_levels,
    mitigated_error,
    "s-",
    label="QSE",
    color="#1f77b4",
)
ax.set_xlabel(r"Depolarizing probability $p$")
ax.set_ylabel(r"$|\langle\overline{Z}\rangle - 1|$")
ax.set_title(r"QSE mitigation of $\langle\overline{Z}\rangle$ on [[5,1,3]]")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
```

Across this range, QSE keeps the logical error far below the unmitigated curve —
consistent with the strong recovery reported for full stabilizer expansion in
the reference paper (their Fig. 3, $S^{(4)}$ hierarchy level).

## What this tutorial covered

- How to prepare a [[5,1,3]] logical state and choose stabilizer check operators
  plus a code Hamiltonian for Mitiq's QSE API.
- How `qse.execute_with_qse` improves a logical observable under depolarizing
  noise relative to the bare estimate.
- How the mitigated error scales with noise strength for the core paper
  demonstration.

```{note}
The QSE guide notes that QSE can be stacked with other techniques such as
zero-noise extrapolation (ZNE). A dedicated stacking example is left for a
follow-up tutorial; see the [QSE use-case guide](../guide/qse-2-use-case.md)
and discussion on composing QEM methods.
```

For lower-level control (projectors, expectation caches, and the generalized
eigenvalue problem), see [What happens when I use QSE?](../guide/qse-4-low-level.md)
and [What is the theory behind QSE?](../guide/qse-5-theory.md).
