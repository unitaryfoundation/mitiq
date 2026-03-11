# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for computing the projector for subspace expansion."""

from collections.abc import Callable, Sequence
from itertools import product

import numpy as np
import numpy.typing as npt
from numpy.linalg import pinv
from scipy.linalg import eigh

from mitiq import QPROGRAM, Executor, Observable, PauliString, QuantumResult


def get_projector(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    pauli_string_to_expectation_cache: dict[PauliString, complex]
    | None = None,
) -> Observable:
    """Computes the projector onto the code space defined by the
    check_operators provided that minimizes the code_hamiltonian.

    Returns: Projector as an Observable.
    """
    if pauli_string_to_expectation_cache is None:
        pauli_string_to_expectation_cache = {}

    S = _compute_overlap_matrix(
        circuit, executor, check_operators, pauli_string_to_expectation_cache
    )
    H = _compute_overlap_matrix(
        circuit,
        executor,
        check_operators,
        pauli_string_to_expectation_cache,
        code_hamiltonian,
    )
    # We only want the smallest eigenvalue and corresponding eigenvector
    _, C = eigh(pinv(S) @ H, subset_by_index=[0, 0])
    # np float type: np.float64

    Cs = C[:, 0]
    projector = Observable(
        *[check_operators[i] * Cs[i] for i in range(len(check_operators))]
    )

    return projector


def get_expectation_value_for_observable(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    observable: PauliString | Observable,
    pauli_expectation_cache: dict[PauliString, complex] | None = None,
) -> float:
    """Provide pauli_string_to_expectation_cache if you want to take advantage
    of caching.

    This function modifies pauli_string_to_expectation_cache in place.
    """

    cache: dict[PauliString, complex] = (
        pauli_expectation_cache if pauli_expectation_cache is not None else {}
    )

    final_executor = (
        executor if isinstance(executor, Executor) else Executor(executor)
    )

    def get_expectation_value_for_one_pauli(
        pauli_string: PauliString,
    ) -> float:
        cache_key = pauli_string.with_coeff(1)
        cache[cache_key] = final_executor.evaluate(
            circuit, Observable(cache_key)
        )[0]
        return (cache[cache_key] * pauli_string.coeff).real

    paulis = (
        [observable]
        if isinstance(observable, PauliString)
        else observable.paulis
    )
    expectation_value = sum(
        get_expectation_value_for_one_pauli(pauli) for pauli in paulis
    )
    return expectation_value


def _compute_overlap_matrix(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    check_operators: Sequence[PauliString],
    pauli_expectation_cache: dict[PauliString, complex] | None = None,
    code_hamiltonian: Observable | None = None,
) -> npt.NDArray[np.float64]:
    num_ops = len(check_operators)

    if pauli_expectation_cache is None:
        pauli_expectation_cache = {}

    H = np.zeros((num_ops, num_ops))
    # Hij = ⟨Ψ|Mi† H Mj|Ψ⟩
    for i, j in product(range(num_ops), repeat=2):
        observable: PauliString | Observable
        if code_hamiltonian:
            observable = (
                check_operators[i] * code_hamiltonian * check_operators[j]
            )
        else:
            observable = check_operators[i] * check_operators[j]
        H[i, j] = get_expectation_value_for_observable(
            circuit, executor, observable, pauli_expectation_cache
        )
    return H
