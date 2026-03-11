# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


"""High-level Quantum Susbapce Expansion tools."""

from collections.abc import Callable, Sequence
from functools import wraps

from mitiq import QPROGRAM, Executor, Observable, PauliString, QuantumResult
from mitiq.qse.qse_utils import (
    get_expectation_value_for_observable,
    get_projector,
)


def execute_with_qse(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
    pauli_string_to_expectation_cache: dict[PauliString, complex]
    | None = None,
) -> float:
    """Function for the calculation of an observable from some circuit of
    interest to be mitigated with quantum subspace expansion (QSE).

    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns a `QuantumResult`.
        check_operators: List of check operators that define the
            stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value of.
        pauli_string_to_expectation_cache: Cache for expectation values of
            Pauli strings used to compute the projector and the observable.

    Returns:
        The expectation value estimated with QSE.
    """
    if pauli_string_to_expectation_cache is None:
        pauli_string_to_expectation_cache = {}

    projector = get_projector(
        circuit,
        executor,
        check_operators,
        code_hamiltonian,
        pauli_string_to_expectation_cache,
    )
    # Compute the expectation value of the observable: <P O P>
    pop = get_expectation_value_for_observable(
        circuit,
        executor,
        projector * observable * projector,
        pauli_string_to_expectation_cache,
    )
    # Compute the normalization factor: <P P>
    pp = get_expectation_value_for_observable(
        circuit,
        executor,
        projector * projector,
        pauli_string_to_expectation_cache,
    )
    return pop / pp


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
    pauli_string_to_expectation_cache: dict[PauliString, complex]
    | None = None,
) -> Callable[[QPROGRAM], float]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with quantum subspace expansion (QSE).

    Args:
        executor: Executes a circuit and returns a `QuantumResult`.
        check_operators: List of check operators that define the
            stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value for.
        pauli_string_to_expectation_cache: Cache for expectation values of
            Pauli strings used to compute the projector and the observable.

    Returns:
        The error-mitigated version of the input executor.
    """
    if pauli_string_to_expectation_cache is None:
        pauli_string_to_expectation_cache = {}

    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(circuit: QPROGRAM) -> float:
            return execute_with_qse(
                circuit,
                executor,
                check_operators,
                code_hamiltonian,
                observable,
                pauli_string_to_expectation_cache,
            )

    else:

        @wraps(executor)
        def new_executor(circuits: list[QPROGRAM]) -> list[float]:
            return [
                execute_with_qse(
                    circuit,
                    executor,
                    check_operators,
                    code_hamiltonian,
                    observable,
                    pauli_string_to_expectation_cache,
                )
                for circuit in circuits
            ]

    return new_executor


def qse_decorator(
    check_operators: Sequence[PauliString],
    code_hamiltonian: Observable,
    observable: Observable,
    pauli_string_to_expectation_cache: dict[PauliString, complex]
    | None = None,
) -> Callable[
    [Callable[[QPROGRAM], QuantumResult]], Callable[[QPROGRAM], float]
]:
    """Decorator which adds an error-mitigation layer based on quantum
    subspace expansion (QSE) to an executor function, i.e., a function which
    executes a quantum circuit with an arbitrary backend and returns a
    ``QuantumResult``.

    Args:
        check_operators: List of check operators that define the
            stabilizer code space.
        code_hamiltonian: Hamiltonian of the code space.
        observable: Observable to compute the mitigated expectation value of.
        pauli_string_to_expectation_cache: Cache for expectation values of
            Pauli strings used to compute the projector and the observable.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """

    if pauli_string_to_expectation_cache is None:
        pauli_string_to_expectation_cache = {}

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult],
    ) -> Callable[[QPROGRAM], float]:
        val = mitigate_executor(
            executor,
            check_operators,
            code_hamiltonian,
            observable,
            pauli_string_to_expectation_cache,
        )
        return val

    return decorator
