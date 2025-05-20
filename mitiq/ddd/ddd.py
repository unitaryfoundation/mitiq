# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level digital dynamical decoupling (DDD) tools."""

from collections.abc import Callable
from functools import partial, wraps
from typing import Any

import numpy as np

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.ddd.insertion import insert_ddd_sequences


def execute_with_ddd(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    rule: Callable[[int], QPROGRAM],
    rule_args: dict[str, Any] | None = None,
    num_trials: int = 1,
    full_output: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    r"""Estimates the error-mitigated expectation value associated to the
    input circuit, via the application of digital dynamical decoupling (DDD).

    Args:
        circuit: The input circuit to execute with DDD.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If None,
            the ``executor`` must return an expectation value. Otherwise,
            the ``QuantumResult`` returned by ``executor`` is used to compute
            the expectation of the observable.
        rule: A function that takes as main argument a slack length (i.e. the
            number of idle moments) of a slack window (i.e. a single-qubit idle
            window in a circuit) and returns the DDD sequence of gates to be
            applied in that window. Mitiq provides standard built-in rules
            that can be directly imported from ``mitiq.ddd.rules``.
        rule_args: An optional dictionary of keyword arguments for ``rule``.
        num_trials: The number of independent experiments to average over.
            A number larger than 1 can be useful to average over multiple
            applications of a rule returning non-deterministic DDD sequences.
        full_output: If ``False`` only the mitigated expectation value is
            returned. If ``True`` a dictionary containing all DDD data is
            returned too.

    Returns:
        The tuple ``(ddd_value, ddd_data)`` where ``ddd_value`` is the
        expectation value estimated with DDD and ``ddd_data`` is a dictionary
        containing all the raw data involved in the DDD process (e.g. the
        circuit filled with DDD sequences). If ``full_output`` is false,
        only ``ddd_value`` is returned.
    """
    # Initialize executor
    if rule_args is None:
        rule_args = {}
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    # Insert DDD sequences in (a copy of) the input circuit
    circuits_with_ddd = construct_circuits(
        circuit, rule, rule_args, num_trials
    )

    results = executor.evaluate(
        circuits_with_ddd,
        observable,
        force_run_all=True,
    )

    assert len(results) == num_trials

    ddd_value = combine_results(results)

    if not full_output:
        return ddd_value

    ddd_data = {
        "ddd_value": ddd_value,
        "ddd_trials": results,
        "circuits_with_ddd": circuits_with_ddd,
    }
    return ddd_value, ddd_data


def combine_results(results: list[float]) -> float:
    """Averages over the DDD results to get the expectation value from using
    DDD.

    Args:
        results: Results as obtained from running circuits.

    Returns:
        The expectation value estimated with DDD.
    """
    return float(np.average(results))


def construct_circuits(
    circuit: QPROGRAM,
    rule: Callable[[int], QPROGRAM],
    rule_args: dict[str, Any] | None = None,
    num_trials: int = 1,
) -> list[QPROGRAM]:
    """Generates a list of circuits with DDD sequences inserted.

    Args:
        circuit: The quantum circuit to be modified with DD.
        rule: A function that takes as main argument a slack length (i.e. the
            number of idle moments) of a slack window (i.e. a single-qubit idle
            window in a circuit) and returns the DDD sequence of gates to be
            applied in that window.
        rule_args: An optional dictionary of keyword arguments for ``rule``.
        num_trials: The number of circuits to generate with DDD insertions.

    Returns:
        A list of circuits with DDD inserted.
    """
    if rule_args is None:
        rule_args = {}
    rule_partial: Callable[[int], QPROGRAM]
    rule_partial = partial(rule, **rule_args)

    # Insert DDD sequences in (a copy of) the input circuit
    circuits_with_ddd = [
        insert_ddd_sequences(circuit, rule_partial) for _ in range(num_trials)
    ]

    return circuits_with_ddd


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    rule: Callable[[int], QPROGRAM],
    rule_args: dict[str, Any] | None = None,
    num_trials: int = 1,
    full_output: bool = False,
) -> Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with digital dynamical decoupling (DDD).

    Args:
        executor: A function that executes a circuit and returns the
            unmitigated `QuantumResult` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        rule: A function that takes as main argument a slack length (i.e. the
            number of idle moments) of a slack window (i.e. a single-qubit idle
            window in a circuit) and returns the DDD sequence of gates to be
            applied in that window. Mitiq provides standard built-in rules
            that can be directly imported from `mitiq.ddd.rules`.
        rule_args: An optional dictionary of keyword arguments for `rule`.
        num_trials: The number of independent experiments to average over.
            A number larger than 1 can be useful to average over multiple
            applications of a rule returning non-deterministic DDD sequences.
        full_output: If False only the mitigated expectation value is returned.
            If True a dictionary containing all DDD data is returned too.

    Returns:
        The error-mitigated version of the input executor.
    """
    if rule_args is None:
        rule_args = {}
    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(
            circuit: QPROGRAM,
        ) -> float | tuple[float, dict[str, Any]]:
            return execute_with_ddd(
                circuit,
                executor,
                observable,
                rule=rule,
                rule_args=rule_args,
                num_trials=num_trials,
                full_output=full_output,
            )

    else:

        @wraps(executor)
        def new_executor(
            circuits: list[QPROGRAM],
        ) -> list[float | tuple[float, dict[str, Any]]]:
            return [
                execute_with_ddd(
                    circuit,
                    executor,
                    observable,
                    rule=rule,
                    rule_args=rule_args,
                    num_trials=num_trials,
                    full_output=full_output,
                )
                for circuit in circuits
            ]

    return new_executor


def ddd_decorator(
    observable: Observable | None = None,
    *,
    rule: Callable[[int], QPROGRAM],
    rule_args: dict[str, Any] | None = None,
    num_trials: int = 1,
    full_output: bool = False,
) -> Callable[
    [Callable[[QPROGRAM], QuantumResult]],
    Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]],
]:
    """Decorator which adds an error-mitigation layer based on digital
    dynamical decoupling (DDD) to an executor function, i.e., a function which
    executes a quantum circuit with an arbitrary backend and returns a
    ``QuantumResult`` (e.g. an expectation value).

    Args:
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        rule: A function that takes as main argument a slack length (i.e. the
            number of idle moments) of a slack window (i.e. a single-qubit idle
            window in a circuit) and returns the DDD sequence of gates to be
            applied in that window. Mitiq provides standard built-in rules
            that can be directly imported from `mitiq.ddd.rules`.
        rule_args: An optional dictionary of keyword arguments for `rule`.
        num_trials: The number of independent experiments to average over.
            A number larger than 1 can be useful to average over multiple
            applications of a rule returning non-deterministic DDD sequences.
        full_output: If False only the mitigated expectation value is returned.
            If True a dictionary containing all DDD data is returned too.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """

    if rule_args is None:
        rule_args = {}

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult],
    ) -> Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]]:
        return mitigate_executor(
            executor,
            observable,
            rule=rule,
            rule_args=rule_args,
            num_trials=num_trials,
            full_output=full_output,
        )

    return decorator
