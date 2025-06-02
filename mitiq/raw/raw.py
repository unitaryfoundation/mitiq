# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Run experiments without error mitigation."""

from collections.abc import Callable

from mitiq import QPROGRAM, Executor, Observable, QuantumResult


def execute(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
) -> float:
    """Evaluates the expectation value associated to the input circuit
    without using error mitigation.

    The only purpose of this function is to provide the same interface for
    non-error-mitigated values as the rest of the techniques in Mitiq. This
    is useful when comparing error-mitigated results to non-error-mitigated
    results.

    Args:
        circuit: The circuit to run.
        executor: Executes a circuit and returns a `QuantumResult`.
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
    """
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    return executor.evaluate(circuit, observable)[0]
