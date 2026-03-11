# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Readout Confusion Inversion."""

from collections.abc import Callable, Sequence
from functools import wraps
from types import MethodType
from typing import cast

import numpy as np
import numpy.typing as npt

from mitiq import QPROGRAM, MeasurementResult
from mitiq.executor.executor import Executor
from mitiq.observable.observable import Observable
from mitiq.rem.inverse_confusion_matrix import mitigate_measurements


def execute_with_rem(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], MeasurementResult],
    observable: Observable,
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> float:
    """Returns the readout error mitigated expectation value utilizing an
    inverse confusion matrix.

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        observable: Observable to compute the expectation value of (required).
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        The expectation value estimated with REM.
    """
    if not isinstance(executor, Executor):
        executor_obj = Executor(executor)
    else:
        executor_obj = executor

    executor_with_rem = mitigate_executor(
        executor_obj, inverse_confusion_matrix=inverse_confusion_matrix
    )
    # Since the input is an Executor the output is an Executor
    executor_with_rem = cast(Executor, executor_with_rem)

    return executor_with_rem.evaluate(circuit, observable)[0]


def mitigate_executor(
    executor: Executor | Callable[[QPROGRAM], MeasurementResult],
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> Executor | Callable[[QPROGRAM], MeasurementResult]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with readout confusion inversion (RCI).
    The type of the output executor will be equal to the type of
    the input executor: an :class:`.Executor` object or a Python callable.

    Args:
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``MeasurementResult``.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        The error-mitigated version of the input executor.
    """
    # We always mitigate an Executor object but, to preserve the input type,
    # we eventually return a callable if the input executor is a callable.
    if not isinstance(executor, Executor):
        executor_obj = Executor(executor)
    else:
        executor_obj = executor

    def post_run(
        self: Executor,
        results: Sequence[MeasurementResult],
    ) -> Sequence[MeasurementResult]:
        return [
            mitigate_measurements(res, inverse_confusion_matrix)
            for res in results
        ]

    setattr(executor_obj, "_post_run", MethodType(post_run, executor_obj))

    if isinstance(executor, Executor):
        new_executor = executor_obj

    elif not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(circuit: QPROGRAM) -> MeasurementResult:
            result = cast(MeasurementResult, executor_obj.run([circuit])[0])
            return result

    elif executor_obj.can_batch:

        @wraps(executor)
        def new_executor(
            circuits: list[QPROGRAM],
        ) -> Sequence[MeasurementResult]:
            results = executor_obj.run(circuits)
            return cast(Sequence[MeasurementResult], results)

    return new_executor


def rem_decorator(
    *,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> Callable[
    [Callable[[QPROGRAM], MeasurementResult]],
    Callable[[QPROGRAM], MeasurementResult],
]:
    """Decorator which adds an error-mitigation layer based on readout
    confusion inversion (RCI) to an executor function, i.e., a function
    which executes a quantum circuit with an arbitrary backend and returns
    a ``MeasurementResult``.

    Args:
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results
            (required).

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """
    # NOTE: most decorators check for whether the decorator has been used
    #   without parenthesis, but that is not possible with this decorator
    #   since arguments are required.

    def decorator(
        executor: Callable[[QPROGRAM], MeasurementResult],
    ) -> Callable[[QPROGRAM], MeasurementResult]:
        mitigated_executor = mitigate_executor(
            executor,
            inverse_confusion_matrix=inverse_confusion_matrix,
        )
        return cast(
            Callable[[QPROGRAM], MeasurementResult],
            mitigated_executor,
        )

    return decorator
