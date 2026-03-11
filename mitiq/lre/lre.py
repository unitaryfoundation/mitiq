# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Extrapolation methods for Layerwise Richardson Extrapolation (LRE)"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.lre.inference import (
    multivariate_richardson_coefficients,
)
from mitiq.lre.multivariate_scaling import (
    multivariate_layer_scaling,
)
from mitiq.zne.scaling import fold_gates_at_random


def construct_circuits(
    circuit: QPROGRAM,
    degree: int,
    fold_multiplier: int,
    folding_method: Callable[
        [QPROGRAM, float], QPROGRAM
    ] = fold_gates_at_random,  # type: ignore [has-type]
    num_chunks: int | None = None,
) -> list[QPROGRAM]:
    """Given a circuit, degree, fold_multiplier, folding_method, and
       num_chunks, outputs a list of circuits that will be used in LRE.

    Args:
        circuit: Circuit to be scaled.
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        folding_method: Unitary folding method. Default is
            :func:`mitiq.zne.scaling.folding.fold_gates_at_random`.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of circuit
            layers (default, each layer is a separate chunk).

    Returns:
        The scaled circuits using the
        :func:`mitiq.lre.multivariate_scaling.layerwise_folding.multivariate_layer_scaling`.
    """
    noise_scaled_circuits = multivariate_layer_scaling(
        circuit, degree, fold_multiplier, num_chunks, folding_method
    )
    return noise_scaled_circuits


def combine_results(
    results: list[float],
    circuit: QPROGRAM,
    degree: int,
    fold_multiplier: int,
    num_chunks: int | None = None,
) -> float:
    """Computes the error-mitigated expectation value associated to the
    input results from executing the scaled circuits and using the multivariate
    richardson coeffecients, via the application of Layerwise Richardson
    Extrapolation (LRE).

    Args:
        results: An array storing the results of running the scaled circuits.
        circuit: Circuit to be scaled.
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of circuit
            layers (default, each layer is a separate chunk).

    Returns:
        The expectation value estimated with LRE.
    """
    linear_combination_coeffs = multivariate_richardson_coefficients(
        circuit, degree, fold_multiplier, num_chunks
    )
    return np.dot(results, linear_combination_coeffs)


def execute_with_lre(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    degree: int,
    fold_multiplier: int,
    observable: Observable | None = None,
    folding_method: Callable[
        [QPROGRAM, float], QPROGRAM
    ] = fold_gates_at_random,  # type: ignore [has-type]
    num_chunks: int | None = None,
) -> float:
    r"""
    Defines the executor required for Layerwise Richardson
    Extrapolation as defined in :cite:`Russo_2024_LRE`.

    Note that this method only works for the multivariate extrapolation
    methods. It does not allows a user to choose which layers in the input
    circuit will be scaled.

    .. seealso::

        If you would prefer to choose the layers for unitary
        folding, use :func:`mitiq.zne.scaling.layer_scaling.get_layer_folding`
        instead.

    Args:
        circuit: Circuit to be scaled.
        executor: Executes a circuit and returns a `float`
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value.
            Otherwise, the ``DensityMatrix`` or ``Bitstrings`` returned by
            ``executor`` is used to compute the expectation of the observable.
        folding_method: Unitary folding method. Default is
            :func:`mitiq.zne.scaling.folding.fold_gates_at_random`.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of circuit
            layers (default, each layer is a separate chunk).

    Returns:
        Error-mitigated expectation value

    """
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    noise_scaled_circuits = multivariate_layer_scaling(
        circuit, degree, fold_multiplier, num_chunks, folding_method
    )

    linear_combination_coeffs = multivariate_richardson_coefficients(
        circuit, degree, fold_multiplier, num_chunks
    )

    # verify the linear combination coefficients and the calculated expectation
    # values have the same length
    if len(noise_scaled_circuits) != len(
        linear_combination_coeffs
    ):  # pragma: no cover
        raise AssertionError(
            "The number of expectation values are not equal "
            + "to the number of coefficients required for "
            + "multivariate extrapolation."
        )

    lre_exp_values = executor.evaluate(noise_scaled_circuits, observable)

    return np.dot(lre_exp_values, linear_combination_coeffs)


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    degree: int,
    fold_multiplier: int,
    observable: Observable | None = None,
    folding_method: Callable[[Any, float], Any] = fold_gates_at_random,
    num_chunks: int | None = None,
) -> Callable[[QPROGRAM], float]:
    """Returns a modified version of the input `executor` which is
    error-mitigated with layerwise richardson extrapolation (LRE).

    Args:
        executor: Executes a circuit and returns a `float`.
        degree: Degree of the multivariate polynomial.
        fold_multiplier Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value.
            Otherwise, the ``DensityMatrix`` or ``Bitstrings`` returned by
            ``executor`` is used to compute the expectation of the observable.
        folding_method: Unitary folding method. Default is
            :func:`mitiq.zne.scaling.folding.fold_gates_at_random`.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of circuit
            layers (default, each layer is a separate chunk).

    Returns:
        Error-mitigated version of the circuit executor.
    """

    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(circuit: QPROGRAM) -> float:
            return execute_with_lre(
                circuit,
                executor,
                degree,
                fold_multiplier,
                observable,
                folding_method,
                num_chunks,
            )

    else:

        @wraps(executor)
        def new_executor(circuits: list[QPROGRAM]) -> list[float]:
            return [
                execute_with_lre(
                    circuit,
                    executor,
                    degree,
                    fold_multiplier,
                    observable,
                    folding_method,
                    num_chunks,
                )
                for circuit in circuits
            ]

    return new_executor


def lre_decorator(
    degree: int,
    fold_multiplier: int,
    observable: Observable | None = None,
    folding_method: Callable[
        [QPROGRAM, float], QPROGRAM
    ] = fold_gates_at_random,
    num_chunks: int | None = None,
) -> Callable[
    [Callable[[QPROGRAM], QuantumResult]], Callable[[QPROGRAM], float]
]:
    """Decorator which adds an error-mitigation layer based on
    layerwise richardson extrapolation (LRE).

    Args:
        degree: Degree of the multivariate polynomial.
        fold_multiplier Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value.
            Otherwise, the ``DensityMatrix`` or ``Bitstrings`` returned by
            ``executor`` is used to compute the expectation of the observable.
        folding_method: Unitary folding method. Default is
            :func:`mitiq.zne.scaling.folding.fold_gates_at_random`.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of circuit
            layers (default, each layer is a separate chunk).

    Returns:
        Error-mitigated decorator.
    """

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult],
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(
            executor,
            degree,
            fold_multiplier,
            observable,
            folding_method,
            num_chunks,
        )

    return decorator
