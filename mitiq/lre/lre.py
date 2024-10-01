# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Extrapolation methods for Layerwise Richardson Extrapolation (LRE)"""

from functools import wraps
from typing import Any, Callable, Optional, Union

import numpy as np
from cirq import Circuit

from mitiq import QPROGRAM
from mitiq.lre import (
    multivariate_layer_scaling,
    multivariate_richardson_coefficients,
)
from mitiq.zne.scaling import fold_gates_at_random


def execute_with_lre(
    input_circuit: Circuit,
    executor: Callable[[Circuit], float],
    degree: int,
    fold_multiplier: int,
    folding_method: Callable[
        [QPROGRAM, float], QPROGRAM
    ] = fold_gates_at_random,  # type: ignore [has-type]
    num_chunks: Optional[int] = None,
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
        input_circuit: Circuit to be scaled.
        executor: Executes a circuit and returns a `float`
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        folding_method: Unitary folding method. Default is
            :func:`fold_gates_at_random`.
        num_chunks: Number of desired approximately equal chunks. When the
            number of chunks is the same as the layers in the input circuit,
            the input circuit is unchanged.


    Returns:
        Error-mitigated expectation value

    """
    noise_scaled_circuits = multivariate_layer_scaling(
        input_circuit, degree, fold_multiplier, num_chunks, folding_method
    )

    linear_combination_coeffs = multivariate_richardson_coefficients(
        input_circuit, degree, fold_multiplier, num_chunks
    )

    # verify the linear combination coefficients and the calculated expectation
    # values have the same length
    if len(noise_scaled_circuits) != len(  # pragma: no cover
        linear_combination_coeffs
    ):
        raise AssertionError(
            "The number of expectation values are not equal "
            + "to the number of coefficients required for "
            + "multivariate extrapolation."
        )

    lre_exp_values = []
    for scaled_circuit in noise_scaled_circuits:
        circ_exp_val = executor(scaled_circuit)
        lre_exp_values.append(circ_exp_val)

    return np.dot(lre_exp_values, linear_combination_coeffs)


def mitigate_executor(
    executor: Callable[[Circuit], float],
    degree: int,
    fold_multiplier: int,
    folding_method: Callable[
        [Union[Any], float], Union[Any]
    ] = fold_gates_at_random,
    num_chunks: Optional[int] = None,
) -> Callable[[Circuit], float]:
    """Returns a modified version of the input `executor` which is
    error-mitigated with layerwise richardson extrapolation (LRE).

    Args:
        input_circuit: Circuit to be scaled.
        executor: Executes a circuit and returns a `float`
        degree: Degree of the multivariate polynomial.
        fold_multiplier Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        folding_method: Unitary folding method. Default is
            :func:`fold_gates_at_random`.
        num_chunks: Number of desired approximately equal chunks. When the
            number of chunks is the same as the layers in the input circuit,
            the input circuit is unchanged.


    Returns:
        Error-mitigated version of the circuit executor.
    """

    @wraps(executor)
    def new_executor(input_circuit: Circuit) -> float:
        return execute_with_lre(
            input_circuit,
            executor,
            degree,
            fold_multiplier,
            folding_method,
            num_chunks,
        )

    return new_executor


def lre_decorator(
    degree: int,
    fold_multiplier: int,
    folding_method: Callable[[Circuit, float], Circuit] = fold_gates_at_random,
    num_chunks: Optional[int] = None,
) -> Callable[[Callable[[Circuit], float]], Callable[[Circuit], float]]:
    """Decorator which adds an error-mitigation layer based on
    layerwise richardson extrapolation (LRE).

    Args:
        input_circuit: Circuit to be scaled.
        executor: Executes a circuit and returns a `float`
        degree: Degree of the multivariate polynomial.
        fold_multiplier Scaling gap value required for unitary folding which
            is used to generate the scale factor vectors.
        folding_method: Unitary folding method. Default is
            :func:`fold_gates_at_random`.
        num_chunks: Number of desired approximately equal chunks. When the
            number of chunks is the same as the layers in the input circuit,
            the input circuit is unchanged.


    Returns:
        Error-mitigated decorator.
    """

    def decorator(
        executor: Callable[[Circuit], float],
    ) -> Callable[[Circuit], float]:
        return mitigate_executor(
            executor,
            degree,
            fold_multiplier,
            folding_method,
            num_chunks,
        )

    return decorator
