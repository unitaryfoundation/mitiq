# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for layerwise folding of input circuits to allow for multivariate
extrapolation as defined in :cite:`Russo_2024_LRE`.
"""

import itertools
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
from cirq import Circuit

from mitiq import QPROGRAM
from mitiq.interface import (
    accept_any_qprogram_as_input,
    accept_qprogram_and_validate,
)
from mitiq.utils import _append_measurements, _pop_measurements
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.scaling.folding import _check_foldable


def _get_num_layers_without_measurements(input_circuit: Circuit) -> int:
    """Checks if the circuit has non-terminal measurements and returns the
    number of layers in the input circuit without the terminal measurements.

        Args:
            input_circuit: Circuit of interest.

        Returns:
            num_layers: the number of layers in the input circuit without the
                terminal measurements.

    """

    _check_foldable(input_circuit)
    circuit = deepcopy(input_circuit)
    _pop_measurements(circuit)
    return len(circuit)


def _get_chunks(
    input_circuit: Circuit, num_chunks: int | None = None
) -> list[Circuit]:
    """Splits a circuit into approximately equal chunks.

    Adapted from:
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

    Args:
        input_circuit: Circuit of interest.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of
            circuit layers (default, each layer is a separate chunk).

    Returns:
        split_circuit: Circuit of interest split into approximately equal
            chunks.

    Raises:
        ValueError:
            When the number of chunks for the input circuit is larger than
            the number of layers in the input circuit.

        ValueError:
            When the number of chunks is less than 1.
    """
    num_layers = _get_num_layers_without_measurements(input_circuit)
    if num_chunks is None:
        num_chunks = num_layers

    if num_chunks < 1:
        raise ValueError(
            "Number of chunks should be greater than or equal to 1."
        )

    if num_chunks > num_layers:
        raise ValueError(
            f"Number of chunks {num_chunks} cannot be greater than the number"
            f" of layers {num_layers}."
        )

    k, m = divmod(num_layers, num_chunks)
    return [
        input_circuit[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_chunks)
    ]


@accept_any_qprogram_as_input
def get_scale_factor_vectors(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: int | None = None,
) -> list[tuple[Any, ...]]:
    """Returns the patterned scale factor vectors required for multivariate
    extrapolation.

    Args:
        input_circuit: Quantum circuit to be scaled.
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap required by unitary folding.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of
            circuit layers (default, each layer is a separate chunk).

    Returns:
        scale_factor_vectors: A vector of scale factors where each
            component in the vector corresponds to the layer in the input
            circuit.
    """

    circuit_chunks = _get_chunks(input_circuit, num_chunks)
    num_layers = len(circuit_chunks)

    # Find the exponents of all the monomial terms required for the folding
    # pattern.
    pattern_full = []
    for i in range(degree + 1):
        for j in itertools.combinations_with_replacement(range(num_layers), i):
            pattern = np.zeros(num_layers, dtype=int)
            # Get the monomial terms in graded lexicographic order.
            for index in j:
                pattern[index] += 1
            # Use the fold multiplier on the folding pattern to determine which
            # layers will be scaled.
            pattern_full.append(tuple(fold_multiplier * pattern))

    # Get the scale factor vectors.
    # The layers are scaled as 2n+1 due to unitary folding.
    return [
        tuple(2 * num_folds + 1 for num_folds in pattern)
        for pattern in pattern_full
    ]


def _multivariate_layer_scaling(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: int | None = None,
    folding_method: Callable[
        [QPROGRAM, float], QPROGRAM
    ] = fold_gates_at_random,
) -> list[Circuit]:
    r"""
    Defines the noise scaling function required for Layerwise Richardson
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
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap required by unitary folding.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of circuit
            layers (default, each layer is a separate chunk).
        folding_method: Unitary folding method. Default is
            :func:`fold_gates_at_random`.

    Returns:
        Multiple folded variations of the input circuit.

    Raises:
        ValueError:
            When the degree for the multinomial is not greater than or
                equal to 1; when the fold multiplier to scale the circuit is
                greater than/equal to 1; when the number of chunks for a
                large circuit is 0 or when the number of chunks in a circuit is
                greater than the number of layers in the input circuit.

    """
    if degree < 1:
        raise ValueError(
            "Multinomial degree must be greater than or equal to 1."
        )
    if fold_multiplier < 1:
        raise ValueError("Fold multiplier must be greater than or equal to 1.")
    circuit_copy = deepcopy(input_circuit)
    terminal_measurements = _pop_measurements(circuit_copy)

    scaling_pattern = get_scale_factor_vectors(
        circuit_copy, degree, fold_multiplier, num_chunks
    )

    chunks = _get_chunks(circuit_copy, num_chunks)

    multiple_folded_circuits = []
    for scale_factor_vector in scaling_pattern:
        folded_circuit = Circuit()
        for chunk, scale_factor in zip(chunks, scale_factor_vector):
            if scale_factor == 1:
                folded_circuit += chunk
            else:
                chunks_circ = Circuit(chunk)
                folded_chunk_circ = folding_method(chunks_circ, scale_factor)
                folded_circuit += folded_chunk_circ
        _append_measurements(folded_circuit, terminal_measurements)
        multiple_folded_circuits.append(folded_circuit)

    return multiple_folded_circuits


multivariate_layer_scaling = accept_qprogram_and_validate(
    _multivariate_layer_scaling, one_to_many=True
)
