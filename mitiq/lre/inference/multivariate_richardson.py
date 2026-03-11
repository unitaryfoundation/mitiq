# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for multivariate richardson extrapolation as defined in
:cite:`Russo_2024_LRE`.
"""

import warnings
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from cirq import Circuit

from mitiq.interface import accept_any_qprogram_as_input
from mitiq.lre.multivariate_scaling.layerwise_folding import (
    get_scale_factor_vectors,
)


def _full_monomial_basis_term_exponents(
    num_layers: int, degree: int
) -> list[tuple[int, ...]]:
    """Finds exponents of monomial terms required to create the sample matrix
    as defined in Section IIB of :cite:`Russo_2024_LRE`.

    $Mj(λ_i, d)$ is the basis of monomial terms for $l$-layers in the input
    circuit up to a specific degree $d$. The linear combination defines our
    polynomial of interest. In general, the number of monomial terms for a
    variable $l$ up to degree $d$  can be determined through the Stars and
    Bars method.

    We assume the terms in the monomial basis are arranged in a graded
    lexicographic order such that the terms with the highest total degree are
    considered to be the largest and the remaining terms are arranged in
    lexicographic order.

    For `degree=2, num_layers=2`, the monomial terms basis are
    ${1, x_1, x_2, x_1**2, x_1x_2, x_2**2}$ i.e. the function returns the
    exponents of x_1, x_2 as
    `[(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]`.
    """
    exponents = {
        exps
        for exps in product(range(degree + 1), repeat=num_layers)
        if sum(exps) <= degree
    }

    return sorted(exponents, key=lambda term: (sum(term), term[::-1]))


@accept_any_qprogram_as_input
def sample_matrix(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: int | None = None,
) -> npt.NDArray[Any]:
    r"""
    Defines the square sample matrix required for multivariate extrapolation as
    defined in :cite:`Russo_2024_LRE`.

    The number of monomial terms should be equal to the
    number of scale factor vectors such that the monomial terms define the rows
    and the scale factor vectors define the columns.

    Args:
        input_circuit: Quantum circuit to be scaled.
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap required by unitary folding.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the
            number of circuit layers (default, each layer is a separate chunk).

    Returns:
        Matrix of the evaluated monomial basis terms from the scale factor
            vectors.

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

    scale_factor_vectors = get_scale_factor_vectors(
        input_circuit, degree, fold_multiplier, num_chunks
    )
    num_layers = len(scale_factor_vectors[0])

    # Evaluate the monomial terms using the values in the scale factor vectors
    # and insert in the sample matrix
    # each row is specific to each scale factor vector
    # each column is a term in the monomial basis
    variable_exp = _full_monomial_basis_term_exponents(num_layers, degree)
    sample_matrix = np.empty((len(variable_exp), len(variable_exp)))

    for i, scale_factors in enumerate(scale_factor_vectors):
        for j, exponent in enumerate(variable_exp):
            evaluated_terms = []
            for base, exp in zip(scale_factors, exponent):
                # raise scale factor value by the exponent dict value
                evaluated_terms.append(base**exp)
            sample_matrix[i, j] = np.prod(evaluated_terms)

    # verify the matrix is square
    mat_row, mat_cols = sample_matrix.shape
    assert mat_row == mat_cols

    return sample_matrix


@accept_any_qprogram_as_input
def multivariate_richardson_coefficients(
    input_circuit: Circuit,
    degree: int,
    fold_multiplier: int,
    num_chunks: int | None = None,
) -> list[float]:
    r"""
    Defines the function to find the linear combination coefficients from the
    sample matrix as required for multivariate extrapolation (defined in
    :cite:`Russo_2024_LRE`).

    We use the sample matrix to find the constants of linear combination
    $c = (c_1, c_2, …, c_M)$ associated with a known vector of noisy
    expectation values :math:`z = (\langle O(λ_1)\rangle,
    \langle O(λ_2)\rangle, ..., \langle O(λ_M)\rangle)^T`.

    The coefficients are found through the ratio of the determinants of $M_i$
    and the sample matrix. The new matrix $M_i$ is defined by replacing the ith
    row of the sample matrix with $e_1 = (1, 0, 0,..., 0)$.


    Args:
        input_circuit: Circuit to be scaled.
        degree: Degree of the multivariate polynomial.
        fold_multiplier: Scaling gap required by unitary folding.
        num_chunks: The number of equally-sized circuit chunks. Noise
            scaling is applied to each chunk independently. Ranges from 1
            (all gates in one chunk, similar to ZNE) to the number of circuit
            layers (default, each layer is a separate chunk).

    Returns:
        List of the evaluated monomial basis terms using the scale factor
            vectors.
    """
    input_sample_matrix = sample_matrix(
        input_circuit, degree, fold_multiplier, num_chunks
    )
    num_layers = len(
        get_scale_factor_vectors(
            input_circuit, degree, fold_multiplier, num_chunks
        )
    )
    try:
        det = np.linalg.det(input_sample_matrix)
    except RuntimeWarning:  # pragma: no cover
        # taken from https://stackoverflow.com/a/19317237
        warnings.warn(  # pragma: no cover
            "To account for overflow error, required determinant of "
            + "large sample matrix is calculated through "
            + "`np.linalg.slogdet`."
        )
        sign, logdet = np.linalg.slogdet(
            input_sample_matrix
        )  # pragma: no cover
        det = sign * np.exp(logdet)  # pragma: no cover

    if np.isinf(det):
        raise ValueError(  # pragma: no cover
            "Determinant of sample matrix cannot be calculated as "
            + "the matrix is too large. Consider chunking your"
            + " input circuit. "
        )
    assert det != 0.0
    coeff_list = []
    # replace a row of the sample matrix with [1, 0, 0, .., 0]
    for i in range(num_layers):
        sample_matrix_copy = input_sample_matrix.copy()
        sample_matrix_copy[i] = np.array([[1] + [0] * (num_layers - 1)])
        coeff_list.append(
            np.linalg.det(sample_matrix_copy)
            / np.linalg.det(input_sample_matrix)
        )

    return coeff_list
