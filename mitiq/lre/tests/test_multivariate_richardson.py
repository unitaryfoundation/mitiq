# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for multivariate extrapolation inference functions."""

import numpy as np
import pytest
from cirq import Circuit, LineQubit, ops

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import convert_from_mitiq
from mitiq.lre.inference.multivariate_richardson import (
    _full_monomial_basis_term_exponents,
    multivariate_richardson_coefficients,
    sample_matrix,
)
from mitiq.lre.multivariate_scaling.layerwise_folding import (
    multivariate_layer_scaling,
)

qreg1 = LineQubit.range(3)
test_circuit1 = Circuit(
    [ops.H.on_each(*qreg1)],
    [ops.CNOT.on(qreg1[0], qreg1[1])],
)
test_circuit2 = Circuit(
    [ops.H.on_each(*qreg1)],
    [ops.CNOT.on(qreg1[0], qreg1[1])],
    [ops.X.on(qreg1[2])],
    [ops.TOFFOLI.on(*qreg1)],
)


@pytest.mark.parametrize(
    "test_num_layers, test_degree, expected",
    [
        (1, 1, [(0,), (1,)]),
        (
            2,
            2,
            [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
        ),
        (
            3,
            2,
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (2, 0, 0),
                (1, 1, 0),
                (0, 2, 0),
                (1, 0, 1),
                (0, 1, 1),
                (0, 0, 2),
            ],
        ),
    ],
)
def test_basis_exp(test_num_layers, test_degree, expected):
    assert (
        _full_monomial_basis_term_exponents(test_num_layers, test_degree)
        == expected
    )


@pytest.mark.parametrize(
    "test_num_layers, test_degree",
    [(1, 1), (2, 2), (3, 2), (10, 4)],
    # TO DO: Note need to add (100, 2) here.
    # This makes the unit test very slow.
)
def test_basis_exp_len(test_num_layers, test_degree):
    calc_dict = _full_monomial_basis_term_exponents(
        test_num_layers, test_degree
    )
    for i in calc_dict:
        assert len(i) == test_num_layers


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@pytest.mark.parametrize(
    "test_circ, test_degree, expected_matrix",
    [
        (
            test_circuit1,
            2,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 3, 1, 9, 3, 1],
                    [1, 1, 3, 1, 3, 9],
                    [1, 5, 1, 25, 5, 1],
                    [1, 3, 3, 9, 9, 9],
                    [1, 1, 5, 1, 5, 25],
                ]
            ),
        ),
        (
            test_circuit2,
            2,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 3, 1, 1, 9, 3, 1, 3, 1, 1],
                    [1, 1, 3, 1, 1, 3, 9, 1, 3, 1],
                    [1, 1, 1, 3, 1, 1, 1, 3, 3, 9],
                    [1, 5, 1, 1, 25, 5, 1, 5, 1, 1],
                    [1, 3, 3, 1, 9, 9, 9, 3, 3, 1],
                    [1, 3, 1, 3, 9, 3, 1, 9, 3, 9],
                    [1, 1, 5, 1, 1, 5, 25, 1, 5, 1],
                    [1, 1, 3, 3, 1, 3, 9, 3, 9, 9],
                    [1, 1, 1, 5, 1, 1, 1, 5, 5, 25],
                ]
            ),
        ),
    ],
)
def test_sample_matrix(test_circ, test_degree, expected_matrix, circuit_type):
    converted_circuit = convert_from_mitiq(test_circ, circuit_type)
    assert np.allclose(
        expected_matrix, sample_matrix(converted_circuit, test_degree, 1)
    )


@pytest.mark.parametrize(
    "test_circ, test_degree, test_fold_multiplier, expected_matrix",
    [
        (
            test_circuit1,
            2,
            3,
            [
                1.5555555555555578,
                -0.38888888888888934,
                -0.38888888888888934,
                0.09722222222222215,
                0.027777777777777804,
                0.09722222222222232,
            ],
        ),
        (
            test_circuit2,
            2,
            2,
            [
                2.4062499999999956,
                -0.6874999999999987,
                -0.6874999999999987,
                -0.6874999999999987,
                0.15624999999999956,
                0.06249999999999997,
                0.06249999999999997,
                0.15624999999999956,
                0.06249999999999997,
                0.15624999999999956,
            ],
        ),
    ],
)
def test_coeffs(test_circ, test_degree, test_fold_multiplier, expected_matrix):
    assert np.allclose(
        expected_matrix,
        multivariate_richardson_coefficients(
            test_circ, test_degree, test_fold_multiplier
        ),
        atol=1e-3,
    )

    assert np.isclose(
        sum(
            multivariate_richardson_coefficients(
                test_circ, test_degree, test_fold_multiplier
            )
        ),
        1.0,
    )


@pytest.mark.parametrize(
    "test_input, test_degree, test_fold_multiplier, error_msg",
    [
        (
            test_circuit1,
            0,
            1,
            "Multinomial degree must be greater than or equal to 1.",
        ),
        (
            test_circuit1,
            1,
            0,
            "Fold multiplier must be greater than or equal to 1.",
        ),
    ],
)
def test_invalid_degree_fold_multiplier_sample_matrix(
    test_input, test_degree, test_fold_multiplier, error_msg
):
    """Ensures that the args for the sample matrix
    an error for an invalid value."""
    with pytest.raises(ValueError, match=error_msg):
        sample_matrix(test_input, test_degree, test_fold_multiplier)


def test_lre_inference_with_chunking():
    """Verify the dimension of a chunked sample matrix for some input circuit
    is smaller than the non-chunked sample matrix for the same input circuit.
    """
    circ = test_circuit1 * 7
    chunked_sample_matrix_dim = sample_matrix(circ, 2, 2, num_chunks=4).shape
    non_chunked_sample_matrix_dim = sample_matrix(circ, 2, 2).shape
    assert chunked_sample_matrix_dim[0] < non_chunked_sample_matrix_dim[0]


def test_sample_matrix_numerical_stability():
    """Verify sample matrix function works for very large circuits."""
    large_circuit = Circuit([ops.H.on(LineQubit(i)) for i in range(10000)])
    matrix = sample_matrix(large_circuit, 5, 10000)
    assert np.isfinite(matrix).all()
    assert not np.isnan(matrix).any()


@pytest.mark.parametrize("num_chunks", [2, 3])
def test_eval(num_chunks):
    """Verify the number of calculated linear combination coefficients matches
    to the number of scaled chunked circuits."""
    coeffs = multivariate_richardson_coefficients(
        7 * test_circuit2, 2, 2, num_chunks
    )
    multiple_scaled_circuits = multivariate_layer_scaling(
        7 * test_circuit2, 2, 2, num_chunks
    )
    assert len(coeffs) == len(multiple_scaled_circuits)
    assert np.isclose(sum(coeffs), 1.0)
