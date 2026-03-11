# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for inverse confusion matrix helper functions."""

from functools import reduce
from math import isclose

import numpy as np
import pytest

from mitiq import MeasurementResult
from mitiq.rem.inverse_confusion_matrix import (
    bitstrings_to_probability_vector,
    closest_positive_distribution,
    generate_inverse_confusion_matrix,
    generate_tensored_inverse_confusion_matrix,
    mitigate_measurements,
    sample_probability_vector,
)


def test_sample_probability_vector_invalid_size():
    with pytest.raises(ValueError, match="power of 2"):
        sample_probability_vector([1 / 3, 1 / 3, 1 / 3], 3)


def test_sample_probability_vector_single_qubit():
    bitstrings = sample_probability_vector(np.array([1, 0]), 10)
    assert all(b == "0" for b in bitstrings)

    bitstrings = sample_probability_vector(np.array([0, 1]), 10)
    assert all(b == "1" for b in bitstrings)

    np.random.seed(0)
    bitstrings = sample_probability_vector(np.array([0.5, 0.5]), 1000)
    assert sum(int(b) for b in bitstrings) == 483


def test_sample_probability_vector_two_qubits():
    bitstrings = sample_probability_vector(np.array([1, 0, 0, 0]), 10)
    assert all(b == "00" for b in bitstrings)

    bitstrings = sample_probability_vector(np.array([0, 1, 0, 0]), 10)
    assert all(b == "01" for b in bitstrings)

    bitstrings = sample_probability_vector(np.array([0, 0, 1, 0]), 10)
    assert all(b == "10" for b in bitstrings)

    bitstrings = sample_probability_vector(np.array([0, 0, 0, 1]), 10)
    assert all(b == "11" for b in bitstrings)


def test_bitstrings_to_probability_vector():
    pv = bitstrings_to_probability_vector([[0]])
    assert (pv == np.array([1, 0])).all()

    pv = bitstrings_to_probability_vector([[1]])
    assert (pv == np.array([0, 1])).all()

    pv = bitstrings_to_probability_vector([[0], [1]])
    assert (pv == np.array([0.5, 0.5])).all()

    pv = bitstrings_to_probability_vector([[0, 0]])
    assert (pv == np.array([1, 0, 0, 0])).all()

    pv = bitstrings_to_probability_vector([[1, 1]])
    assert (pv == np.array([0, 0, 0, 1])).all()


@pytest.mark.parametrize("_", range(10))
def test_probability_vector_roundtrip(_):
    pv = np.random.rand(4)
    pv /= np.sum(pv)
    assert isclose(
        np.linalg.norm(
            pv
            - bitstrings_to_probability_vector(
                sample_probability_vector(pv, 1000)
            )
        ),
        0,
        abs_tol=0.1,
    )


def test_generate_inverse_confusion_matrix():
    num_qubits = 2
    identity = np.identity(4)
    assert (
        generate_inverse_confusion_matrix(num_qubits, p0=0, p1=0) == identity
    ).all()
    assert (
        generate_inverse_confusion_matrix(num_qubits, p0=1, p1=1)
        == np.flipud(identity)
    ).all()


@pytest.mark.parametrize(
    "num_qubits, confusion_matrices, expected",
    [
        (2, [np.identity(2), np.identity(2)], np.identity(4)),
        (2, [np.identity(4)], np.identity(4)),
        (3, [np.identity(4), np.identity(2)], np.identity(8)),
        # all are faulty qubits, flipping to opposite value
        (
            2,
            [np.flipud(np.identity(2)), np.flipud(np.identity(2))],
            np.flipud(np.identity(4)),
        ),
        (
            3,
            [np.flipud(np.identity(4)), np.flipud(np.identity(2))],
            np.flipud(np.identity(8)),
        ),
        # wrongly sized confusion matrices
        (3, [np.identity(2), np.identity(2)], ValueError),
        # one qubit flips values, one is perfect, one is random
        (
            3,
            [np.flipud(np.identity(2)), np.identity(2), 0.5 * np.ones((2, 2))],
            np.linalg.pinv(
                reduce(
                    np.kron,
                    [
                        np.flipud(np.identity(2)),
                        np.identity(2),
                        0.5 * np.ones((2, 2)),
                    ],
                )
            ),
        ),
    ],
)
def test_generate_tensored_inverse_confusion_matrix(
    num_qubits, confusion_matrices, expected
):
    if expected is ValueError:
        with pytest.raises(ValueError):
            generate_tensored_inverse_confusion_matrix(
                num_qubits, confusion_matrices
            )
    else:
        assert np.allclose(
            generate_tensored_inverse_confusion_matrix(
                num_qubits, confusion_matrices
            ),
            expected,
        )


def test_mitigate_measurements():
    identity = np.identity(4)

    measurements = MeasurementResult([[1, 0]])
    assert mitigate_measurements(measurements, identity) == measurements
    assert mitigate_measurements(measurements, np.flipud(identity)).result == [
        [0, 1]
    ]

    measurements = MeasurementResult([[0, 1]])
    assert mitigate_measurements(measurements, identity) == measurements
    assert mitigate_measurements(measurements, np.flipud(identity)).result == [
        [1, 0]
    ]


def test_closest_positive_distribution():
    inputs = [
        [0.3, 0.7],  # Test optimal input
        [-0.1, 1.1],  # Test negative elements
        [10, 10],  # Test normalization
        [-1, 1, -1, 1],  # Test more elements
        [-1, 0.1, -1, 0.2],  # Non-trivial problem
    ]
    expected = [
        [0.3, 0.7],
        [0, 1],
        [0.5, 0.5],
        [0, 0.5, 0, 0.5],
        [0, 0.450317, 0, 0.549683],
    ]
    for quasi_prob, prob in zip(inputs, expected):
        assert np.allclose(
            closest_positive_distribution(quasi_prob),
            prob,
            atol=1e-5,
        )


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_closest_positive_distribution_error():
    """Test unfeasible problem to trigger error."""
    with pytest.raises(ValueError, match="REM failed to determine"):
        closest_positive_distribution([0, 0])
