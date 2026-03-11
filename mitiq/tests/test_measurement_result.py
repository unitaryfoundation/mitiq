# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for measurement results."""

import numpy as np
import pytest

from mitiq import MeasurementResult


@pytest.mark.parametrize("asarray", (True, False))
@pytest.mark.parametrize("qubit_indices", ((0, 1), (1, 20)))
def test_measurement_result(asarray, qubit_indices):
    bitstrings = [[0, 0], [0, 1], [1, 0]]
    if asarray:
        bitstrings = np.array(bitstrings)
    result = MeasurementResult(bitstrings, qubit_indices=qubit_indices)

    assert result.nqubits == 2
    assert result.qubit_indices == qubit_indices
    assert result.shots == 3
    assert np.allclose(result.result, bitstrings)


def test_measurement_result_bad_qubit_indices():
    with pytest.raises(ValueError, match="MeasurementResult has"):
        MeasurementResult([[0], [1]], qubit_indices=(1, 5))


def test_measurement_result_not_bits():
    with pytest.raises(ValueError, match="should look like"):
        MeasurementResult(result=[[2]])

    with pytest.raises(ValueError, match="should look like"):
        MeasurementResult(result=[[0, 0], [0, 1], [-1, 0]])


def test_measurement_result_invoked_with_dict():
    with pytest.raises(TypeError, match="from_counts"):
        MeasurementResult({"001": 123, "010": 456})


def test_filter_qubits():
    result = MeasurementResult([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert np.allclose(result.filter_qubits([0]), np.array([[0], [0], [1]]))
    assert np.allclose(result.filter_qubits([1]), np.array([[0], [1], [0]]))
    assert np.allclose(result.filter_qubits([2]), np.array([[1], [0], [0]]))

    assert np.allclose(
        result.filter_qubits([0, 1]), np.array([[0, 0], [0, 1], [1, 0]])
    )
    assert np.allclose(
        result.filter_qubits([0, 2]), np.array([[0, 1], [0, 0], [1, 0]])
    )
    assert np.allclose(
        result.filter_qubits([1, 2]), np.array([[0, 1], [1, 0], [0, 0]])
    )


def test_empty():
    result = MeasurementResult([])
    assert result.nqubits == 0
    assert result.shots == 0
    assert result.result == []


def test_convert_to_array():
    bitstrings = [[0, 0], [0, 1], [1, 0]]
    result = MeasurementResult(bitstrings)
    assert np.allclose(result.asarray, np.array(bitstrings))

    assert np.allclose(MeasurementResult([]).asarray, np.array([]))


@pytest.mark.parametrize("qubit_indices", ((0, 1), (1, 20)))
def test_measurement_result_with_strings(qubit_indices):
    """Try using strings instead of lists of integers."""
    bitstrings = ["00", "01", "10"]
    int_bitstrings = [[0, 0], [0, 1], [1, 0]]

    result = MeasurementResult(bitstrings, qubit_indices=qubit_indices)

    assert result.nqubits == 2
    assert result.qubit_indices == qubit_indices
    assert result.shots == 3
    assert result.result == int_bitstrings


@pytest.mark.parametrize("qubit_indices", ((0, 1), (1, 20)))
def test_measurement_result_from_counts(qubit_indices):
    """Test initialization from a dictionary of counts."""
    counts = {"00": 1, "01": 2}
    int_bitstrings = [[0, 0], [0, 1], [0, 1]]

    result = MeasurementResult.from_counts(
        counts=counts,
        qubit_indices=qubit_indices,
    )
    assert result.nqubits == 2
    assert result.qubit_indices == qubit_indices
    assert result.shots == 3
    assert result.result == int_bitstrings


def test_measurement_result_get_counts():
    """Test initialization from a dictionary of counts."""
    counts = {"00": 1, "01": 2}

    int_bitstrings = [[0, 0], [0, 1], [0, 1]]
    result = MeasurementResult(
        result=int_bitstrings,
        qubit_indices=(1, 20),
    )
    assert result.get_counts() == counts
    # Info about qubit indices is expected to be lost
    new_res = MeasurementResult.from_counts(result.get_counts())
    assert new_res.qubit_indices == (0, 1)  # Default values


def test_measurement_result_to_from_dictionary():
    """Test initialization from a dictionary of counts."""
    data = {
        "counts": {"00": 1, "01": 2},
        "shots": 3,
        "nqubits": 2,
        "qubit_indices": (1, 7),
    }
    assert MeasurementResult.from_dict(data).to_dict() == data


def test_measurement_repr_():
    """Test string representation and printing."""
    counts = {"00": 1000, "01": 200}
    result = MeasurementResult.from_counts(
        counts=counts,
        qubit_indices=(1, 20),
    )
    expected = (
        "MeasurementResult: {'nqubits': 2, 'qubit_indices': (1, 20),"
        " 'shots': 1200, 'counts': {'00': 1000, '01': 200}}"
    )
    assert repr(result) == expected
    assert str(result) == expected


def test_measurement_result_prob_distribution():
    """Test initialization from a dictionary of counts."""
    int_bitstrings = [[0, 0], [0, 1], [0, 1], [0, 1]]
    result = MeasurementResult(
        result=int_bitstrings,
        qubit_indices=(1, 20),
    )
    dist = result.prob_distribution()
    assert len(dist) == 2
    assert np.isclose(dist["00"], 0.25)
    assert np.isclose(dist["01"], 0.75)
