# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for classical post-processing functions for classical shadows."""

import numpy as np

import mitiq
from mitiq.shadows.classical_postprocessing import (
    classical_snapshot,
    expectation_estimation_shadow,
    get_pauli_fidelities,
    get_single_shot_pauli_fidelity,
    shadow_state_reconstruction,
)
from mitiq.utils import operator_ptm_vector_rep


def test_get_single_shot_pauli_fidelity():
    b_list = "01"
    u_list = "XY"
    expected_result = {"00": 1.0, "01": 0.0, "10": 0.0, "11": 0.0}
    assert get_single_shot_pauli_fidelity(b_list, u_list) == expected_result
    b_list = "01101"
    u_list = "XYZYZ"
    assert get_single_shot_pauli_fidelity(b_list, u_list) == {
        "00000": 1.0,
        "10000": 0.0,
        "01000": 0.0,
        "00100": -1.0,
        "00010": 0.0,
        "00001": -1.0,
        "11000": 0.0,
        "10100": 0.0,
        "10010": 0.0,
        "10001": 0.0,
        "01100": 0.0,
        "01010": 0.0,
        "01001": 0.0,
        "00110": 0.0,
        "00101": 1.0,
        "00011": 0.0,
        "11100": 0.0,
        "11010": 0.0,
        "11001": 0.0,
        "10110": 0.0,
        "10101": 0.0,
        "10011": 0.0,
        "01110": 0.0,
        "01101": 0.0,
        "01011": 0.0,
        "00111": 0.0,
        "11110": 0.0,
        "11101": 0.0,
        "11011": 0.0,
        "10111": 0.0,
        "01111": 0.0,
        "11111": 0.0,
    }


def test_get_single_shot_pauli_fidelity_with_locality():
    b_list = "11101"
    u_list = "XYZYZ"
    assert get_single_shot_pauli_fidelity(b_list, u_list, locality=2) == {
        "00000": 1.0,
        "10000": 0.0,
        "01000": 0.0,
        "00100": -1.0,
        "00010": 0.0,
        "00001": -1.0,
        "11000": 0.0,
        "10100": 0.0,
        "10010": 0.0,
        "10001": 0.0,
        "01100": 0.0,
        "01010": 0.0,
        "01001": 0.0,
        "00110": 0.0,
        "00101": 1.0,
        "00011": 0.0,
    }


def test_get_pauli_fidelity():
    calibration_measurement_outcomes = (
        ["00", "01", "10", "11"],
        ["XX", "YY", "ZZ", "XY"],
    )
    k_calibration = 2
    expected_result = {"00": 1, "10": -0.25, "01": 0.25, "11": -0.25}
    result = get_pauli_fidelities(
        calibration_measurement_outcomes, k_calibration
    )
    assert result == expected_result


def test_classical_snapshot_cal():
    b_list_shadow = "01"
    u_list_shadow = "XY"
    f_est = {"00": 1, "01": 1 / 3, "10": 1 / 3, "11": 1 / 9}
    expected_result = operator_ptm_vector_rep(
        np.array(
            [
                [0.25 + 0.0j, 0.0 + 0.75j, 0.75 + 0.0j, 0.0 + 2.25j],
                [0.0 - 0.75j, 0.25 + 0.0j, 0.0 - 2.25j, 0.75 + 0.0j],
                [0.75 + 0.0j, 0.0 + 2.25j, 0.25 + 0.0j, 0.0 + 0.75j],
                [0.0 - 2.25j, 0.75 + 0.0j, 0.0 - 0.75j, 0.25 + 0.0j],
            ]
        )
    )
    np.testing.assert_array_almost_equal(
        classical_snapshot(b_list_shadow, u_list_shadow, f_est),
        expected_result,
    )


def test_classical_snapshot():
    b_list = "01"
    u_list = "XY"
    expected_result = np.array(
        [
            [0.25, 0.75j, 0.75, 2.25j],
            [-0.75j, 0.25, -2.25j, 0.75],
            [0.75, 2.25j, 0.25, 0.75j],
            [-2.25j, 0.75, -0.75j, 0.25],
        ]
    )
    result = classical_snapshot(b_list, u_list, False)
    np.testing.assert_allclose(result, expected_result)


def test_shadow_state_reconstruction():
    bitstrings = ["010", "001", "000"]
    paulistrings = ["XYZ", "ZYX", "YXZ"]
    measurement_outcomes = (bitstrings, paulistrings)

    expected_state = np.array(
        [
            [0.5, -0.5, 0.5, 1.5j, 0.5 - 0.5j, 0, 0, 0],
            [-0.5, 0, 1.5j, -0.25 - 0.75j, 0, -0.25 + 0.25j, 0, 0],
            [0.5, -1.5j, 0.5, -0.5, -3.0j, 0, 0.5 - 0.5j, 0],
            [-1.5j, -0.25 + 0.75j, -0.5, 0, 0, 1.5j, 0, -0.25 + 0.25j],
            [0.5 + 0.5j, 0, 3.0j, 0, 0.25, 0.25, 0.5 + 0.75j, -0.75j],
            [0, -0.25 - 0.25j, 0, -1.5j, 0.25, -0.25, -0.75j, -0.25],
            [0, 0, 0.5 + 0.5j, 0, 0.5 - 0.75j, 0.75j, 0.25, 0.25],
            [0, 0, 0, -0.25 - 0.25j, 0.75j, -0.25, 0.25, -0.25],
        ]
    )

    state = shadow_state_reconstruction(measurement_outcomes)
    np.testing.assert_almost_equal(state, expected_state)


def test_shadow_state_reconstruction_cal():
    bitstrings, paulistrings = ["01", "01"], ["XY", "XY"]
    measurement_outcomes = (bitstrings, paulistrings)
    fidelities = {"00": 1, "01": 1 / 3, "10": 1 / 3, "11": 1 / 9}

    expected_state_vec = operator_ptm_vector_rep(
        np.array(
            [
                [0.25, 0.75j, 0.75, 2.25j],
                [-0.75j, 0.25, -2.25j, 0.75],
                [0.75, 2.25j, 0.25, 0.75j],
                [-2.25j, 0.75, -0.75j, 0.25],
            ]
        )
    )
    state = shadow_state_reconstruction(measurement_outcomes, fidelities)
    np.testing.assert_almost_equal(state, expected_state_vec)


def test_expectation_estimation_shadow():
    measurement_outcomes = ["0101", "0110"], ["ZZXX", "ZZXX"]
    pauli = mitiq.PauliString("ZZ")
    batch_size = 1
    expected_result = -9

    result = expectation_estimation_shadow(
        measurement_outcomes, pauli, batch_size
    )
    assert np.isclose(result, expected_result)


def test_expectation_estimation_shadow_cal():
    bitstrings = ["0101", "0110"]
    paulistrings = ["YXZZ", "XXXX"]
    fidelities = {
        "0000": 1,
        "0001": 1 / 3,
        "0010": 1 / 3,
        "0100": 1 / 3,
        "1000": 1 / 3,
        "0011": 1 / 9,
        "0110": 1 / 9,
        "1100": 1 / 9,
        "0101": 1 / 9,
        "1010": 1 / 9,
        "1001": 1 / 9,
        "0111": 1 / 27,
        "1110": 1 / 27,
        "1011": 1 / 27,
        "1101": 1 / 27,
        "1111": 1 / 81,
    }

    measurement_outcomes = bitstrings, paulistrings
    pauli = mitiq.PauliString("YXZZ")
    batch_size = 1
    expected_result = 81 / 2

    result = expectation_estimation_shadow(
        measurement_outcomes, pauli, batch_size, fidelities
    )
    assert np.isclose(result, expected_result)


def test_expectation_estimation_shadow_no_indices():
    """
    Test expectation estimation for a shadow with no matching indices.
    The result should be 0 as there are no matching
    """
    pauli = mitiq.PauliString("XYZ")
    measurement_outcomes = ["101", "010", "101"], ["ZXY", "YZX", "ZZY"]
    batch_size = 1

    result = expectation_estimation_shadow(
        measurement_outcomes, pauli, batch_size
    )

    assert result == 0
