# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np

import mitiq
from mitiq.shadows.shadows_utils import (
    batch_calibration_data,
    create_string,
    fidelity,
    n_measurements_opts_expectation_bound,
    n_measurements_tomography_bound,
    valid_bitstrings,
)


def test_create_string():
    str_len = 5
    loc_list = [1, 3]
    assert create_string(str_len, loc_list) == "01010"


def test_valid_bitstrings():
    num_qubits = 5
    bitstrings_on_5_qubits = valid_bitstrings(num_qubits)
    assert len(bitstrings_on_5_qubits) == 2**num_qubits
    assert all(b == "0" or b == "1" for b in bitstrings_on_5_qubits.pop())

    num_qubits = 4
    max_hamming_weight = 2
    bitstrings_on_3_qubits_hamming_2 = valid_bitstrings(
        num_qubits, max_hamming_weight
    )
    assert len(bitstrings_on_3_qubits_hamming_2) == sum(
        math.comb(num_qubits, i) for i in range(max_hamming_weight + 1)
    )  # sum_{i == 0}^{max_hamming_weight} (num_qubits choose i)


def test_batch_calibration_data():
    data = (["010", "110", "000", "001"], ["XXY", "ZYY", "ZZZ", "XYZ"])
    num_batches = 2
    for bits, paulis in batch_calibration_data(data, num_batches):
        assert len(bits) == len(paulis) == num_batches


def test_n_measurements_tomography_bound():
    assert n_measurements_tomography_bound(0.5, 2) == 2176
    assert n_measurements_tomography_bound(1.0, 1) == 136
    assert n_measurements_tomography_bound(0.1, 3) == 217599


def test_n_measurements_opts_expectation_bound():
    observables = [
        mitiq.PauliString("X"),
        mitiq.PauliString("Y"),
        mitiq.PauliString("Z"),
    ]
    N, K = n_measurements_opts_expectation_bound(0.5, observables, 0.1)
    assert isinstance(N, int)
    assert isinstance(K, int)


def test_fidelity():
    state_vector = np.array([0.5, 0.5, 0.5, 0.5])
    rho = np.eye(4) / 4
    assert np.isclose(
        fidelity(state_vector, rho), 0.25
    ), f"Expected 0.25, got {fidelity(state_vector, rho)}"
