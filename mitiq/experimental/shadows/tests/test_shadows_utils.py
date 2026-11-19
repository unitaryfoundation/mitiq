# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import pytest

import mitiq
from mitiq.experimental.shadows.shadows_utils import (
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
    assert np.isclose(fidelity(state_vector, rho), 0.25), (
        f"Expected 0.25, got {fidelity(state_vector, rho)}"
    )


def test_fidelity_matrix_matrix_self_is_one():
    """A state has fidelity 1 with itself, mixed states included.

    Regression test: the density-matrix branch omitted the inner square
    root of the Uhlmann formula, so it returned ``tr(sigma @ rho)`` --
    the purity when the two states are equal. A maximally mixed qubit
    therefore reported 0.5 with itself, and a 3-qubit mixed state 0.237.
    Pure states were unaffected, which is why this went unnoticed.
    """
    for num_qubits in (1, 2, 3):
        dim = 2**num_qubits
        maximally_mixed = np.eye(dim) / dim
        assert np.isclose(fidelity(maximally_mixed, maximally_mixed), 1.0)

        rng = np.random.default_rng(num_qubits)
        ginibre = rng.normal(size=(dim, dim)) + 1j * rng.normal(
            size=(dim, dim)
        )
        mixed = ginibre @ ginibre.conj().T
        mixed /= np.trace(mixed).real
        assert np.isclose(fidelity(mixed, mixed), 1.0)


def test_fidelity_matrix_matrix_orthogonal_and_known_values():
    """Known density-matrix pairs give their analytic fidelities."""
    ket_0 = np.array([1.0, 0.0], dtype=complex)
    ket_1 = np.array([0.0, 1.0], dtype=complex)
    rho_0 = np.outer(ket_0, ket_0.conj())
    rho_1 = np.outer(ket_1, ket_1.conj())
    maximally_mixed = np.eye(2) / 2

    assert np.isclose(fidelity(rho_0, rho_1), 0.0)
    assert np.isclose(fidelity(rho_0, rho_0), 1.0)
    # F(|0><0|, I/2) = <0|I/2|0> = 1/2
    assert np.isclose(fidelity(rho_0, maximally_mixed), 0.5)


def test_fidelity_agrees_across_representations():
    """A pure state gives the same fidelity as a vector or as a matrix.

    The four dimension branches must use one convention. Passing the same
    pure state as a vector and as its density matrix has to agree,
    otherwise a caller's result depends on how they happened to hold the
    state.
    """
    rng = np.random.default_rng(11)
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    vector /= np.linalg.norm(vector)
    density = np.outer(vector, vector.conj())

    other = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    other = other @ other.conj().T
    other /= np.trace(other).real

    as_vector = fidelity(vector, other)
    as_matrix = fidelity(density, other)
    assert np.isclose(as_vector, as_matrix)


def test_fidelity_invalid_dimensions():
    """Anything that is not a vector or a square matrix is rejected."""
    with pytest.raises(ValueError, match="Invalid input dimensions"):
        fidelity(np.zeros((2, 2, 2)), np.eye(2) / 2)
