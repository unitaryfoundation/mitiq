# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests related to the functions contained in `mitiq.pec.channels`."""

import numpy as np
from cirq import (
    AmplitudeDampingChannel,
    Circuit,
    LineQubit,
    Y,
    depolarize,
    kraus,
)
from pytest import raises

from mitiq.pec.channels import (
    _circuit_to_choi,
    _max_ent_state_circuit,
    _operation_to_choi,
    choi_to_super,
    kraus_to_choi,
    kraus_to_super,
    super_to_choi,
)
from mitiq.pec.representations.damping import amplitude_damping_kraus
from mitiq.utils import matrix_to_vector, vector_to_matrix


def test_max_ent_state_circuit():
    """Tests 1-qubit and 2-qubit maximally entangled states are generated."""
    two_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    four_state = np.array(3 * [1, 0, 0, 0, 0] + [1]) / 2.0
    assert np.allclose(
        _max_ent_state_circuit(2).final_state_vector(), two_state
    )
    assert np.allclose(
        _max_ent_state_circuit(4).final_state_vector(), four_state
    )


def test_max_ent_state_circuit_error():
    """Tests an error is raised if the argument num_qubits is not valid."""
    for num_qubits in [0, 1, 3, 5, 2.0]:
        with raises(ValueError, match="The argument 'num_qubits' must"):
            _max_ent_state_circuit(num_qubits)
    # Test expected good arguments are ok
    for num_qubits in [2, 4, 6, 8]:
        assert _max_ent_state_circuit(num_qubits)


def test_operation_to_choi():
    """Tests the Choi matrix of a depolarizing channel is recovered."""
    # Define first the expected result
    base_noise = 0.01
    max_ent_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    identity_part = np.outer(max_ent_state, max_ent_state)
    mixed_part = np.eye(4) / 4.0
    epsilon = base_noise * 4.0 / 3.0
    choi = (1.0 - epsilon) * identity_part + epsilon * mixed_part
    # Choi matrix of the double application of a depolarizing channel
    choi_twice = sum(
        [
            ((1.0 - epsilon) ** 2 * identity_part),
            (2 * epsilon - epsilon**2) * mixed_part,
        ]
    )

    # Evaluate the Choi matrix of one or two depolarizing channels
    q = LineQubit(0)
    noisy_operation = depolarize(base_noise).on(q)
    noisy_sequence = [noisy_operation, noisy_operation]
    assert np.allclose(choi, _operation_to_choi(noisy_operation))
    assert np.allclose(choi_twice, _operation_to_choi(noisy_sequence))


def test_circuit_to_choi():
    """Tests _circuit_to_choi is consistent with _operation_to_choi."""
    base_noise = 0.01
    q = LineQubit(0)
    noisy_operation = depolarize(base_noise).on(q)
    assert np.allclose(
        _operation_to_choi(noisy_operation),
        _circuit_to_choi(Circuit(noisy_operation)),
    )
    noisy_sequence = [noisy_operation, noisy_operation]
    assert np.allclose(
        _operation_to_choi(noisy_sequence),
        _circuit_to_choi(Circuit(noisy_sequence)),
    )


def test_non_squared_dimension():
    with raises(ValueError, match="must be a square number"):
        vector_to_matrix(np.random.rand(7))
    with raises(ValueError, match="must be a square number"):
        choi_to_super(np.random.rand(7, 7))


def test_kraus_to_super():
    """Tests the function on random channels acting on random states.
    Channels and states are non-physical, but this is irrelevant for the test.
    """
    for num_qubits in (1, 2, 3, 4, 5):
        d = 2**num_qubits
        fake_kraus_ops = [
            np.random.rand(d, d) + 1.0j * np.random.rand(d, d)
            for _ in range(7)
        ]
        super_op = kraus_to_super(fake_kraus_ops)
        fake_state = np.random.rand(d, d) + 1.0j * np.random.rand(d, d)
        result_with_kraus = sum(
            [k @ fake_state @ k.conj().T for k in fake_kraus_ops]
        )
        result_with_super = vector_to_matrix(
            super_op @ matrix_to_vector(fake_state)
        )
        assert np.allclose(result_with_kraus, result_with_super)


def test_super_to_choi():
    for noise_level in [0, 0.3, 1]:
        super_damping = kraus_to_super(amplitude_damping_kraus(noise_level, 1))
        # Apply Pauli Y to get some complex numbers
        super_op = np.kron(kraus(Y)[0], kraus(Y)[0].conj()) @ super_damping
        choi_state = super_to_choi(super_op)
        # expected result
        q = LineQubit(0)
        choi_expected = _operation_to_choi(
            [AmplitudeDampingChannel(noise_level)(q), Y(q)]
        )
        assert np.allclose(choi_state, choi_expected)


def test_choi_to_super():
    # Note: up to normalization, choi_to_super is equal to super_to_choi
    # and therefore we just run the following consistency test.
    for dim in (2, 4, 8, 16):
        rand_mat = np.random.rand(dim**2, dim**2)
        assert np.allclose(super_to_choi(choi_to_super(rand_mat)), rand_mat)
        assert np.allclose(choi_to_super(super_to_choi(rand_mat)), rand_mat)


def test_kraus_to_choi():
    for dim in (2, 4, 8, 16):
        rand_kraus_ops = [np.random.rand(dim, dim) for _ in range(7)]
        super_op = kraus_to_super(rand_kraus_ops)
        expected_choi = super_to_choi(super_op)
        assert np.allclose(kraus_to_choi(rand_kraus_ops), expected_choi)
