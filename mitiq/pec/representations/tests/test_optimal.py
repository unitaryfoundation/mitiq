# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import numpy as np
from cirq import (
    CNOT,
    CZ,
    AmplitudeDampingChannel,
    Circuit,
    DepolarizingChannel,
    H,
    I,
    LineQubit,
    ResetChannel,
    X,
    Y,
    Z,
    kraus,
    reset,
)
from pytest import mark, raises

from mitiq.interface import convert_from_mitiq
from mitiq.pec.channels import (
    _circuit_to_choi,
    _operation_to_choi,
    choi_to_super,
    kraus_to_choi,
    kraus_to_super,
)
from mitiq.pec.representations import (
    _represent_operation_with_amplitude_damping_noise,
    amplitude_damping_kraus,
    global_depolarizing_kraus,
    represent_operation_with_local_depolarizing_noise,
)
from mitiq.pec.representations.optimal import (
    find_optimal_representation,
    minimize_one_norm,
)
from mitiq.pec.types import NoisyOperation


def test_minimize_one_norm_failure_error():
    ideal_matrix = np.random.rand(2, 2)
    basis_matrices = [np.random.rand(2, 2)]
    with raises(RuntimeError, match="optimal representation failed"):
        minimize_one_norm(ideal_matrix, basis_matrices)


def test_minimize_one_norm_with_depolarized_choi():
    for noise_level in [0.01, 0.02, 0.03]:
        q = LineQubit(0)
        ideal_matrix = _operation_to_choi(H(q))
        basis_matrices = [
            _operation_to_choi(
                [H(q), gate(q), DepolarizingChannel(noise_level, 1)(q)]
            )
            for gate in [I, X, Y, Z, H]
        ]
        optimal_coeffs = minimize_one_norm(ideal_matrix, basis_matrices)
        represented_mat = sum(
            [eta * mat for eta, mat in zip(optimal_coeffs, basis_matrices)]
        )
        assert np.allclose(ideal_matrix, represented_mat)

        # Optimal analytic result by Takagi (arXiv:2006.12509)
        eps = 4.0 / 3.0 * noise_level
        expected = (1.0 + 0.5 * eps) / (1.0 - eps)
        assert np.isclose(np.linalg.norm(optimal_coeffs, 1), expected)


def test_minimize_one_norm_with_depolarized_superoperators():
    for noise_level in [0.01, 0.02, 0.03]:
        depo_kraus = global_depolarizing_kraus(noise_level, num_qubits=1)
        depo_super = kraus_to_super(depo_kraus)
        ideal_matrix = kraus_to_super(kraus(H))
        basis_matrices = [
            depo_super @ kraus_to_super(kraus(gate)) @ ideal_matrix
            for gate in [I, X, Y, Z, H]
        ]
        optimal_coeffs = minimize_one_norm(ideal_matrix, basis_matrices)
        represented_mat = sum(
            [eta * mat for eta, mat in zip(optimal_coeffs, basis_matrices)]
        )
        assert np.allclose(ideal_matrix, represented_mat)

        # Optimal analytic result by Takagi (arXiv:2006.12509)
        eps = 4.0 / 3.0 * noise_level
        expected = (1.0 + 0.5 * eps) / (1.0 - eps)
        assert np.isclose(np.linalg.norm(optimal_coeffs, 1), expected)


@mark.parametrize("noise_level", [0.01, 0.02, 0.03])
def test_minimize_one_norm_with_amp_damp_choi(noise_level):
    q = LineQubit(0)
    ideal_matrix = _operation_to_choi(H(q))
    basis_matrices = [
        _operation_to_choi(
            [H(q), gate(q), AmplitudeDampingChannel(noise_level)(q)]
        )
        for gate in [I, Z]
    ]
    # Append reset channel
    reset_kraus = kraus(ResetChannel())
    basis_matrices.append(kraus_to_choi(reset_kraus))
    optimal_coeffs = minimize_one_norm(ideal_matrix, basis_matrices, tol=1e-7)
    represented_mat = sum(
        [eta * mat for eta, mat in zip(optimal_coeffs, basis_matrices)]
    )
    assert np.allclose(ideal_matrix, represented_mat)

    # Optimal analytic result by Takagi (arXiv:2006.12509)
    expected = (1.0 + noise_level) / (1.0 - noise_level)
    assert np.isclose(np.linalg.norm(optimal_coeffs, 1), expected)


def test_minimize_one_norm_with_amp_damp_superoperators():
    for noise_level in [0.01, 0.02, 0.03]:
        damp_kraus = amplitude_damping_kraus(noise_level, num_qubits=1)
        damp_super = kraus_to_super(damp_kraus)
        ideal_matrix = kraus_to_super(kraus(H))
        basis_matrices = [
            damp_super @ kraus_to_super(kraus(gate)) @ ideal_matrix
            for gate in [I, Z]
        ]
        # Append reset channel
        reset_kraus = kraus(ResetChannel())
        basis_matrices.append(kraus_to_super(reset_kraus))
        optimal_coeffs = minimize_one_norm(
            ideal_matrix, basis_matrices, tol=1.0e-6
        )
        represented_mat = sum(
            [eta * mat for eta, mat in zip(optimal_coeffs, basis_matrices)]
        )
        assert np.allclose(ideal_matrix, represented_mat)

        # Optimal analytic result by Takagi (arXiv:2006.12509)
        expected = (1.0 + noise_level) / (1.0 - noise_level)
        assert np.isclose(np.linalg.norm(optimal_coeffs, 1), expected)


def test_minimize_one_norm_tolerance():
    depo_kraus = global_depolarizing_kraus(noise_level=0.1, num_qubits=1)
    depo_super = kraus_to_super(depo_kraus)
    ideal_matrix = kraus_to_super(kraus(H))
    basis_matrices = [
        depo_super @ kraus_to_super(kraus(gate)) @ ideal_matrix
        for gate in [I, X, Y, Z]
    ]
    previous_minimum = 0.0
    previous_error = 1.0
    for tol in [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8]:
        optimal_coeffs = minimize_one_norm(ideal_matrix, basis_matrices, tol)
        represented_mat = sum(
            [eta * mat for eta, mat in zip(optimal_coeffs, basis_matrices)]
        )
        worst_case_error = np.max(abs(ideal_matrix - represented_mat))
        minimum = np.linalg.norm(optimal_coeffs, 1)
        # Reducing "tol" should decrease the worst case error
        # and should also increase the objective function
        assert worst_case_error < previous_error
        assert minimum > previous_minimum
        previous_error = worst_case_error
        previous_minimum = minimum


@mark.parametrize("circ_type", ["cirq", "qiskit", "pyquil", "braket"])
def test_find_optimal_representation_depolarizing_two_qubit_gates(circ_type):
    """Test optimal representation agrees with a known analytic result."""
    for ideal_gate, noise_level in product([CNOT, CZ], [0.1, 0.5]):
        q = LineQubit.range(2)
        ideal_op = Circuit(ideal_gate(*q))
        implementable_circuits = [Circuit(ideal_op)]
        # Append two-qubit-gate with Paulis on one qubit
        for gate in [X, Y, Z]:
            implementable_circuits.append(Circuit([ideal_op, gate(q[0])]))
            implementable_circuits.append(Circuit([ideal_op, gate(q[1])]))
        # Append two-qubit gate with Paulis on both qubits
        for gate_a, gate_b in product([X, Y, Z], repeat=2):
            implementable_circuits.append(
                Circuit([ideal_op, gate_a(q[0]), gate_b(q[1])])
            )
        noisy_circuits = [
            circ + Circuit(DepolarizingChannel(noise_level).on_each(*q))
            for circ in implementable_circuits
        ]
        super_operators = [
            choi_to_super(_circuit_to_choi(circ)) for circ in noisy_circuits
        ]

        # Define circuits with native types
        implementable_native = [
            convert_from_mitiq(c, circ_type) for c in implementable_circuits
        ]
        ideal_op_native = convert_from_mitiq(ideal_op, circ_type)

        noisy_operations = [
            NoisyOperation(ideal, real)
            for ideal, real in zip(implementable_native, super_operators)
        ]

        # Find optimal representation
        rep = find_optimal_representation(
            ideal_op_native, noisy_operations, tol=1.0e-8
        )
        # Expected analytical result
        expected_rep = represent_operation_with_local_depolarizing_noise(
            ideal_op_native,
            noise_level,
        )
        assert np.allclose(np.sort(rep.coeffs), np.sort(expected_rep.coeffs))
        assert rep == expected_rep


@mark.parametrize("circ_type", ["cirq", "qiskit", "pyquil", "braket"])
def test_find_optimal_representation_single_qubit_depolarizing(circ_type):
    """Test optimal representation agrees with a known analytic result."""
    for ideal_gate, noise_level in product([X, Y, H], [0.1, 0.3]):
        q = LineQubit(0)

        ideal_op = Circuit(ideal_gate(q))
        implementable_circuits = [Circuit(ideal_op)]
        # Add ideal gate followed by Paulis
        for gate in [X, Y, Z]:
            implementable_circuits.append(Circuit([ideal_op, gate(q)]))

        noisy_circuits = [
            circ + Circuit(DepolarizingChannel(noise_level).on_each(q))
            for circ in implementable_circuits
        ]
        super_operators = [
            choi_to_super(_circuit_to_choi(circ)) for circ in noisy_circuits
        ]

        # Define circuits with native types
        implementable_native = [
            convert_from_mitiq(c, circ_type) for c in implementable_circuits
        ]
        ideal_op_native = convert_from_mitiq(ideal_op, circ_type)

        noisy_operations = [
            NoisyOperation(ideal, real)
            for ideal, real in zip(implementable_native, super_operators)
        ]
        # Find optimal representation
        rep = find_optimal_representation(
            ideal_op_native,
            noisy_operations,
            tol=1.0e-8,
        )
        # Expected analytical result
        expected_rep = represent_operation_with_local_depolarizing_noise(
            ideal_op_native,
            noise_level,
        )
        assert np.allclose(np.sort(rep.coeffs), np.sort(expected_rep.coeffs))
        assert rep == expected_rep


# After fixing the GitHub issue gh-702, other circuit types could be added.
@mark.parametrize("circ_type", ["cirq"])
def test_find_optimal_representation_single_qubit_amp_damping(circ_type):
    """Test optimal representation of agrees with a known analytic result."""
    for ideal_gate, noise_level in product([X, Y, H], [0.1, 0.3]):
        q = LineQubit(0)

        ideal_op = Circuit(ideal_gate(q))
        implementable_circuits = [Circuit(ideal_op)]
        # Add ideal gate followed by Paulis and reset
        for gate in [Z, reset]:
            implementable_circuits.append(Circuit([ideal_op, gate(q)]))

        noisy_circuits = [
            circ + Circuit(AmplitudeDampingChannel(noise_level).on_each(q))
            for circ in implementable_circuits
        ]

        super_operators = [
            choi_to_super(_circuit_to_choi(circ)) for circ in noisy_circuits
        ]

        # Define circuits with native types
        implementable_native = [
            convert_from_mitiq(c, circ_type) for c in implementable_circuits
        ]
        ideal_op_native = convert_from_mitiq(ideal_op, circ_type)

        noisy_operations = [
            NoisyOperation(ideal, real)
            for ideal, real in zip(implementable_native, super_operators)
        ]
        # Find optimal representation
        rep = find_optimal_representation(
            ideal_op_native,
            noisy_operations,
            tol=1.0e-7,
            initial_guess=[0, 0, 0],
        )
        # Expected analytical result
        expected_rep = _represent_operation_with_amplitude_damping_noise(
            ideal_op_native,
            noise_level,
        )
        assert np.allclose(np.sort(rep.coeffs), np.sort(expected_rep.coeffs))
        assert rep == expected_rep


def test_find_optimal_representation_no_superoperator_error():
    q = LineQubit(0)
    # Define noisy operation without superoperator matrix
    noisy_op = NoisyOperation(Circuit(X(q)))
    with raises(ValueError, match="numerical superoperator matrix"):
        find_optimal_representation(Circuit(X(q)), [noisy_op])


def test_initial_guess_in_minimize_one_norm():
    for noise_level in [0.7, 0.9]:
        depo_kraus = global_depolarizing_kraus(noise_level, num_qubits=1)
        depo_super = kraus_to_super(depo_kraus)
        ideal_matrix = kraus_to_super(kraus(H))
        basis_matrices = [
            depo_super @ kraus_to_super(kraus(gate)) @ ideal_matrix
            for gate in [I, X, Y, Z, H]
        ]
        optimal_coeffs = minimize_one_norm(
            ideal_matrix,
            basis_matrices,
            initial_guess=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        represented_mat = sum(
            [eta * mat for eta, mat in zip(optimal_coeffs, basis_matrices)]
        )
        assert np.allclose(ideal_matrix, represented_mat)

        # Test bad argument
        with raises(ValueError, match="shapes"):
            minimize_one_norm(
                ideal_matrix,
                basis_matrices,
                initial_guess=[1],
            )
