# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Quantum Subspace Expansion top level API."""

from unittest.mock import patch

import cirq
import numpy as np
import pytest

from mitiq import QPROGRAM, Observable, PauliString
from mitiq.interface import convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.qse import (
    execute_with_qse,
    get_projector,
    mitigate_executor,
    qse_decorator,
)
from mitiq.qse.qse_utils import _compute_overlap_matrix


def execute_with_depolarized_noise(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0],
        noise_model_function=cirq.depolarize,
        noise_level=(0.01,),
    )


def batched_execute_with_depolarized_noise(circuits) -> list[np.ndarray]:
    return [execute_with_depolarized_noise(circuit) for circuit in circuits]


def execute_no_noise(circuit: QPROGRAM) -> np.ndarray:
    return compute_density_matrix(
        convert_to_mitiq(circuit)[0], noise_level=(0,)
    )


@pytest.fixture
def prepare_setup():
    qc = prepare_logical_0_state_for_5_1_3_code()
    (
        check_operators,
        code_hamiltonian,
    ) = get_5_1_3_code_check_operators_and_code_hamiltonian()
    return qc, check_operators, code_hamiltonian


def test_execute_with_qse_no_noise(prepare_setup):
    qc, check_operators, code_hamiltonian = prepare_setup

    observable = get_observable_in_code_space(PauliString("ZZZZZ"))
    mitigated_value = execute_with_qse(
        qc, execute_no_noise, check_operators, code_hamiltonian, observable
    )
    assert np.isclose(mitigated_value.real, 1)

    observable = get_observable_in_code_space(PauliString("XXXXX"))
    mitigated_value = execute_with_qse(
        qc, execute_no_noise, check_operators, code_hamiltonian, observable
    )
    assert np.isclose(mitigated_value.real, 0.5)


def test_execute_with_qse(prepare_setup):
    qc, check_operators, code_hamiltonian = prepare_setup

    observable = get_observable_in_code_space(PauliString("ZZZZZ"))
    unmitigated_value = observable.expectation(
        qc, execute_with_depolarized_noise
    )
    mitigated_value = execute_with_qse(
        qc,
        execute_with_depolarized_noise,
        check_operators,
        code_hamiltonian,
        observable,
    )
    assert abs(mitigated_value.real - 1) < abs(unmitigated_value.real - 1)

    observable = get_observable_in_code_space(PauliString("XXXXX"))
    unmitigated_value = observable.expectation(
        qc, execute_with_depolarized_noise
    )
    mitigated_value = execute_with_qse(
        qc,
        execute_with_depolarized_noise,
        check_operators,
        code_hamiltonian,
        observable,
    )
    assert abs(mitigated_value.real - 0.5) < abs(unmitigated_value.real - 0.5)


@patch("mitiq.qse.qse.execute_with_qse")
def test_mitigate_executor_batched(mock_execute_with_qse, prepare_setup):
    qc, check_operators, code_hamiltonian = prepare_setup

    observable = get_observable_in_code_space(PauliString("XXXXX"))
    batched_mitigated_executor = mitigate_executor(
        batched_execute_with_depolarized_noise,
        check_operators,
        code_hamiltonian,
        observable,
    )

    num_circuits = 3
    circuits = [qc] * num_circuits
    batched_mitigated_executor(circuits)

    mock_execute_with_qse.assert_called_with(
        circuits[0],
        batched_execute_with_depolarized_noise,
        check_operators,
        code_hamiltonian,
        observable,
        {},
    )
    assert mock_execute_with_qse.call_count == num_circuits


@patch("mitiq.qse.qse.execute_with_qse")
def test_qse_decorator(mock_execute_with_qse, prepare_setup):
    qc, check_operators, code_hamiltonian = prepare_setup

    observable = get_observable_in_code_space(PauliString("ZZZZZ"))

    @qse_decorator(
        check_operators=check_operators,
        code_hamiltonian=code_hamiltonian,
        observable=observable,
    )
    def decorated_executor(circuit: QPROGRAM) -> np.ndarray:
        return compute_density_matrix(
            convert_to_mitiq(circuit)[0],
            noise_model_function=cirq.depolarize,
            noise_level=(0.01,),
        )

    decorated_executor(qc)
    mock_execute_with_qse.assert_called_once()


def test_get_projector(prepare_setup):
    qc, check_operators, code_hamiltonian = prepare_setup

    P = get_projector(qc, execute_no_noise, check_operators, code_hamiltonian)
    uniform_projector = Observable(*[-0.25 * c for c in check_operators])
    assert P == uniform_projector


def test_compute_overlap_matrix(prepare_setup):
    qc, check_operators, _ = prepare_setup

    S = _compute_overlap_matrix(qc, execute_no_noise, check_operators, {})
    assert np.allclose(S, np.ones(16))

    S = _compute_overlap_matrix(
        qc, execute_with_depolarized_noise, check_operators, {}
    )
    # Diagonal terms are all 1 because
    # ⟨Ψ|C_i C_i|Ψ⟩ = ⟨Ψ|Ψ⟩ = 1
    assert np.allclose(np.diag(np.diag(S)), np.eye(16))

    # Off diagonal terms are less than 1 because
    # ⟨Ψ|C_i C_j|Ψ⟩  = ⟨Ψ|C_k|Ψ⟩ < 1
    # (since |Ψ⟩ was rotated a bit from the logical subspace)
    # All off diagonal terms are the same because of the symmetry of the
    # total depolarizing noise.
    off_diag_elements = S[np.where(~np.eye(16, dtype=bool))]
    assert np.allclose(off_diag_elements, off_diag_elements[0])
    assert off_diag_elements[0] < 1


def test_compute_overlap_matrix_with_hamiltonian(prepare_setup):
    qc, check_operators, code_hamiltonian = prepare_setup

    # If we have a full set of check operators that form a group then all
    # entries of the H matrix should be the same.
    # H_jk = sum over all i's ⟨Ψ|C_i|Ψ⟩

    H = _compute_overlap_matrix(
        qc, execute_no_noise, check_operators, {}, code_hamiltonian
    )
    assert np.allclose(H, np.full(16, -16))

    H = _compute_overlap_matrix(
        qc,
        execute_with_depolarized_noise,
        check_operators,
        {},
        code_hamiltonian,
    )
    assert np.allclose(H, H[0][0])
    assert H[0][0].real > -16


def get_observable_in_code_space(observable: cirq.PauliString):
    FIVE_I = PauliString("IIIII")
    projector_onto_code_space = [
        PauliString("XZZXI"),
        PauliString("IXZZX"),
        PauliString("XIXZZ"),
        PauliString("ZXIXZ"),
    ]

    observable_in_code_space = Observable(FIVE_I)
    all_paulis = projector_onto_code_space + [observable]
    for g in all_paulis:
        observable_in_code_space *= 0.5 * Observable(FIVE_I, g)
    return observable_in_code_space


def get_5_1_3_code_check_operators_and_code_hamiltonian() -> tuple:
    """
    Returns the check operators and code Hamiltonian for the [[5,1,3]] code
    The check operators are computed from the stabilizer generators:
    (1+G1)(1+G2)(1+G3)(1+G4)  G = [XZZXI, IXZZX, XIXZZ, ZXIXZ]
    source: https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code
    """
    Ms = [
        "YIYXX",
        "ZIZYY",
        "IXZZX",
        "ZXIXZ",
        "YYZIZ",
        "XYIYX",
        "YZIZY",
        "ZZXIX",
        "XZZXI",
        "ZYYZI",
        "IYXXY",
        "IZYYZ",
        "YXXYI",
        "XXYIY",
        "XIXZZ",
        "IIIII",
    ]
    Ms_as_pauliStrings = [
        PauliString(M, coeff=1, support=range(5)) for M in Ms
    ]
    negative_Ms_as_pauliStrings = [
        PauliString(M, coeff=-1, support=range(5)) for M in Ms
    ]
    Hc = Observable(*negative_Ms_as_pauliStrings)
    return Ms_as_pauliStrings, Hc


def prepare_logical_0_state_for_5_1_3_code():
    """
    To simplify the testing logic. We hardcode the the logical 0 and logical 1
    states of the [[5,1,3]] code, copied from:
    https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code
    We then use Gram-Schmidt orthogonalization to fill up the rest of the
    matrix with orthonormal vectors.
    Following this we construct a circuit that has this matrix as its gate.
    """

    def gram_schmidt(
        orthogonal_vecs: list[np.ndarray],
    ) -> np.ndarray:
        orthonormalVecs = [
            vec / np.sqrt(np.vdot(vec, vec)) for vec in orthogonal_vecs
        ]
        dim = np.shape(orthogonal_vecs[0])[0]
        for i in range(dim - len(orthogonal_vecs)):
            new_vec = np.zeros(dim)
            new_vec[i] = 1
            projs = sum(
                [
                    np.vdot(new_vec, cached_vec) * cached_vec
                    for cached_vec in orthonormalVecs
                ]
            )
            new_vec -= projs
            orthonormalVecs.append(
                new_vec / np.sqrt(np.vdot(new_vec, new_vec))
            )
        return np.reshape(orthonormalVecs, (32, 32)).T

    logical_0_state = np.zeros(32)

    # bitstrings ["00000", "10010", "01001", "10100", "01010", "00101"]
    np.put(logical_0_state, [0, 18, 9, 20, 10, 5], 1 / 4)

    # bitstrings: [
    #  "11011", "00110", "11000", "11101", "00011",
    #  "11110", "01111", "10001", "01100", "10111"
    # ]
    np.put(logical_0_state, [27, 6, 24, 29, 3, 30, 15, 17, 12, 23], -1 / 4)

    logical_1_state = np.zeros(32)

    # bitstrings ["11111", "01101", "10110", "01011", "10101", "11010"]
    np.put(logical_1_state, [31, 13, 22, 11, 21, 26], 1 / 4)

    # bitstrings: [
    #  "00100", "11001", "00111", "00010", "11100",
    #  "00001", "10000", "01110", "10011", "01000"
    # ]
    np.put(logical_1_state, [4, 25, 7, 2, 28, 1, 16, 14, 19, 8], -1 / 4)

    # Fill up the rest of the matrix with orthonormal vectors
    matrix = gram_schmidt([logical_0_state, logical_1_state])
    circuit = cirq.Circuit()
    g = cirq.MatrixGate(matrix)
    qubits = cirq.LineQubit.range(5)
    circuit.append(g(*qubits))

    return circuit
