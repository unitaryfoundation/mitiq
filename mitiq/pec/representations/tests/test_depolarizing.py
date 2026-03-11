# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from cirq import (
    CCNOT,
    CNOT,
    CZ,
    ISWAP,
    SWAP,
    Circuit,
    DepolarizingChannel,
    Gate,
    H,
    LineQubit,
    MeasurementGate,
    X,
    Y,
    Z,
)

from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.pec.channels import _circuit_to_choi, _operation_to_choi
from mitiq.pec.representations import (
    represent_operation_with_global_depolarizing_noise,
    represent_operation_with_local_depolarizing_noise,
    represent_operations_in_circuit_with_global_depolarizing_noise,
    represent_operations_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec.representations.depolarizing import (
    global_depolarizing_kraus,
    local_depolarizing_kraus,
)
from mitiq.utils import _equal


def single_qubit_depolarizing_overhead(noise_level: float) -> float:
    """See :cite:`Temme_2017_PRL` for more information.

    Args:
        noise_level: multiplier of noise level in :cite:`Temme_2017_PRL`

    Returns:
        Depolarizing overhead value with noise level considered.
    """
    epsilon = 4 / 3 * noise_level
    return (1 + epsilon / 2) / (1 - epsilon)


def two_qubit_depolarizing_overhead(noise_level: float) -> float:
    """See :cite:`Temme_2017_PRL` for more information.

    Args:
        noise_level: multiplier of noise level in :cite:`Temme_2017_PRL`

    Returns:
        Depolarizing overhead value with noise level considered.
    """
    epsilon = 16 / 15 * noise_level
    return (1 + 7 * epsilon / 8) / (1 - epsilon)


@pytest.mark.parametrize("noise", [0, 0.1, 0.7])
@pytest.mark.parametrize("gate", [X, Y, Z, H])
def test_single_qubit_representation_norm(gate: Gate, noise: float):
    q = LineQubit(0)
    optimal_norm = single_qubit_depolarizing_overhead(noise)
    norm = represent_operation_with_global_depolarizing_noise(
        Circuit(gate(q)),
        noise,
    ).norm
    assert np.isclose(optimal_norm, norm)


@pytest.mark.parametrize("noise", [0, 0.1, 0.7])
@pytest.mark.parametrize("gate", (CZ, CNOT, ISWAP, SWAP))
def test_two_qubit_representation_norm(gate: Gate, noise: float):
    qreg = LineQubit.range(2)
    optimal_norm = two_qubit_depolarizing_overhead(noise)
    norm = represent_operation_with_global_depolarizing_noise(
        Circuit(gate(*qreg)),
        noise,
    ).norm
    assert np.isclose(optimal_norm, norm)


def test_three_qubit_depolarizing_representation_error():
    q0, q1, q2 = LineQubit.range(3)
    with pytest.raises(ValueError):
        represent_operation_with_global_depolarizing_noise(
            Circuit(CCNOT(q0, q1, q2)),
            0.05,
        )


@pytest.mark.parametrize("noise", [0, 0.1, 0.7])
@pytest.mark.parametrize("gate", [X, Y, Z, H, CZ, CNOT, ISWAP, SWAP])
def test_depolarizing_representation_with_choi(gate: Gate, noise: float):
    """Tests the representation by comparing exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    op_rep = represent_operation_with_global_depolarizing_noise(
        Circuit(gate.on(*qreg)),
        noise,
    )
    choi_components = []
    for coeff, noisy_op in op_rep.basis_expansion:
        implementable_circ = noisy_op.circuit
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        depolarizing_op = DepolarizingChannel(noise, len(qreg))(*qreg)
        implementable_circ.append(depolarizing_op)
        sequence_choi = _circuit_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)
    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=10**-6)


@pytest.mark.parametrize("noise", [0, 0.1, 0.7])
@pytest.mark.parametrize("gate", [X, Y, Z, H, CZ, CNOT, ISWAP, SWAP])
def test_local_depolarizing_representation_with_choi(gate: Gate, noise: float):
    """Tests the representation by comparing exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    op_rep = represent_operation_with_local_depolarizing_noise(
        Circuit(gate.on(*qreg)),
        noise,
    )
    choi_components = []
    for coeff, noisy_op in op_rep.basis_expansion:
        implementable_circ = noisy_op.circuit
        # The representation assume local noise on each qubit.
        depolarizing_op = DepolarizingChannel(noise).on_each(*qreg)
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        implementable_circ.append(depolarizing_op)
        sequence_choi = _circuit_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)
    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=10**-6)


def test_three_qubit_local_depolarizing_representation_error():
    q0, q1, q2 = LineQubit.range(3)
    with pytest.raises(ValueError):
        represent_operation_with_local_depolarizing_noise(
            Circuit(CCNOT(q0, q1, q2)),
            0.05,
        )


@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit", "pyquil"])
def test_represent_operations_in_circuit_global(circuit_type: str):
    """Tests all operation representations are created."""
    qreg = LineQubit.range(2)
    circ_mitiq = Circuit([CNOT(*qreg), H(qreg[0]), Y(qreg[1]), CNOT(*qreg)])
    circ = convert_from_mitiq(circ_mitiq, circuit_type)

    reps = represent_operations_in_circuit_with_global_depolarizing_noise(
        ideal_circuit=circ,
        noise_level=0.1,
    )

    # For each operation in circ we should find its representation
    for op in convert_to_mitiq(circ)[0].all_operations():
        found = False
        for rep in reps:
            if _equal(rep.ideal, Circuit(op), require_qubit_equality=True):
                found = True
        assert found

    # The number of reps. should match the number of unique operations
    assert len(reps) == 3


@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit", "pyquil"])
def test_represent_operations_in_circuit_local(circuit_type: str):
    """Tests all operation representations are created."""
    qreg = LineQubit.range(2)
    circ_mitiq = Circuit([CNOT(*qreg), H(qreg[0]), Y(qreg[1]), CNOT(*qreg)])
    circ = convert_from_mitiq(circ_mitiq, circuit_type)

    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        ideal_circuit=circ,
        noise_level=0.1,
    )

    for op in convert_to_mitiq(circ)[0].all_operations():
        found = False
        for rep in reps:
            if _equal(rep.ideal, Circuit(op), require_qubit_equality=True):
                found = True
        assert found

    # The number of reps. should match the number of unique operations
    assert len(reps) == 3


@pytest.mark.parametrize(
    "rep_function",
    [
        represent_operations_in_circuit_with_local_depolarizing_noise,
        represent_operations_in_circuit_with_global_depolarizing_noise,
    ],
)
@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit", "pyquil"])
def test_represent_operations_in_circuit_with_measurements(
    circuit_type: str,
    rep_function,
):
    """Tests measurements in circuit are ignored (not represented)."""
    q0, q1 = LineQubit.range(2)
    circ_mitiq = Circuit(
        X(q1),
        MeasurementGate(num_qubits=1)(q0),
        X(q1),
        MeasurementGate(num_qubits=1)(q0),
    )
    circ = convert_from_mitiq(circ_mitiq, circuit_type)

    reps = rep_function(ideal_circuit=circ, noise_level=0.1)

    for op in convert_to_mitiq(circ)[0].all_operations():
        found = False
        for rep in reps:
            if _equal(rep.ideal, Circuit(op), require_qubit_equality=True):
                found = True
        if isinstance(op.gate, MeasurementGate):
            assert not found
        else:
            assert found

    # Number of unique gates excluding measurement gates
    assert len(reps) == 1


@pytest.mark.parametrize(
    "kraus_function",
    [global_depolarizing_kraus, local_depolarizing_kraus],
)
def test_depolarizing_kraus(kraus_function):
    expected = [
        [[0.5, 0.0], [0.0, 0.5]],
        [[0.0 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.0 + 0.0j]],
        [[0.0 + 0.0j, 0.0 - 0.5j], [0.0 + 0.5j, 0.0 + 0.0j]],
        [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.5 + 0.0j]],
    ]
    assert np.allclose(kraus_function(3 / 4, 1), expected)
    # Test normalization of kraus operators
    for num_qubits in (1, 2, 3):
        for noise_level in (0.1, 1):
            kraus_ops = kraus_function(noise_level, num_qubits)
            dual_channel = sum([k.conj().T @ k for k in kraus_ops])
            assert np.allclose(dual_channel, np.eye(2**num_qubits))
