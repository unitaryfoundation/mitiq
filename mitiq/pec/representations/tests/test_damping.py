# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from cirq import AmplitudeDampingChannel, Circuit, Gate, H, LineQubit, X, Y, Z

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import UnsupportedCircuitError, convert_from_mitiq
from mitiq.pec.channels import _circuit_to_choi, _operation_to_choi
from mitiq.pec.representations.damping import (
    amplitude_damping_kraus,
    represent_operation_with_amplitude_damping_noise,
)

# Frontends which can express the reset operation used by the representation.
RESET_SUPPORTED_TYPES = ["cirq", "qiskit", "openqasm"]
# Frontends which cannot: they either fail the conversion or drop the reset.
RESET_UNSUPPORTED_TYPES = ["braket", "pennylane", "pyquil", "qibo"]


def test_all_frontends_are_covered():
    """Every supported frontend either expresses a reset or does not, so a new
    frontend has to be added to one of the two lists above.
    """
    assert set(RESET_SUPPORTED_TYPES) | set(RESET_UNSUPPORTED_TYPES) == set(
        SUPPORTED_PROGRAM_TYPES.keys()
    )


@pytest.mark.parametrize("circuit_type", RESET_SUPPORTED_TYPES)
@pytest.mark.parametrize("noise", [0, 0.1, 0.7])
@pytest.mark.parametrize("gate", [X, Y, Z, H])
def test_single_qubit_representation_norm(
    gate: Gate, noise: float, circuit_type: str
):
    q = LineQubit(0)
    optimal_norm = (1 + noise) / (1 - noise)
    circuit = convert_from_mitiq(Circuit(gate(q)), circuit_type)
    norm = represent_operation_with_amplitude_damping_noise(
        circuit,
        noise,
    ).norm
    assert np.isclose(optimal_norm, norm)


@pytest.mark.parametrize("circuit_type", RESET_SUPPORTED_TYPES)
@pytest.mark.parametrize("noise", [0, 0.1, 0.7])
@pytest.mark.parametrize("gate", [X, Y, Z, H])
def test_amplitude_damping_representation_with_choi(
    gate: Gate,
    noise: float,
    circuit_type: str,
):
    """Tests the representation by comparing exact Choi matrices."""
    q = LineQubit(0)
    ideal_circuit = convert_from_mitiq(Circuit(gate.on(q)), circuit_type)
    ideal_choi = _circuit_to_choi(Circuit(gate.on(q)))
    op_rep = represent_operation_with_amplitude_damping_noise(
        ideal_circuit,
        noise,
    )
    choi_components = []
    for coeff, noisy_op in op_rep.basis_expansion:
        # A conversion can relabel the qubits of the input circuit, while the
        # Choi helper assumes line qubits, so relabel them back.
        (native_qubit,) = noisy_op.circuit.all_qubits()
        implementable_circ = noisy_op.circuit.transform_qubits(
            {native_qubit: q}
        )
        damping_op = AmplitudeDampingChannel(noise).on(q)
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        implementable_circ.append(damping_op)
        sequence_choi = _operation_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)

    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=1e-7)


@pytest.mark.parametrize("circuit_type", RESET_SUPPORTED_TYPES)
def test_representation_keeps_the_input_circuit_type(circuit_type: str):
    """The noisy operations are returned in the type of the input circuit."""
    q = LineQubit(0)
    circuit = convert_from_mitiq(Circuit(H(q)), circuit_type)
    op_rep = represent_operation_with_amplitude_damping_noise(circuit, 0.1)
    for noisy_op in op_rep.noisy_operations:
        assert type(noisy_op.native_circuit) is type(circuit)


@pytest.mark.parametrize("circuit_type", RESET_UNSUPPORTED_TYPES)
def test_representation_without_reset_support(circuit_type: str):
    """Frontends which cannot express a reset get a clear error rather than a
    representation which silently drops it.
    """
    q = LineQubit(0)
    circuit = convert_from_mitiq(Circuit(H(q)), circuit_type)
    with pytest.raises(UnsupportedCircuitError, match="reset"):
        represent_operation_with_amplitude_damping_noise(circuit, 0.1)


def test_two_qubit_representation_error():
    qubits = LineQubit.range(2)
    circuit = Circuit(H.on_each(*qubits))
    with pytest.raises(ValueError, match="single-qubit"):
        represent_operation_with_amplitude_damping_noise(circuit, 0.1)


def test_damping_kraus():
    expected = [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]
    assert np.allclose(amplitude_damping_kraus(1, 1), expected)
    expected = [
        [[1.0, 0.0], [0.0, np.sqrt(0.5)]],
        [[0.0, np.sqrt(0.5)], [0.0, 0.0]],
    ]
    assert np.allclose(amplitude_damping_kraus(0.5, 1), expected)
    # Test normalization of kraus operators
    for num_qubits in (1, 2, 3):
        for noise_level in (0.1, 1):
            kraus_ops = amplitude_damping_kraus(noise_level, num_qubits)
            dual_channel = sum([k.conj().T @ k for k in kraus_ops])
            assert np.allclose(dual_channel, np.eye(2**num_qubits))
