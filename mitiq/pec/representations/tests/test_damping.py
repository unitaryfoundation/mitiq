# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from cirq import AmplitudeDampingChannel, Circuit, Gate, H, LineQubit, X, Y, Z

from mitiq.interface import convert_from_mitiq
from mitiq.pec.channels import _circuit_to_choi, _operation_to_choi
from mitiq.pec.representations.damping import (
    _represent_operation_with_amplitude_damping_noise,
    amplitude_damping_kraus,
)


@pytest.mark.parametrize("noise", [0, 0.1, 0.7])
@pytest.mark.parametrize("gate", [X, Y, Z, H])
def test_single_qubit_representation_norm(gate: Gate, noise: float):
    q = LineQubit(0)
    optimal_norm = (1 + noise) / (1 - noise)
    norm = _represent_operation_with_amplitude_damping_noise(
        Circuit(gate(q)),
        noise,
    ).norm
    assert np.isclose(optimal_norm, norm)


# When _represent_operation_with_amplitude_damping_noise will
# support more circuit types we can add them below.
@pytest.mark.parametrize("circuit_type", ["cirq"])
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
    op_rep = _represent_operation_with_amplitude_damping_noise(
        ideal_circuit,
        noise,
    )
    choi_components = []
    for coeff, noisy_op in op_rep.basis_expansion:
        implementable_circ = noisy_op.circuit
        depolarizing_op = AmplitudeDampingChannel(noise).on(q)
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        implementable_circ.append(depolarizing_op)
        sequence_choi = _operation_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)

    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=1e-7)


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
