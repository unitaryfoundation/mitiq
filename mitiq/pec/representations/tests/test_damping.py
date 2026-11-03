# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from cirq import AmplitudeDampingChannel, Circuit, Gate, H, LineQubit, X, Y, Z

from mitiq.interface import convert_from_mitiq
from mitiq.interface.conversions import CircuitConversionError
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


# pyquil and braket are omitted: the damping basis contains `reset`, which is
# not a unitary and which neither converter has a case for
# (`CircuitConversionError`). See
# test_amplitude_damping_representation_rejects_frontends_without_reset. The
# depolarizing representation supports them because its basis is Paulis only.
@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit"])
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
        # A frontend round-trip renames the qubit (qiskit yields
        # `NamedQubit("q_0")`, not `LineQubit(0)`), and `_circuit_to_choi`
        # builds its maximally entangled state on `LineQubit`s. Map the
        # circuit back onto `LineQubit(0)` so the two do not add up to a
        # three-qubit system.
        implementable_circ = noisy_op.circuit
        (noisy_qubit,) = implementable_circ.all_qubits()
        implementable_circ = implementable_circ.transform_qubits(
            {noisy_qubit: q}
        )
        depolarizing_op = AmplitudeDampingChannel(noise).on(q)
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        implementable_circ.append(depolarizing_op)
        sequence_choi = _operation_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)

    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=1e-7)


@pytest.mark.parametrize("circuit_type", ["qiskit"])
@pytest.mark.parametrize("noise", [0.1, 0.7])
def test_amplitude_damping_representation_is_frontend_independent(
    circuit_type: str,
    noise: float,
):
    """The quasi-probabilities must not depend on the input frontend."""
    q = LineQubit(0)
    cirq_circuit = Circuit(X(q))
    converted = convert_from_mitiq(cirq_circuit, circuit_type)

    cirq_rep = _represent_operation_with_amplitude_damping_noise(
        cirq_circuit, noise
    )
    converted_rep = _represent_operation_with_amplitude_damping_noise(
        converted, noise
    )

    assert np.allclose(cirq_rep.coeffs, converted_rep.coeffs)
    assert np.isclose(cirq_rep.norm, converted_rep.norm)


@pytest.mark.parametrize("circuit_type", ["pyquil", "braket"])
def test_amplitude_damping_representation_rejects_frontends_without_reset(
    circuit_type: str,
) -> None:
    """Frontends with no `reset` cannot carry this basis, and say so.

    This pins the boundary rather than the bug: if a converter later grows a
    reset case, this test fails and the docstring's claim about that frontend
    should be revisited.
    """
    pytest.importorskip(
        {"pyquil": "pyquil", "braket": "braket"}[circuit_type],
        reason=f"{circuit_type} is not installed",
    )
    q = LineQubit(0)
    circuit = convert_from_mitiq(Circuit(X(q)), circuit_type)

    with pytest.raises(CircuitConversionError):
        _represent_operation_with_amplitude_damping_noise(circuit, 0.1)


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
