# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from cirq import (
    CNOT,
    AmplitudeDampingChannel,
    Circuit,
    Gate,
    H,
    LineQubit,
    X,
    Y,
    Z,
)

from mitiq.interface import CircuitConversionError, convert_from_mitiq
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


# The Mitiq conversions for the remaining frontends (pyquil, braket,
# qibo, pennylane) cannot faithfully carry the non-unitary reset
# operation of the basis; see
# test_amplitude_damping_representation_unsupported_frontend_raises.
@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit", "openqasm"])
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
        # Frontend round trips may rename the qubit (e.g. to a
        # NamedQubit), while _operation_to_choi assumes LineQubit(0).
        (current_qubit,) = implementable_circ.all_qubits()
        implementable_circ = implementable_circ.transform_qubits(
            {current_qubit: q}
        )
        depolarizing_op = AmplitudeDampingChannel(noise).on(q)
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        implementable_circ.append(depolarizing_op)
        sequence_choi = _operation_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)

    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=1e-7)


@pytest.mark.parametrize(
    "circuit_type", ["pyquil", "braket", "qibo", "pennylane"]
)
@pytest.mark.parametrize("noise", [0, 0.1])
def test_amplitude_damping_representation_unsupported_frontend_raises(
    circuit_type: str,
    noise: float,
):
    """Frontends whose conversions cannot carry the non-unitary reset must
    raise instead of returning a physically wrong representation.

    The pyquil, braket and qibo conversions raise while converting the
    reset-bearing basis circuit (for pyquil the blocker is the
    Cirq-to-pyQuil converter, not the pyQuil language, which has a
    ``RESET`` instruction). The pennylane conversion instead succeeds
    and silently deletes the reset; since the reset coefficient
    ``eta_2 = -noise / (1 - noise)`` vanishes at ``noise = 0``, the loss
    is invisible in any zero-noise check, so the raise must be
    deterministic and independent of the noise level.
    """
    ideal_circuit = convert_from_mitiq(Circuit(X(LineQubit(0))), circuit_type)
    with pytest.raises(CircuitConversionError):
        _represent_operation_with_amplitude_damping_noise(
            ideal_circuit,
            noise,
        )


def test_amplitude_damping_representation_pennylane_reset_loss_is_loud():
    """The pennylane converter drops the reset with only a warning, which
    would otherwise return a representation that is type-valid but wrong
    for any nonzero noise. The error must name the offending frontend and
    the unsupported reset operation."""
    ideal_circuit = convert_from_mitiq(Circuit(X(LineQubit(0))), "pennylane")
    with pytest.raises(CircuitConversionError) as excinfo:
        _represent_operation_with_amplitude_damping_noise(ideal_circuit, 0.1)
    assert "pennylane" in str(excinfo.value)
    assert "reset" in str(excinfo.value)


def test_amplitude_damping_representation_openqasm_string():
    """The OpenQASM support of this representation is exactly the
    ``QasmStringType`` OpenQASM 2.0 string produced by
    ``convert_from_mitiq(circuit, "openqasm")``; the representation built
    from it has the expected norm."""
    noise = 0.1
    qasm = convert_from_mitiq(Circuit(X(LineQubit(0))), "openqasm")
    assert "OPENQASM 2.0;" in qasm

    op_rep = _represent_operation_with_amplitude_damping_noise(qasm, noise)
    assert np.isclose(op_rep.norm, (1 + noise) / (1 - noise))


def test_amplitude_damping_representation_two_qubits_raises():
    q0, q1 = LineQubit.range(2)
    with pytest.raises(ValueError, match="single-qubit"):
        _represent_operation_with_amplitude_damping_noise(
            Circuit(CNOT(q0, q1)),
            0.1,
        )


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
