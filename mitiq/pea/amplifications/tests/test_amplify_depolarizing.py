# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mitiq.pea.amplify_depolarizing functions."""

import pytest
from cirq import (
    CCNOT,
    CNOT,
    Circuit,
    H,
    LineQubit,
    MeasurementGate,
    X,
    Y,
)

from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_op_with_global_depolarizing_noise,
    amplify_noisy_op_with_local_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.typing import SUPPORTED_PROGRAM_TYPES
from mitiq.utils import _equal


def single_qubit_depolarizing_overhead(noise_level: float) -> float:
    """See :cite:`Temme_2017_PRL` for more information.

    Args:
        noise_level: multiplier of noise level in :cite:`Temme_2017_PRL`

    Returns:
        Depolarizing overhead value with noise level considered.
    """
    epsilon = 3 / 4 * noise_level
    return 2 / 3 * (epsilon - 1)


def two_qubit_depolarizing_overhead(noise_level: float) -> float:
    """See :cite:`Temme_2017_PRL` for more information.

    Args:
        noise_level: multiplier of noise level in :cite:`Temme_2017_PRL`

    Returns:
        Depolarizing overhead value with noise level considered.
    """
    epsilon = 15 / 16 * noise_level
    return (epsilon - 1) / (epsilon + 7 / 8)


def test_three_qubit_depolarizing_amplification_error():
    q0, q1, q2 = LineQubit.range(3)
    with pytest.raises(ValueError):
        amplify_noisy_op_with_global_depolarizing_noise(
            Circuit(CCNOT(q0, q1, q2)),
            0.05,
        )


def test_three_qubit_local_depolarizing_amplification_error():
    q0, q1, q2 = LineQubit.range(3)
    with pytest.raises(ValueError):
        amplify_noisy_op_with_local_depolarizing_noise(
            Circuit(CCNOT(q0, q1, q2)),
            0.05,
        )


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_amplify_operations_in_circuit_global(circuit_type: str):
    """Tests all operation amplifications are created."""
    qreg = LineQubit.range(2)
    circ_mitiq = Circuit([CNOT(*qreg), H(qreg[0]), Y(qreg[1]), CNOT(*qreg)])
    circ = convert_from_mitiq(circ_mitiq, circuit_type)

    amps = amplify_noisy_ops_in_circuit_with_global_depolarizing_noise(
        ideal_circuit=circ,
        noise_level=0.1,
    )

    # For each operation in circ we should find its amplification
    for op in convert_to_mitiq(circ)[0].all_operations():
        found = False
        for amp in amps:
            if _equal(amp.ideal, Circuit(op), require_qubit_equality=True):
                found = True
        assert found

    # The number of amps. should match the number of unique operations
    assert len(amps) == 3


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_amplify_operations_in_circuit_local(circuit_type: str):
    """Tests all operation amplifications are created."""
    qreg = LineQubit.range(2)
    circ_mitiq = Circuit([CNOT(*qreg), H(qreg[0]), Y(qreg[1]), CNOT(*qreg)])
    circ = convert_from_mitiq(circ_mitiq, circuit_type)

    amps = amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
        ideal_circuit=circ,
        noise_level=0.1,
    )

    for op in convert_to_mitiq(circ)[0].all_operations():
        found = False
        for amp in amps:
            if _equal(amp.ideal, Circuit(op), require_qubit_equality=True):
                found = True
        assert found

    # The number of amps. should match the number of unique operations
    assert len(amps) == 3


@pytest.mark.parametrize(
    "amplification_function",
    [
        amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
        amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    ],
)
@pytest.mark.parametrize("circuit_type", ["cirq", "qiskit", "pyquil"])
def test_amplify_operations_in_circuit_with_measurements(
    circuit_type: str,
    amplification_function,
):
    """Tests measurements in circuit are ignored (not noise amplified)."""
    q0, q1 = LineQubit.range(2)
    circ_mitiq = Circuit(
        X(q1),
        MeasurementGate(num_qubits=1)(q0),
        X(q1),
        MeasurementGate(num_qubits=1)(q0),
    )
    circ = convert_from_mitiq(circ_mitiq, circuit_type)

    amps = amplification_function(ideal_circuit=circ, noise_level=0.1)

    for op in convert_to_mitiq(circ)[0].all_operations():
        found = False
        for amp in amps:
            if _equal(amp.ideal, Circuit(op), require_qubit_equality=True):
                found = True
        if isinstance(op.gate, MeasurementGate):
            assert not found
        else:
            assert found

    # Number of unique gates excluding measurement gates
    assert len(amps) == 1
