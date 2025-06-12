# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for PEA."""

import cirq
import numpy as np
import pytest

from mitiq.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pea.pea import construct_circuits
from mitiq.pec import (
    OperationRepresentation,
)
from mitiq.pec.pec import LargeSampleWarning


# Noisy representations of Pauli and CNOT operations for testing.
def get_pauli_and_cnot_representations(
    base_noise: float,
) -> list[OperationRepresentation]:
    qreg = cirq.LineQubit.range(2)

    # Generate all ideal single-qubit Pauli operations for both qubits
    pauli_gates = [cirq.X, cirq.Y, cirq.Z]
    ideal_operations = []

    for gate in pauli_gates:
        for qubit in qreg:
            ideal_operations.append(gate(qubit))

    # Add CNOT operation too
    ideal_operations.append(cirq.CNOT(*qreg))

    # Generate all representations
    return amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
        ideal_circuit=cirq.Circuit(ideal_operations),
        noise_level=base_noise,
    )


BASE_NOISE = 0.02
pauli_representations = get_pauli_and_cnot_representations(BASE_NOISE)
noiseless_pauli_representations = get_pauli_and_cnot_representations(0.0)

# Simple circuits for testing.
q0, q1 = cirq.LineQubit.range(2)
oneq_circ = cirq.Circuit(cirq.Z.on(q0), cirq.Z.on(q0))
twoq_circ = cirq.Circuit(cirq.Y.on(q1), cirq.CNOT.on(q0, q1), cirq.Y.on(q1))


@pytest.mark.parametrize("precision", [0.2, 0.1])
def test_precision_option_used_in_num_samples(precision):
    """Tests that the 'precision' argument is used to deduce num_samples."""
    scaled_circuits, _, _ = construct_circuits(
        oneq_circ,
        scale_factors=[1, 3, 5, 7],
        noise_model="global_depolarizing",
        epsilon=0.02,
        precision=precision,
        full_output=True,
        random_state=1,
    )
    # we expect num_samples = 1/precision^2:
    assert np.allclose(
        [precision**2 * len(c) for c in scaled_circuits],
        [1] * len(scaled_circuits),
        atol=0.2,
    )


def test_precision_ignored_when_num_samples_present():
    num_expected_circuits = 123
    scaled_circuits, _, _ = construct_circuits(
        oneq_circ,
        scale_factors=[1, 3, 5, 7],
        noise_model="global_depolarizing",
        epsilon=0.02,
        precision=0.1,
        num_samples=num_expected_circuits,
        full_output=True,
        random_state=1,
    )
    assert all([len(c) == num_expected_circuits for c in scaled_circuits])


@pytest.mark.parametrize("bad_value", (0, -1, 2))
def test_bad_precision_argument(bad_value):
    """Tests that if 'precision' is not within (0, 1] an error is raised."""
    with pytest.raises(ValueError, match="The value of 'precision' should"):
        construct_circuits(
            oneq_circ,
            scale_factors=[1, 3, 5, 7],
            noise_model="global_depolarizing",
            epsilon=0.02,
            precision=bad_value,
        )


def test_large_sample_size_warning():
    """Ensure a warning is raised when sample size is greater than 100k."""

    with pytest.warns(LargeSampleWarning):
        construct_circuits(
            oneq_circ,
            scale_factors=[1],
            noise_model="global_depolarizing",
            epsilon=0.02,
            num_samples=100_001,
        )


@pytest.mark.parametrize("scale_factors", [[1, 3, 5], [1, 3, 5, 7]])
def test_scale_factors(scale_factors):
    scaled_circuits, _, _ = construct_circuits(
        oneq_circ,
        scale_factors,
        noise_model="global_depolarizing",
        epsilon=0.02,
        num_samples=50,
        full_output=True,
        random_state=1,
    )
    assert len(scaled_circuits) == len(scale_factors)
