# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for PEA."""

import cirq
import numpy as np
import pytest

from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.pea import combine_results, construct_circuits, execute_with_pea
from mitiq.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec import (
    OperationRepresentation,
)
from mitiq.pec.pec import LargeSampleWarning
from mitiq.typing import QPROGRAM, SUPPORTED_PROGRAM_TYPES
from mitiq.zne.inference import LinearFactory


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
        random_state=1,
    )
    assert len(scaled_circuits) == len(scale_factors)


extrapolation_method = LinearFactory.extrapolate


def test_combining_results():
    """simple arithmetic test"""
    scaled_results = [[0.1, 0.2, 0.3], [0.12, 0.24, 0.36], [0.16, 0.32, 0.48]]
    scale_factors = [1, 1.2, 1.6]
    scaled_norms = [23, 27.6, 36.8]
    scaled_signs = [[1, -1, 1], [1, -1, 1], [1, -1, 1]]
    pea_estimate = combine_results(
        scale_factors,
        scaled_results,
        scaled_norms,
        scaled_signs,
        extrapolation_method,
    )

    assert np.isclose(pea_estimate, -2.55, atol=0.01)


BASE_NOISE = 0.02


def executor(circuit: QPROGRAM, noise: float = BASE_NOISE) -> float:
    """A noisy executor function which executes the input circuit with `noise`
    depolarizing noise and returns the expectation value of the ground state
    projector. Simulation will be slow for "large circuits" (> a few qubits).
    """
    circuit, _ = convert_to_mitiq(circuit)
    return compute_density_matrix(
        circuit, noise_model_function=cirq.depolarize, noise_level=(noise,)
    )[0, 0].real


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_execute_with_pea_mitigates_noise(circuit, circuit_type):
    """Tests that execute_with_pea mitigates the error of a noisy
    expectation value.
    """
    circuit = convert_from_mitiq(circuit, circuit_type)

    true_noiseless_value = executor(circuit, noise=0.0)
    unmitigated = executor(circuit)

    scale_factors = [1, 1.2, 1.6]
    epsilon = 0.02
    noise_model = "local_depolarizing"
    mitigated = execute_with_pea(
        circuit,
        executor,
        scale_factors,
        noise_model,
        epsilon,
        extrapolation_method,
        random_state=101,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    assert error_mitigated < error_unmitigated
    assert np.isclose(mitigated, true_noiseless_value, atol=0.1)
