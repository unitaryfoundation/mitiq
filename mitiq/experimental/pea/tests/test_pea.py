# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for PEA."""

import cirq
import numpy as np
import pytest

from mitiq.experimental.pea import (
    combine_results,
    construct_circuits,
    execute_with_pea,
)
from mitiq.experimental.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.experimental.pea.scale_amplifications import scale_representations
from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.pec import (
    OperationRepresentation,
)
from mitiq.pec.pec import LargeSampleWarning, sample_circuit
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


def test_combining_results():
    """simple arithmetic test"""
    pea_estimate = combine_results(
        scale_factors=[1, 1.2, 1.6],
        scaled_results=[
            [0.1, 0.2, 0.3],
            [0.12, 0.24, 0.36],
            [0.16, 0.32, 0.48],
        ],
        scaled_norms=[23, 27.6, 36.8],
        scaled_signs=[[1, -1, 1], [1, -1, 1], [1, -1, 1]],
        extrapolation_method=LinearFactory.extrapolate,
    )
    assert np.isclose(pea_estimate, -2.55, atol=0.01)


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

    mitigated = execute_with_pea(
        circuit,
        executor,
        scale_factors=[1, 1.2, 1.6],
        noise_model="local_depolarizing",
        epsilon=0.02,
        extrapolation_method=LinearFactory.extrapolate,
        random_state=101,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    assert error_mitigated < error_unmitigated
    assert np.isclose(mitigated, true_noiseless_value, atol=0.1)


def test_pea_data_with_full_output():
    """Tests that execute_with_pea mitigates the error of a noisy
    expectation value.
    """
    precision = 0.5
    epsilon = 0.02
    pea_value, pea_data = execute_with_pea(
        twoq_circ,
        executor,
        scale_factors=[1, 1.2, 1.6],
        noise_model="local_depolarizing",
        epsilon=epsilon,
        extrapolation_method=LinearFactory.extrapolate,
        precision=precision,
        full_output=True,
    )
    # Get num samples from precision
    _, _, norm = sample_circuit(
        twoq_circ,
        amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
            twoq_circ, epsilon
        ),
        num_samples=1,
    )
    num_samples = int((norm / precision) ** 2)

    # Manually get raw expectation values
    scaled_exp_values = [
        [executor(c) for c in s_circ]
        for s_circ in pea_data["scaled_sampled_circuits"]
    ]
    assert pea_data["num_samples"] == num_samples
    assert pea_data["precision"] == precision
    assert np.isclose(pea_data["pea_value"], pea_value)
    assert np.allclose(
        pea_data["scaled_expectation_values"], scaled_exp_values
    )


# Representations interface (issue #2936).


def global_depolarizing_representations(circuit, base_noise):
    """Base (unscaled) global-depolarizing representations for a circuit."""
    return amplify_noisy_ops_in_circuit_with_global_depolarizing_noise(
        circuit, base_noise
    )


@pytest.mark.parametrize("scale_factor", [1, 3, 5])
def test_scale_representations_matches_global_rebuild(scale_factor):
    """The canonical scaler reproduces the legacy global-depolarizing rebuild
    at every scale factor, not only at scale_factor=1."""
    base = global_depolarizing_representations(twoq_circ, BASE_NOISE)
    scaled = scale_representations(base, scale_factor)
    rebuilt = amplify_noisy_ops_in_circuit_with_global_depolarizing_noise(
        twoq_circ, scale_factor * BASE_NOISE
    )
    assert scaled == rebuilt
    for rep in scaled:
        assert np.isclose(sum(rep.coeffs), 1.0)


@pytest.mark.parametrize("scale_factor", [0.5, 1, 3, 5])
def test_scale_representations_preserves_unit_sum(scale_factor):
    """Scaling keeps the quasi-probability coefficients summing to one, even
    for two-qubit local depolarizing where the legacy rebuild is not linear."""
    base = amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
        twoq_circ, BASE_NOISE
    )
    for rep in scale_representations(base, scale_factor):
        assert np.isclose(sum(rep.coeffs), 1.0)


def test_scale_representations_requires_identity_term():
    """A representation with no identity term cannot be scaled."""
    base = global_depolarizing_representations(oneq_circ, BASE_NOISE)[0]
    # Drop the identity term (first coeff/op) so the deviation is undefined.
    broken = OperationRepresentation(
        base.ideal,
        base.noisy_operations[1:],
        [c / sum(base.coeffs[1:]) for c in base.coeffs[1:]],
        base.is_qubit_dependent,
    )
    with pytest.raises(ValueError, match="without a non-zero identity term"):
        scale_representations([broken], 3)


def test_scale_representations_requires_nonzero_identity_weight():
    """A representation whose identity term has zero weight cannot scale."""
    base = global_depolarizing_representations(oneq_circ, BASE_NOISE)[0]
    coeffs = list(base.coeffs)
    # Move all weight off the identity term while keeping the unit sum.
    coeffs[1] += coeffs[0]
    coeffs[0] = 0.0
    zero_identity = OperationRepresentation(
        base.ideal,
        base.noisy_operations,
        coeffs,
        base.is_qubit_dependent,
    )
    with pytest.raises(ValueError, match="without a non-zero identity term"):
        scale_representations([zero_identity], 3)


def test_construct_circuits_raises_on_empty_representations():
    with pytest.raises(ValueError, match="'representations' is empty"):
        construct_circuits(oneq_circ, scale_factors=[1], representations=[])


@pytest.mark.parametrize("scale_factors", [[1], [1, 3, 5]])
def test_construct_circuits_representations_equivalent_to_noise_model(
    scale_factors,
):
    """Passing global-depolarizing representations yields the exact same
    sampled circuits, signs and norms as the noise_model path, at every scale
    factor (acceptance criterion of issue #2936)."""
    reps = global_depolarizing_representations(twoq_circ, BASE_NOISE)

    circuits_reps, signs_reps, norms_reps = construct_circuits(
        twoq_circ,
        scale_factors=scale_factors,
        representations=reps,
        num_samples=40,
        random_state=7,
    )
    circuits_nm, signs_nm, norms_nm = construct_circuits(
        twoq_circ,
        scale_factors=scale_factors,
        noise_model="global_depolarizing",
        epsilon=BASE_NOISE,
        num_samples=40,
        random_state=7,
    )

    assert circuits_reps == circuits_nm
    assert signs_reps == signs_nm
    assert np.allclose(norms_reps, norms_nm)


def test_construct_circuits_with_representations_shape():
    """construct_circuits accepts representations and returns one entry per
    scale factor."""
    reps = global_depolarizing_representations(oneq_circ, BASE_NOISE)
    scaled_circuits, _, _ = construct_circuits(
        oneq_circ,
        scale_factors=[1, 3, 5],
        representations=reps,
        num_samples=30,
        random_state=1,
    )
    assert len(scaled_circuits) == 3
    assert all(len(c) == 30 for c in scaled_circuits)


def test_construct_circuits_raises_if_both_provided():
    reps = global_depolarizing_representations(oneq_circ, BASE_NOISE)
    with pytest.raises(ValueError, match="not both"):
        construct_circuits(
            oneq_circ,
            scale_factors=[1],
            representations=reps,
            noise_model="global_depolarizing",
            epsilon=BASE_NOISE,
        )


def test_construct_circuits_raises_if_neither_provided():
    with pytest.raises(ValueError, match="either 'representations'"):
        construct_circuits(oneq_circ, scale_factors=[1])


def test_construct_circuits_raises_if_noise_model_without_epsilon():
    with pytest.raises(ValueError, match="'epsilon' must be given"):
        construct_circuits(
            oneq_circ,
            scale_factors=[1],
            noise_model="global_depolarizing",
        )


def test_construct_circuits_noise_model_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="legacy"):
        construct_circuits(
            oneq_circ,
            scale_factors=[1],
            noise_model="global_depolarizing",
            epsilon=BASE_NOISE,
            num_samples=10,
            random_state=1,
        )


def test_execute_with_pea_with_representations():
    """execute_with_pea accepts representations and mitigates noise as well as
    the equivalent noise_model call."""
    reps = amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
        twoq_circ, BASE_NOISE
    )
    true_value = executor(twoq_circ, noise=0.0)

    mitigated = execute_with_pea(
        twoq_circ,
        executor,
        scale_factors=[1, 1.2, 1.6],
        extrapolation_method=LinearFactory.extrapolate,
        representations=reps,
        random_state=101,
    )
    assert isinstance(mitigated, float)
    assert np.isclose(mitigated, true_value, atol=0.1)


def test_execute_with_pea_raises_if_both_provided():
    reps = global_depolarizing_representations(oneq_circ, BASE_NOISE)
    with pytest.raises(ValueError, match="not both"):
        execute_with_pea(
            oneq_circ,
            executor,
            scale_factors=[1, 1.2, 1.6],
            extrapolation_method=LinearFactory.extrapolate,
            representations=reps,
            noise_model="global_depolarizing",
            epsilon=BASE_NOISE,
        )
