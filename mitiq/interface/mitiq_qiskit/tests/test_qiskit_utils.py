# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for qiskit executors (qiskit_utils.py)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeLimaV2

from mitiq import MeasurementResult, Observable, PauliString
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    compute_expectation_value_on_noisy_backend,
    execute,
    execute_with_noise,
    execute_with_shots,
    execute_with_shots_and_noise,
    initialized_depolarizing_noise,
    sample_bitstrings,
)

NOISE = 0.007
ONE_QUBIT_GS_PROJECTOR = np.array([[1, 0], [0, 0]])
TWO_QUBIT_GS_PROJECTOR = np.diag([1, 0, 0, 0])
SHOTS = 1_000


def test_execute():
    """Tests the Qiskit wavefunction simulation executor returns
    appropriate expectation value given an observable.
    """

    circ = QuantumCircuit(1)
    expected_value = execute(circ, obs=ONE_QUBIT_GS_PROJECTOR)
    assert expected_value == 1.0

    second_circ = QuantumCircuit(1)
    second_circ.x(0)
    expected_value = execute(second_circ, obs=ONE_QUBIT_GS_PROJECTOR)
    assert expected_value == 0.0


def test_execute_with_shots():
    """Tests the Qiskit wavefunction sampling simulation executor returns
    appropriate expectation value given an observable.
    """

    circ = QuantumCircuit(1, 1)
    expectation_value = execute_with_shots(
        circuit=circ, obs=ONE_QUBIT_GS_PROJECTOR, shots=SHOTS
    )
    assert expectation_value == 1.0

    second_circ = QuantumCircuit(1)
    second_circ.x(0)
    expectation_value = execute_with_shots(
        circuit=second_circ, obs=ONE_QUBIT_GS_PROJECTOR, shots=SHOTS
    )
    assert expectation_value == 0.0


def test_execute_with_depolarizing_noise_single_qubit():
    """Tests the noisy sampling executor across increasing levels
    of single qubit gate noise
    """

    single_qubit_circ = QuantumCircuit(1)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    single_qubit_circ.z(0)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_execute_with_depolarizing_noise_two_qubit():
    """Tests the noisy sampling executor across increasing levels of
    two qubit gate noise.
    """

    two_qubit_circ = QuantumCircuit(2)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    two_qubit_circ.cx(0, 1)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_noise(
        circuit=two_qubit_circ,
        obs=TWO_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_execute_with_shots_and_depolarizing_noise_single_qubit():
    """Tests the noisy sampling executor across increasing levels
    of single qubit gate noise.
    """

    single_qubit_circ = QuantumCircuit(1, 1)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    single_qubit_circ.z(0)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_shots_and_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
        shots=SHOTS,
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_execute_with_shots_and_depolarizing_noise_two_qubit():
    """Tests the noisy sampling executor across increasing levels of
    two qubit gate noise.
    """

    two_qubit_circ = QuantumCircuit(2, 2)
    # noise model is defined on gates so include the gate to
    # demonstrate noise
    two_qubit_circ.cx(0, 1)

    noiseless_exp_value = 1.0

    expectation_value = execute_with_shots_and_noise(
        circuit=two_qubit_circ,
        obs=TWO_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
        shots=SHOTS,
    )
    # anticipate that the expectation value will be less than
    # the noiseless simulation of the same circuit
    assert expectation_value < noiseless_exp_value


def test_circuit_is_not_mutated_by_executors():
    single_qubit_circ = QuantumCircuit(1, 1)
    single_qubit_circ.z(0)
    expected_circuit = single_qubit_circ.copy()
    execute_with_shots_and_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
        shots=SHOTS,
    )
    assert single_qubit_circ.data == expected_circuit.data
    assert single_qubit_circ == expected_circuit
    execute_with_noise(
        circuit=single_qubit_circ,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(NOISE),
    )
    assert single_qubit_circ.data == expected_circuit.data
    assert single_qubit_circ == expected_circuit


def test_sample_bitstrings():
    """Tests that the function sample_bitstrings returns a valid
    mitiq.MeasurementResult.
    """

    two_qubit_circ = QuantumCircuit(2, 1)
    two_qubit_circ.cx(0, 1)
    two_qubit_circ.measure(0, 0)

    measurement_result = sample_bitstrings(
        circuit=two_qubit_circ,
        backend=None,
        noise_model=initialized_depolarizing_noise(0),
        shots=5,
    )
    assert measurement_result.result == [[0], [0], [0], [0], [0]]
    assert measurement_result.qubit_indices == (0,)


def test_sample_bitstrings_with_measure_all():
    """Tests that the function sample_bitstrings returns a valid
    mitiq.MeasurementResult when "measure_all" is True.
    """
    two_qubit_circ = QuantumCircuit(2)
    two_qubit_circ.cx(0, 1)
    measurement_result = sample_bitstrings(
        circuit=two_qubit_circ,
        backend=None,
        noise_model=initialized_depolarizing_noise(0),
        shots=2,
        measure_all=True,
    )
    assert measurement_result.result == [[0, 0], [0, 0]]
    assert measurement_result.qubit_indices == (0, 1)
    assert isinstance(measurement_result, MeasurementResult)


def test_sample_bitstrings_with_backend():
    """Tests that the function sample_bitstrings returns a valid
    mitiq.MeasurementResult if a qiskit backend is used.
    """
    two_qubit_circ = QuantumCircuit(2)
    two_qubit_circ.cx(0, 1)
    measurement_result = sample_bitstrings(
        circuit=two_qubit_circ,
        backend=FakeLimaV2(),
        shots=5,
        measure_all=True,
    )
    assert len(measurement_result.result) == 5
    assert len(measurement_result.result[0]) == 2
    assert measurement_result.qubit_indices == (0, 1)


def test_sample_bitstrings_error_message():
    """Tests that an error is given backend and nose_model are both None."""
    two_qubit_circ = QuantumCircuit(2)
    two_qubit_circ.cx(0, 1)
    with pytest.raises(ValueError, match="Either a backend or a noise model"):
        sample_bitstrings(
            circuit=two_qubit_circ,
            shots=5,
        )


def test_compute_expectation_value_on_noisy_backend_with_noise_model():
    """Tests the evaluation of an expectation value assuming a noise model."""
    obs = Observable(PauliString("X"))
    qiskit_circuit = QuantumCircuit(1)
    qiskit_circuit.h(0)

    # First we try without noise
    noiseless_expval = compute_expectation_value_on_noisy_backend(
        qiskit_circuit,
        obs,
        noise_model=initialized_depolarizing_noise(0),
    )

    assert isinstance(noiseless_expval, complex)
    assert np.isclose(np.imag(noiseless_expval), 0.0)
    assert np.isclose(np.real(noiseless_expval), 1.0)

    # Now we try with noise
    expval = compute_expectation_value_on_noisy_backend(
        qiskit_circuit,
        obs,
        noise_model=initialized_depolarizing_noise(0.01),
    )

    assert isinstance(expval, complex)
    assert np.isclose(np.imag(expval), 0.0)
    # With noise the result is non-deterministic
    assert 0.9 < np.real(expval) < 1.0


def test_compute_expectation_value_on_noisy_backend_with_qiskit_backend():
    """Tests the evaluation of an expectation value on a noisy backed"""
    obs = Observable(PauliString("X"))
    qiskit_circuit = QuantumCircuit(1)
    qiskit_circuit.h(0)

    expval = compute_expectation_value_on_noisy_backend(
        qiskit_circuit,
        obs,
        backend=FakeLimaV2(),
    )

    assert isinstance(expval, complex)
    assert np.isclose(np.imag(expval), 0.0)
    # With noise the result is non-deterministic
    assert 0.9 < np.real(expval) < 1.0
