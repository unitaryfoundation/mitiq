# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Test classical shadow estimation process."""

from numbers import Number

import cirq

import mitiq
from mitiq import MeasurementResult
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)
from mitiq.shadows.shadows import (
    classical_post_processing,
    pauli_twirling_calibrate,
    shadow_quantum_processing,
)

# define a fully entangled state
num_qubits: int = 2
qubits = cirq.LineQubit.range(num_qubits)
circuit = cirq.Circuit([cirq.H(q) for q in qubits])
circuit.append(cirq.CNOT(qubits[0], qubits[1]))
observables = [mitiq.PauliString("X", support=(i,)) for i in range(num_qubits)]


def executor(
    circuit: cirq.Circuit,
) -> MeasurementResult:
    return cirq_sample_bitstrings(
        circuit,
        noise_level=(0,),
        shots=1,
        sampler=cirq.Simulator(),
    )


def test_pauli_twirling_calibrate():
    # Call the function with valid inputs
    result = pauli_twirling_calibrate(
        qubits=qubits, executor=executor, num_total_measurements_calibration=2
    )

    # Check that the dictionary contains the correct number of entries
    assert len(result) <= 2**num_qubits

    for value in result.values():
        assert isinstance(value, Number)

    # Call shadow_quantum_processing to get shadow_outcomes
    shadow_outcomes = (["11", "00"], ["ZZ", "XX"])

    # Call the function with valid inputs
    result = pauli_twirling_calibrate(
        zero_state_shadow_outcomes=shadow_outcomes,
        num_total_measurements_calibration=2,
    )

    # Check that the dictionary contains the correct number of entries
    assert len(result) <= 2**num_qubits

    for value in result.values():
        assert isinstance(value, Number)


def test_shadow_quantum_processing():
    # Call the function with valid inputs
    result = shadow_quantum_processing(
        circuit, executor, num_total_measurements_shadow=10
    )

    # Check that the result is a tuple
    assert isinstance(result, tuple), f"Expected a tuple, got {type(result)}"

    # Check that the tuple contains two lists
    assert (
        len(result) == 2
    ), f"Expected two lists in the tuple, got {len(result)}"
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


def test_classical_post_processing():
    # Call shadow_quantum_processing to get shadow_outcomes
    shadow_outcomes = (["11", "00"], ["ZZ", "XX"])

    # Call pauli_twirling_calibrate to get calibration_results
    calibration_results = {"00": 1, "01": 1 / 3, "10": 1 / 3, "11": 1 / 9}

    # Call the function with valid inputs and state_reconstruction=True
    result = classical_post_processing(
        shadow_outcomes, state_reconstruction=True
    )

    # Check that the result is a dictionary
    assert isinstance(
        result, dict
    ), f"Expected a dictionary, got {type(result)}"

    # Check that the dictionary contains the expected keys
    assert "reconstructed_state" in result

    # Call the function with valid inputs and observables provided
    result = classical_post_processing(
        shadow_outcomes, observables=observables
    )
    result_cal = classical_post_processing(
        shadow_outcomes,
        calibration_results=calibration_results,
        observables=observables,
        k_shadows=1,
    )
    # Check that the result is a dictionary
    assert isinstance(result, dict)
    assert result_cal == result

    # Check that the dictionary contains the expected keys
    for obs in observables:
        assert str(obs) in result
        assert isinstance(result[str(obs)], float)
