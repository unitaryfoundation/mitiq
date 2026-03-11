# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quantum processing functions for classical shadows."""

import importlib
from collections.abc import Callable
from unittest.mock import patch

import cirq
import pytest
from qiskit_aer import Aer

import mitiq
from mitiq import MeasurementResult
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)
from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    sample_bitstrings as qiskit_sample_bitstrings,
)
from mitiq.shadows.quantum_processing import (
    generate_random_pauli_strings,
    get_rotated_circuits,
    random_pauli_measurement,
)


def test_tqdm_import_available():
    # Test the case where tqdm is available
    import tqdm as tqdm_orig

    assert tqdm_orig is not None
    assert mitiq.shadows.quantum_processing.tqdm is not None


def test_tqdm_import_not_available():
    with patch.dict("sys.modules", {"tqdm": None}):
        importlib.reload(
            mitiq.shadows.quantum_processing
        )  # Reload the module to trigger the import
        assert mitiq.shadows.quantum_processing.tqdm is None

    # Reload the module again to restore the original tqdm import.
    # Otherwise, the rest of the tests are affected by the patch (issue #2318)
    importlib.reload(mitiq.shadows.quantum_processing)


def test_generate_random_pauli_strings():
    """Tests that the function generates random Pauli strings."""
    num_qubits = 5
    num_strings = 10

    # Generate random pauli strings
    result = generate_random_pauli_strings(num_qubits, num_strings)

    # Check that the result is a list
    assert isinstance(result, list)

    # Check that the number of strings matches the input
    assert len(result) == num_strings

    # Check that each string has the right number of qubits
    for pauli_string in result:
        assert len(pauli_string) == num_qubits

    # Check that each string contains only the letters X, Y, and Z
    for pauli_string in result:
        assert set(pauli_string).issubset(set(["X", "Y", "Z"]))

    # Check that the function raises an exception for negative num_qubits
    # or num_strings
    with pytest.raises(ValueError):
        generate_random_pauli_strings(-1, num_strings)

    with pytest.raises(ValueError):
        generate_random_pauli_strings(num_qubits, -1)


def cirq_executor(circuit: cirq.Circuit) -> MeasurementResult:
    return cirq_sample_bitstrings(
        circuit,
        noise_level=(0,),
        shots=1,
        sampler=cirq.Simulator(),
    )


def qiskit_executor(circuit: cirq.Circuit) -> MeasurementResult:
    return qiskit_sample_bitstrings(
        to_qiskit(circuit),
        noise_model=None,
        backend=Aer.get_backend("aer_simulator"),
        shots=1,
        measure_all=False,
    )


def test_get_rotated_circuits():
    """Tests that the circuit is rotated."""

    # define circuit
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[1], qubits[2]))
    circuit.append(cirq.CNOT(qubits[2], qubits[3]))

    # define the pauli measurements to be performed on the circuit
    pauli_strings = ["XYZX"]
    # Rotate the circuit.
    rotated_circuits = get_rotated_circuits(circuit, pauli_strings)
    # Verify that the circuit was rotated.
    circuit_1 = circuit.copy()
    circuit_1.append(cirq.H(qubits[0]))
    circuit_1.append(cirq.S(qubits[1]) ** -1)
    circuit_1.append(cirq.H(qubits[1]))
    circuit_1.append(cirq.H(qubits[3]))
    circuit_1.append(cirq.measure(*qubits))

    assert rotated_circuits[0] == circuit_1

    for rc in rotated_circuits:
        assert isinstance(rc, cirq.Circuit)


# define a simple test circuit for the following tests
def simple_test_circuit(qubits):
    circuit = cirq.Circuit()
    num_qubits = len(qubits)
    circuit.append(cirq.H.on_each(*qubits))
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


@pytest.mark.parametrize("n_qubits", [1, 2, 5])
@pytest.mark.parametrize("executor", [cirq_executor, qiskit_executor])
def test_random_pauli_measurement_no_errors(n_qubits, executor):
    """Test that random_pauli_measurement runs without errors."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    random_pauli_measurement(
        circuit, n_total_measurements=10, executor=executor
    )


@pytest.mark.parametrize("n_qubits", [1, 2, 5])
@pytest.mark.parametrize("executor", [cirq_executor, qiskit_executor])
def test_random_pauli_measurement_output_dimensions(
    n_qubits: int, executor: Callable
):
    """Test that random_pauli_measurement returns the correct output
    dimensions."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    n_total_measurements = 10
    shadow_outcomes, pauli_strings = random_pauli_measurement(
        circuit, n_total_measurements, executor=executor
    )
    shadow_outcomes_shape = len(shadow_outcomes), len(shadow_outcomes[0])
    pauli_strings_shape = len(pauli_strings), len(pauli_strings[0])
    assert shadow_outcomes_shape == (n_total_measurements, n_qubits), (
        f"Shadow outcomes have incorrect shape, expected "
        f"{(n_total_measurements, n_qubits)}, got {shadow_outcomes_shape}"
    )
    assert pauli_strings_shape == (n_total_measurements, n_qubits), (
        f"Pauli strings have incorrect shape, expected "
        f"{(n_total_measurements, n_qubits)}, got {pauli_strings_shape}"
    )


@pytest.mark.parametrize("n_qubits", [1, 2, 5])
@pytest.mark.parametrize("executor", [cirq_executor, qiskit_executor])
def test_random_pauli_measurement_output_types(
    n_qubits: int, executor: Callable
):
    """Test that random_pauli_measurement returns the correct output types."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = simple_test_circuit(qubits)
    shadow_outcomes, pauli_strings = random_pauli_measurement(
        circuit, n_total_measurements=10, executor=executor
    )
    assert isinstance(shadow_outcomes[0], str)
    assert isinstance(pauli_strings[0], str)
