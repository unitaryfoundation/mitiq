# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for PEC."""

from functools import partial
from unittest.mock import patch

import cirq
import numpy as np
import pyquil
import pytest
import qiskit

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES, Observable, PauliString
from mitiq.interface import convert_from_mitiq, convert_to_mitiq, mitiq_cirq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.pec import (
    NoisyOperation,
    OperationRepresentation,
    combine_results,
    construct_circuits,
    execute_with_pec,
    mitigate_executor,
    pec_decorator,
)
from mitiq.pec.pec import LargeSampleWarning
from mitiq.pec.representations import (
    represent_operations_in_circuit_with_local_depolarizing_noise,
)


# Noisy representations of Pauli and CNOT operations for testing.
def get_pauli_and_cnot_representations(
    base_noise: float,
    qubits: list[cirq.Qid] | None = None,
) -> list[OperationRepresentation]:
    if qubits is None:
        qreg = cirq.LineQubit.range(2)
    else:
        qreg = qubits

    # Generate all ideal single-qubit Pauli operations for both qubits
    pauli_gates = [cirq.X, cirq.Y, cirq.Z]
    ideal_operations = []

    for gate in pauli_gates:
        for qubit in qreg:
            ideal_operations.append(gate(qubit))

    # Add CNOT operation too
    ideal_operations.append(cirq.CNOT(*qreg))

    # Generate all representations
    return represent_operations_in_circuit_with_local_depolarizing_noise(
        ideal_circuit=cirq.Circuit(ideal_operations),
        noise_level=base_noise,
    )


BASE_NOISE = 0.02
pauli_representations = get_pauli_and_cnot_representations(BASE_NOISE)
noiseless_pauli_representations = get_pauli_and_cnot_representations(0.0)


def serial_executor(circuit: QPROGRAM, noise: float = BASE_NOISE) -> float:
    """A noisy executor function which executes the input circuit with `noise`
    depolarizing noise and returns the expectation value of the ground state
    projector. Simulation will be slow for "large circuits" (> a few qubits).
    """
    circuit, _ = convert_to_mitiq(circuit)
    return compute_density_matrix(
        circuit, noise_model_function=cirq.depolarize, noise_level=(noise,)
    )[0, 0].real


def batched_executor(circuits) -> list[float]:
    return [serial_executor(circuit) for circuit in circuits]


def noiseless_serial_executor(circuit: QPROGRAM) -> float:
    return serial_executor(circuit, noise=0.0)


def fake_executor(
    circuit: cirq.Circuit, random_state: np.random.RandomState
) -> float:
    """A fake executor which just samples from a normal distribution."""
    return random_state.randn()


# Simple circuits for testing.
q0, q1 = cirq.LineQubit.range(2)
oneq_circ = cirq.Circuit(cirq.Z.on(q0), cirq.Z.on(q0))
twoq_circ = cirq.Circuit(cirq.Y.on(q1), cirq.CNOT.on(q0, q1), cirq.Y.on(q1))


def test_execute_with_pec_cirq_trivial_decomposition():
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    rep = OperationRepresentation(
        circuit,
        [NoisyOperation(circuit)],
        [1.0],
    )

    unmitigated = serial_executor(circuit)
    mitigated = execute_with_pec(
        circuit,
        serial_executor,
        representations=[rep],
        num_samples=10,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


def test_execute_with_pec_pyquil_trivial_decomposition():
    circuit = pyquil.Program(pyquil.gates.H(0))
    rep = OperationRepresentation(
        circuit,
        [NoisyOperation(circuit)],
        [1.0],
    )
    unmitigated = serial_executor(circuit)

    mitigated = execute_with_pec(
        circuit,
        serial_executor,
        representations=[rep],
        num_samples=10,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


def test_execute_with_pec_qiskit_trivial_decomposition():
    qreg = qiskit.QuantumRegister(1)
    circuit = qiskit.QuantumCircuit(qreg)
    _ = circuit.x(qreg)
    rep = OperationRepresentation(
        circuit,
        [NoisyOperation(circuit)],
        [1.0],
    )
    unmitigated = serial_executor(circuit)

    mitigated = execute_with_pec(
        circuit,
        serial_executor,
        representations=[rep],
        num_samples=10,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
def test_execute_with_pec_cirq_noiseless_decomposition(circuit):
    unmitigated = noiseless_serial_executor(circuit)

    mitigated = execute_with_pec(
        circuit,
        noiseless_serial_executor,
        representations=noiseless_pauli_representations,
        num_samples=10,
        random_state=1,
    )

    assert np.isclose(unmitigated, mitigated)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_pyquil_noiseless_decomposition_multiqubit(nqubits: int):
    circuit = pyquil.Program(pyquil.gates.H(q) for q in range(nqubits))

    # Decompose H(q) for each qubit q into Paulis.
    representations = []
    for q in range(nqubits):
        representation = OperationRepresentation(
            pyquil.Program(pyquil.gates.H(q)),
            [
                NoisyOperation(pyquil.Program(pyquil.gates.X(q))),
                NoisyOperation(pyquil.Program(pyquil.gates.Z(q))),
            ],
            coeffs=[0.5, 0.5],
        )
        representations.append(representation)

    exact = noiseless_serial_executor(circuit)
    pec_value = execute_with_pec(
        circuit,
        noiseless_serial_executor,
        representations=representations,
        num_samples=500,
        random_state=1,
        force_run_all=False,
    )
    assert np.isclose(pec_value, exact, atol=0.1)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_qiskit_noiseless_decomposition_multiqubit(nqubits: int):
    qreg = [qiskit.QuantumRegister(1) for _ in range(nqubits)]
    circuit = qiskit.QuantumCircuit(*qreg)
    for q in qreg:
        circuit.h(q)

    # Decompose H(q) for each qubit q into Paulis.
    representations = []
    for q in qreg:
        opcircuit = qiskit.QuantumCircuit(q)
        opcircuit.h(q)

        xcircuit = qiskit.QuantumCircuit(q)
        xcircuit.x(q)

        zcircuit = qiskit.QuantumCircuit(q)
        zcircuit.z(q)

        representation = OperationRepresentation(
            opcircuit,
            [
                NoisyOperation(circuit=xcircuit),
                NoisyOperation(circuit=zcircuit),
            ],
            [0.5, 0.5],
        )
        representations.append(representation)

    exact = noiseless_serial_executor(circuit)
    pec_value = execute_with_pec(
        circuit,
        noiseless_serial_executor,
        representations=representations,
        num_samples=500,
        random_state=1,
        force_run_all=False,
    )
    assert np.isclose(pec_value, exact, atol=0.1)


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
@pytest.mark.parametrize("executor", [serial_executor, batched_executor])
@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_execute_with_pec_mitigates_noise(circuit, executor, circuit_type):
    """Tests that execute_with_pec mitigates the error of a noisy
    expectation value.
    """
    circuit = convert_from_mitiq(circuit, circuit_type)

    true_noiseless_value = 1.0
    unmitigated = serial_executor(circuit)

    if circuit_type in ["qiskit", "pennylane", "qibo", "cudaq"]:
        # Note this is an important subtlety necessary because of conversions.
        reps = get_pauli_and_cnot_representations(
            base_noise=BASE_NOISE,
            qubits=[cirq.NamedQubit(name) for name in ("q_0", "q_1")],
        )
        circuit, _ = convert_to_mitiq(circuit)
    else:
        reps = pauli_representations

    mitigated = execute_with_pec(
        circuit,
        executor,
        representations=reps,
        num_samples=100,
        force_run_all=False,
        random_state=101,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    assert error_mitigated < error_unmitigated
    assert np.isclose(mitigated, true_noiseless_value, atol=0.1)


def test_execute_with_pec_with_observable():
    circuit = twoq_circ
    obs = Observable(PauliString("ZZ"))
    executor = partial(
        mitiq_cirq.compute_density_matrix,
        noise_model_function=cirq.depolarize,
        noise_level=(BASE_NOISE,),
    )
    true_value = 1.0

    noisy_value = obs.expectation(circuit, mitiq_cirq.compute_density_matrix)
    pec_value = execute_with_pec(
        circuit,
        executor,
        observable=obs,
        representations=pauli_representations,
        num_samples=100,
        force_run_all=False,
        random_state=101,
    )
    assert abs(pec_value - true_value) < abs(noisy_value - true_value)
    assert np.isclose(pec_value, true_value, atol=0.1)


def test_execute_with_pec_partial_representations():
    # Only use the CNOT representation.
    reps = [pauli_representations[-1]]

    pec_value = execute_with_pec(
        twoq_circ,
        executor=partial(
            mitiq_cirq.compute_density_matrix,
            noise_model_function=cirq.depolarize,
            noise_level=(BASE_NOISE,),
        ),
        observable=Observable(PauliString("ZZ")),
        representations=reps,
        num_samples=100,
        force_run_all=False,
        random_state=101,
    )
    assert isinstance(pec_value, float)


@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
@pytest.mark.parametrize("seed", (2, 3))
def test_execute_with_pec_with_different_samples(circuit, seed: int):
    """Tests that, on average, the error decreases as the number of samples is
    increased.
    """
    small_sample_number = 10
    large_sample_number = 100

    errors = []
    for num_samples in (small_sample_number, large_sample_number):
        mitigated = execute_with_pec(
            circuit,
            serial_executor,
            representations=pauli_representations,
            num_samples=num_samples,
            force_run_all=True,
            random_state=seed,
        )
        errors.append(abs(mitigated - 1.0))

    assert np.average(errors[1]) < np.average(errors[0])


@pytest.mark.parametrize("num_samples", [100, 500])
def test_execute_with_pec_error_scaling(num_samples: int):
    """Tests that the error associated to the PEC value scales as
    1/sqrt(num_samples).
    """
    _, pec_data = execute_with_pec(
        oneq_circ,
        partial(fake_executor, random_state=np.random.RandomState(0)),
        representations=pauli_representations,
        num_samples=num_samples,
        force_run_all=True,
        full_output=True,
    )
    # The error should scale as 1/sqrt(num_samples)
    normalized_error = pec_data["pec_error"] * np.sqrt(num_samples)
    assert np.isclose(normalized_error, 1.0, atol=0.1)


@pytest.mark.parametrize("precision", [0.2, 0.1])
def test_precision_option_used_in_num_samples(precision: float):
    """Tests that the 'precision' argument is used to deduce num_samples."""
    circuits, _, _ = construct_circuits(
        oneq_circ,
        representations=pauli_representations,
        precision=precision,
        full_output=True,
        random_state=1,
    )
    num_circuits = len(circuits)
    # we expect num_samples = 1/precision^2:
    assert np.isclose(precision**2 * num_circuits, 1, atol=0.2)


def test_precision_ignored_when_num_samples_present():
    """Check precision is ignored when num_samples is given."""
    num_expected_circuits = 123
    circuits, _, _ = construct_circuits(
        oneq_circ,
        representations=pauli_representations,
        precision=0.1,
        num_samples=num_expected_circuits,
        full_output=True,
        random_state=1,
    )
    num_circuits = len(circuits)
    assert num_circuits == num_expected_circuits


@pytest.mark.parametrize("bad_value", (0, -1, 2))
def test_bad_precision_argument(bad_value: int):
    """Tests that if 'precision' is not within (0, 1] an error is raised."""
    with pytest.raises(ValueError, match="The value of 'precision' should"):
        construct_circuits(
            oneq_circ,
            representations=pauli_representations,
            precision=bad_value,
        )


@patch("mitiq.pec.pec.sample_circuit")
def test_large_sample_size_warning(mock_sample_circuit):
    """Ensure a warning is raised when sample size is greater than 100k."""

    mock_sample_circuit.return_value = ([], [], 0.911)

    with pytest.warns(LargeSampleWarning):
        construct_circuits(
            oneq_circ,
            representations=[],
            num_samples=100_001,
        )

    assert mock_sample_circuit.call_count == 2


def test_pec_data_with_full_output():
    """Tests that execute_with_pec mitigates the error of a noisy
    expectation value.
    """
    precision = 0.5
    pec_value, pec_data = execute_with_pec(
        twoq_circ,
        serial_executor,
        precision=precision,
        representations=pauli_representations,
        full_output=True,
        random_state=102,
    )
    # Get num samples from precision
    norm = 1.0
    for op in twoq_circ.all_operations():
        for rep in pauli_representations:
            if rep.ideal == cirq.Circuit(op):
                norm *= rep.norm
    num_samples = int((norm / precision) ** 2)

    # Manually get raw expectation values
    exp_values = [serial_executor(c) for c in pec_data["sampled_circuits"]]

    assert pec_data["num_samples"] == num_samples
    assert pec_data["precision"] == precision
    assert np.isclose(pec_data["pec_value"], pec_value)
    assert np.isclose(
        pec_data["pec_error"],
        np.std(pec_data["unbiased_estimators"]) / np.sqrt(num_samples),
    )
    assert np.isclose(np.average(pec_data["unbiased_estimators"]), pec_value)
    assert np.allclose(pec_data["measured_expectation_values"], exp_values)


def decorated_serial_executor(circuit: QPROGRAM) -> float:
    """Returns a decorated serial executor for use with other tests. The serial
    executor is decorated with the same representations as those that are used
    in the tests for trivial decomposition.
    """
    rep = OperationRepresentation(
        circuit,
        [NoisyOperation(circuit)],
        [1.0],
    )

    @pec_decorator(representations=[rep], precision=0.08, force_run_all=False)
    def decorated_executor(qp):
        return serial_executor(qp)

    return decorated_executor(circuit)


def test_mitigate_executor_qiskit():
    """Performs the same test as
    test_execute_with_pec_qiskit_trivial_decomposition(), but using
    mitigate_executor() instead of execute_with_pec().
    """
    qreg = qiskit.QuantumRegister(1)
    circuit = qiskit.QuantumCircuit(qreg)
    _ = circuit.x(qreg)
    rep = OperationRepresentation(
        circuit,
        [NoisyOperation(circuit)],
        [1.0],
    )
    unmitigated = serial_executor(circuit)

    mitigated_executor = mitigate_executor(
        serial_executor,
        representations=[rep],
        num_samples=10,
        random_state=1,
    )
    mitigated = mitigated_executor(circuit)

    assert np.isclose(unmitigated, mitigated)

    batched_unmitigated = batched_executor([circuit] * 3)

    batched_mitigated_executor = mitigate_executor(
        batched_executor,
        representations=[rep],
        num_samples=10,
        random_state=1,
    )
    batched_mitigated = batched_mitigated_executor([circuit] * 3)

    assert [
        np.isclose(batched_unmitigated_value, batched_mitigated_value)
        for batched_unmitigated_value, batched_mitigated_value in zip(
            batched_unmitigated, batched_mitigated
        )
    ]


def test_pec_decorator_qiskit():
    """Performs the same test as test_mitigate_executor_qiskit(), but using
    pec_decorator() instead of mitigate_executor().
    """
    qreg = qiskit.QuantumRegister(1)
    circuit = qiskit.QuantumCircuit(qreg)
    _ = circuit.x(qreg)

    unmitigated = serial_executor(circuit)

    mitigated = decorated_serial_executor(circuit)

    assert np.isclose(unmitigated, mitigated)


def test_mitigate_executor_cirq():
    """Performs the same test as
    test_execute_with_pec_cirq_trivial_decomposition(), but using
    mitigate_executor() instead of execute_with_pec().
    """
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    rep = OperationRepresentation(
        circuit,
        [NoisyOperation(circuit)],
        [1.0],
    )
    unmitigated = serial_executor(circuit)

    mitigated_executor = mitigate_executor(
        serial_executor,
        representations=[rep],
        num_samples=10,
        random_state=1,
    )
    mitigated = mitigated_executor(circuit)

    assert np.isclose(unmitigated, mitigated)


def test_pec_decorator_cirq():
    """Performs the same test as test_mitigate_executor_cirq(), but using
    pec_decorator() instead of mitigate_executor().
    """
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))

    unmitigated = serial_executor(circuit)

    mitigated = decorated_serial_executor(circuit)

    assert np.isclose(unmitigated, mitigated)


def test_mitigate_executor_pyquil():
    """Performs the same test as
    test_execute_with_pec_pyquil_trivial_decomposition(), but using
    mitigate_executor() instead of execute_with_pec().
    """
    circuit = pyquil.Program(pyquil.gates.H(0))
    rep = OperationRepresentation(
        circuit,
        [NoisyOperation(circuit)],
        [1.0],
    )
    unmitigated = serial_executor(circuit)

    mitigated_executor = mitigate_executor(
        serial_executor,
        representations=[rep],
        num_samples=10,
        random_state=1,
    )
    mitigated = mitigated_executor(circuit)

    assert np.isclose(unmitigated, mitigated)


def test_pec_decorator_pyquil():
    """Performs the same test as test_mitigate_executor_pyquil(), but using
    pec_decorator() instead of mitigate_executor().
    """
    circuit = pyquil.Program(pyquil.gates.H(0))

    unmitigated = serial_executor(circuit)

    mitigated = decorated_serial_executor(circuit)

    assert np.isclose(unmitigated, mitigated)


def test_doc_is_preserved():
    """Tests that the doc of the original executor is preserved."""

    representations = get_pauli_and_cnot_representations(0)

    def first_executor(circuit):
        """Doc of the original executor."""
        return 0

    mit_executor = mitigate_executor(
        first_executor, representations=representations
    )
    assert mit_executor.__doc__ == first_executor.__doc__

    @pec_decorator(representations=representations)
    def second_executor(circuit):
        """Doc of the original executor."""
        return 0

    assert second_executor.__doc__ == first_executor.__doc__


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_executed_circuits_have_the_expected_type(circuit_type: str):
    circuit = convert_from_mitiq(oneq_circ, circuit_type)
    circuit_type = type(circuit)

    # Fake executor just for testing types
    def type_detecting_executor(circuit: QPROGRAM):
        assert type(circuit) is circuit_type
        return 0.0

    mitigated = execute_with_pec(
        circuit,
        executor=type_detecting_executor,
        representations=pauli_representations,
        num_samples=1,
    )
    assert np.isclose(mitigated, 0.0)


def test_combining_results():
    """simple arithmetic test"""
    results = [0.1, 0.2, 0.3]
    norm = 23
    signs = [1, -1, 1]
    pec_estimate = combine_results(results, norm, signs)
    assert np.isclose(pec_estimate, 1.53, atol=0.01)
