# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for circuit conversions."""

import cirq
import numpy as np
import pennylane as qml
import pytest
import qibo
import qiskit
from braket.circuits import Circuit as BKCircuit
from braket.circuits import Instruction
from braket.circuits import gates as braket_gates
from pyquil import Program, gates

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import (
    UnsupportedCircuitError,
    accept_any_qprogram_as_input,
    accept_qprogram_and_validate,
    atomic_one_to_many_converter,
    convert_from_mitiq,
    convert_to_mitiq,
    register_mitiq_converters,
)
from mitiq.interface.mitiq_qiskit import from_qasm, to_qasm
from mitiq.utils import _equal

QASMType = str

# Cirq Bell circuit.
cirq_qreg = cirq.LineQubit.range(2)
cirq_circuit = cirq.Circuit(
    cirq.ops.H.on(cirq_qreg[0]), cirq.ops.CNOT.on(*cirq_qreg)
)

# Qiskit Bell circuit.
qiskit_qreg = qiskit.QuantumRegister(2)
qiskit_circuit = qiskit.QuantumCircuit(qiskit_qreg)
qiskit_circuit.h(qiskit_qreg[0])
qiskit_circuit.cx(*qiskit_qreg)
qasm_str = qiskit.qasm2.dumps(qiskit_circuit)


# pyQuil Bell circuit.
pyquil_circuit = Program(gates.H(0), gates.CNOT(0, 1))

# Braket Bell circuit.
braket_circuit = BKCircuit(
    [
        Instruction(braket_gates.H(), 0),
        Instruction(braket_gates.CNot(), [0, 1]),
    ]
)

circuit_types = {
    "cirq": cirq.Circuit,
    "qiskit": qiskit.QuantumCircuit,
    "pyquil": Program,
    "braket": BKCircuit,
    "pennylane": qml.tape.QuantumTape,
    "qibo": qibo.models.circuit.Circuit,
}


@accept_qprogram_and_validate
def scaling_function(circ: cirq.Circuit, *args, **kwargs) -> cirq.Circuit:
    return circ


def one_to_many_circuit_modifier(
    circ: cirq.Circuit, *args, **kwargs
) -> list[cirq.Circuit]:
    return [circ, circ[0:1] + circ]


# Apply one-to-many decorator
one_to_many_circuit_modifier = accept_qprogram_and_validate(
    one_to_many_circuit_modifier,
    one_to_many=True,
)


@accept_any_qprogram_as_input
def get_wavefunction(circ: cirq.Circuit) -> np.ndarray:
    return circ.final_state_vector()


@atomic_one_to_many_converter
def returns_several_circuits(circ: cirq.Circuit, *args, **kwargs):
    return [circ] * 5


@pytest.mark.parametrize(
    "circuit", (qiskit_circuit, pyquil_circuit, braket_circuit)
)
def test_to_mitiq(circuit):
    converted_circuit, input_type = convert_to_mitiq(circuit)
    assert _equal(converted_circuit, cirq_circuit)
    assert input_type in circuit.__module__


def test_register_from_to_mitiq():
    class CircuitStr(str):
        __module__ = "qasm"

    qasm_circuit = CircuitStr(qasm_str)

    register_mitiq_converters(
        qasm_circuit.__module__,
        convert_to_function=to_qasm,
        convert_from_function=from_qasm,
    )
    converted_circuit = convert_from_mitiq(cirq_circuit, "qasm")
    converted_qasm = CircuitStr(converted_circuit)
    circuit, input_type = convert_to_mitiq(converted_qasm)
    assert _equal(circuit, cirq_circuit)
    assert input_type == qasm_circuit.__module__


@pytest.mark.parametrize("item", ("circuit", 1, None))
def test_to_mitiq_bad_types(item):
    with pytest.raises(
        UnsupportedCircuitError,
        match="Could not determine the package of the input circuit.",
    ):
        convert_to_mitiq(item)


@pytest.mark.parametrize("to_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_from_mitiq(to_type):
    converted_circuit = convert_from_mitiq(cirq_circuit, to_type)
    circuit, input_type = convert_to_mitiq(converted_circuit)
    assert _equal(circuit, cirq_circuit)
    assert input_type == to_type


def test_unsupported_circuit_error():
    class CircuitStr(str):
        __module__ = "qasm"

    mock_circuit = CircuitStr("mock")

    with pytest.raises(
        UnsupportedCircuitError,
        match="Conversion to circuit type unsupported_circuit_type",
    ):
        convert_from_mitiq(mock_circuit, "unsupported_circuit_type")


@pytest.mark.parametrize(
    "circuit_and_expected",
    [
        (cirq.Circuit(cirq.X.on(cirq.LineQubit(0))), np.array([0, 1])),
        (cirq_circuit, np.array([1, 0, 0, 1]) / np.sqrt(2)),
    ],
)
@pytest.mark.parametrize("to_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_accept_any_qprogram_as_input(circuit_and_expected, to_type):
    circuit, expected = circuit_and_expected
    wavefunction = get_wavefunction(convert_from_mitiq(circuit, to_type))
    assert np.allclose(wavefunction, expected)


@pytest.mark.parametrize(
    "circuit_and_type",
    (
        (qiskit_circuit, "qiskit"),
        (pyquil_circuit, "pyquil"),
        (braket_circuit, "braket"),
    ),
)
def test_converter(circuit_and_type):
    circuit, input_type = circuit_and_type

    # Return the input type
    scaled = scaling_function(circuit)
    assert isinstance(scaled, circuit_types[input_type])

    # Return a Cirq Circuit
    cirq_scaled = scaling_function(circuit, return_mitiq=True)
    assert isinstance(cirq_scaled, cirq.Circuit)
    assert _equal(cirq_scaled, cirq_circuit)


@pytest.mark.parametrize("nbits", [1, 10])
@pytest.mark.parametrize("measure", [True, False])
def test_converter_keeps_register_structure_qiskit(nbits, measure):
    qreg = qiskit.QuantumRegister(nbits)
    creg = qiskit.ClassicalRegister(nbits)
    circ = qiskit.QuantumCircuit(qreg, creg)
    circ.h(qreg)

    if measure:
        circ.measure(qreg, creg)

    scaled = scaling_function(circ)

    assert scaled.qregs == circ.qregs
    assert scaled.cregs == circ.cregs
    assert scaled == circ


@pytest.mark.parametrize("nbits", [1, 10])
@pytest.mark.parametrize("measure", [True, False])
def test_converter_keeps_register_structure_qiskit_one_to_many(nbits, measure):
    qreg = qiskit.QuantumRegister(nbits)
    creg = qiskit.ClassicalRegister(nbits)
    circ = qiskit.QuantumCircuit(qreg, creg)
    circ.h(qreg)

    if measure:
        circ.measure(qreg, creg)

    out_circuits = one_to_many_circuit_modifier(circ)

    for out_circ in out_circuits:
        assert out_circ.qregs == circ.qregs
        assert out_circ.cregs == circ.cregs
        print(out_circ)
    assert out_circuits[0] == circ
    assert out_circuits[1] != circ


@pytest.mark.parametrize("to_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_atomic_one_to_many_converter(to_type):
    circuit = convert_from_mitiq(cirq_circuit, to_type)
    circuits = returns_several_circuits(circuit)
    for circuit in circuits:
        assert isinstance(circuit, circuit_types[to_type])

    circuits = returns_several_circuits(circuit, return_mitiq=True)
    for circuit in circuits:
        assert isinstance(circuit, cirq.Circuit)


def test_noise_scaling_converter_with_qiskit_idle_qubits_and_barriers():
    """Idle qubits must be preserved even if the input has barriers.
    Test input:
         ┌───┐ ░
    q_0: ┤ X ├─░─
         └───┘ ░
    q_1: ──────░─
         ┌───┐ ░
    q_2: ┤ X ├─░─
         └───┘ ░
    q_3: ────────
    Expected output:
         ┌───┐
    q_0: ┤ X ├
         └───┘
    q_1: ─────
         ┌───┐
    q_2: ┤ X ├
         └───┘
    q_3: ─────
    """
    test_circuit_qiskit = qiskit.QuantumCircuit(4)
    test_circuit_qiskit.x(0)
    test_circuit_qiskit.x(2)
    test_circuit_qiskit.barrier(0, 1, 2)
    test_copy = test_circuit_qiskit.copy()

    scaled = scaling_function(test_circuit_qiskit)
    # Mitiq is expected to remove qiskit barriers
    expected = qiskit.QuantumCircuit(4)
    expected.x(0)
    expected.x(2)
    assert scaled == expected
    # Mitiq should not mutate the input circuit
    assert test_circuit_qiskit == test_copy
