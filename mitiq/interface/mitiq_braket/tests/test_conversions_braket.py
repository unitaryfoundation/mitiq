# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq_ionq.ionq_native_gates as cirq_ionq_ops
import numpy as np
import pytest
from braket.circuits import Circuit as BKCircuit
from braket.circuits import Instruction
from braket.circuits import gates as braket_gates
from cirq import Circuit, LineQubit, ops, protocols, testing

from mitiq.interface.mitiq_braket.conversions import from_braket, to_braket
from mitiq.utils import _equal


@pytest.mark.parametrize(
    "qreg", (LineQubit.range(2), [LineQubit(1), LineQubit(6)])
)
def test_to_from_braket_bell_circuit(qreg):
    cirq_circuit = Circuit(ops.H(qreg[0]), ops.CNOT(*qreg))
    test_circuit = from_braket(to_braket(cirq_circuit))
    assert _equal(test_circuit, cirq_circuit, require_qubit_equality=True)


def test_from_braket_non_parameterized_single_qubit_gates():
    braket_circuit = BKCircuit()
    instructions = [
        Instruction(braket_gates.I(), target=0),
        Instruction(braket_gates.X(), target=1),
        Instruction(braket_gates.Y(), target=2),
        Instruction(braket_gates.Z(), target=3),
        Instruction(braket_gates.H(), target=0),
        Instruction(braket_gates.S(), target=1),
        Instruction(braket_gates.Si(), target=2),
        Instruction(braket_gates.T(), target=3),
        Instruction(braket_gates.Ti(), target=0),
        Instruction(braket_gates.V(), target=1),
        Instruction(braket_gates.Vi(), target=2),
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    for i, op in enumerate(cirq_circuit.all_operations()):
        assert np.allclose(
            instructions[i].operator.to_matrix(), protocols.unitary(op)
        )

    qreg = LineQubit.range(4)
    expected_cirq_circuit = Circuit(
        ops.I(qreg[0]),
        ops.X(qreg[1]),
        ops.Y(qreg[2]),
        ops.Z(qreg[3]),
        ops.H(qreg[0]),
        ops.S(qreg[1]),
        ops.S(qreg[2]) ** -1,
        ops.T(qreg[3]),
        ops.T(qreg[0]) ** -1,
        ops.X(qreg[1]) ** 0.5,
        ops.X(qreg[2]) ** -0.5,
    )
    assert _equal(cirq_circuit, expected_cirq_circuit)


@pytest.mark.parametrize("qubit_index", (0, 3))
def test_from_braket_parameterized_single_qubit_gates(qubit_index):
    braket_circuit = BKCircuit()
    pgates = [
        braket_gates.Rx,
        braket_gates.Ry,
        braket_gates.Rz,
        braket_gates.PhaseShift,
        braket_gates.GPi,
        braket_gates.GPi2,
    ]
    angles = np.random.RandomState(11).random(len(pgates))
    instructions = [
        Instruction(rot(a), target=qubit_index)
        for rot, a in zip(pgates, angles)
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    for i, op in enumerate(cirq_circuit.all_operations()):
        assert np.allclose(
            instructions[i].operator.to_matrix(), protocols.unitary(op)
        )

    qubit = LineQubit(qubit_index)
    expected_cirq_circuit = Circuit(
        ops.rx(angles[0]).on(qubit),
        ops.ry(angles[1]).on(qubit),
        ops.rz(angles[2]).on(qubit),
        ops.Z.on(qubit) ** (angles[3] / np.pi),
        cirq_ionq_ops.GPIGate(phi=angles[4] / (2 * np.pi)).on(qubit),
        cirq_ionq_ops.GPI2Gate(phi=angles[5] / (2 * np.pi)).on(qubit),
    )
    assert _equal(
        cirq_circuit, expected_cirq_circuit, require_qubit_equality=True
    )


@pytest.mark.parametrize("qubit_index", (0, 3))
def test_from_braket_single_qubit_two_parameter_gates(qubit_index):
    braket_circuit = BKCircuit()
    pgates = [
        braket_gates.PRx,
    ]

    angles = np.random.RandomState(11).random((len(pgates), 2))
    instructions = [
        Instruction(rot(a[0], a[1]), target=qubit_index)
        for rot, a in zip(pgates, angles)
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    for i, op in enumerate(cirq_circuit.all_operations()):
        assert np.allclose(
            instructions[i].operator.to_matrix(), protocols.unitary(op)
        )

    qubit = LineQubit(qubit_index)
    expected_cirq_circuit = Circuit(
        ops.PhasedXPowGate(
            phase_exponent=angles[0][1] / np.pi,
            exponent=angles[0][0] / np.pi,
            global_shift=-0.5,
        ).on(qubit)
    )
    assert _equal(
        cirq_circuit, expected_cirq_circuit, require_qubit_equality=True
    )


def test_from_braket_non_parameterized_two_qubit_gates():
    braket_circuit = BKCircuit()
    instructions = [
        Instruction(braket_gates.CNot(), target=[2, 3]),
        Instruction(braket_gates.Swap(), target=[3, 4]),
        Instruction(braket_gates.ISwap(), target=[2, 3]),
        Instruction(braket_gates.CZ(), target=(3, 4)),
        Instruction(braket_gates.CY(), target=(2, 3)),
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    qreg = LineQubit.range(2, 5)
    expected_cirq_circuit = Circuit(
        ops.CNOT(*qreg[:2]),
        ops.SWAP(*qreg[1:]),
        ops.ISWAP(*qreg[:2]),
        ops.CZ(*qreg[1:]),
        ops.ControlledGate(ops.Y).on(*qreg[:2]),
    )
    assert np.allclose(
        protocols.unitary(cirq_circuit),
        protocols.unitary(expected_cirq_circuit),
    )


def test_from_braket_parameterized_two_qubit_gates():
    pgates = [
        braket_gates.CPhaseShift,
        braket_gates.CPhaseShift00,
        braket_gates.CPhaseShift01,
        braket_gates.CPhaseShift10,
        braket_gates.PSwap,
        braket_gates.XX,
        braket_gates.YY,
        braket_gates.ZZ,
        braket_gates.XY,
    ]
    angles = np.random.RandomState(2).random(len(pgates))
    instructions = [
        Instruction(rot(a), target=[0, 1]) for rot, a in zip(pgates, angles)
    ]

    cirq_circuits = list()
    for instr in instructions:
        braket_circuit = BKCircuit()
        braket_circuit.add_instruction(instr)
        cirq_circuits.append(from_braket(braket_circuit))

    for instr, cirq_circuit in zip(instructions, cirq_circuits):
        assert np.allclose(instr.operator.to_matrix(), cirq_circuit.unitary())


def test_from_braket_parameterized_two_qubit_three_parameters_gates():
    pgates = [
        braket_gates.MS,
    ]
    angles = np.random.RandomState(2).random((len(pgates), 3))
    instructions = [
        Instruction(rot(a[0], a[1], a[2]), target=[0, 1])
        for rot, a in zip(pgates, angles)
    ]

    cirq_circuits = list()
    for instr in instructions:
        braket_circuit = BKCircuit()
        braket_circuit.add_instruction(instr)
        cirq_circuits.append(from_braket(braket_circuit))

    for instr, cirq_circuit in zip(instructions, cirq_circuits):
        assert np.allclose(instr.operator.to_matrix(), cirq_circuit.unitary())


def test_from_braket_three_qubit_gates():
    braket_circuit = BKCircuit()
    instructions = [
        Instruction(braket_gates.CCNot(), target=[1, 2, 3]),
        Instruction(braket_gates.CSwap(), target=[1, 2, 3]),
    ]
    for instr in instructions:
        braket_circuit.add_instruction(instr)
    cirq_circuit = from_braket(braket_circuit)

    qreg = LineQubit.range(1, 4)
    expected_cirq_circuit = Circuit(ops.TOFFOLI(*qreg), ops.FREDKIN(*qreg))
    assert np.allclose(
        protocols.unitary(cirq_circuit),
        protocols.unitary(expected_cirq_circuit),
    )


def _rotation_of_pi_over_7(num_qubits):
    matrix = np.identity(2**num_qubits)
    matrix[0:2, 0:2] = [
        [np.cos(np.pi / 7), np.sin(np.pi / 7)],
        [-np.sin(np.pi / 7), np.cos(np.pi / 7)],
    ]
    return matrix


def test_from_braket_raises_on_unsupported_gates():
    for num_qubits in range(1, 5):
        braket_circuit = BKCircuit()
        instr = Instruction(
            braket_gates.Unitary(_rotation_of_pi_over_7(num_qubits)),
            target=list(range(num_qubits)),
        )
        braket_circuit.add_instruction(instr)
        with pytest.raises(ValueError):
            from_braket(braket_circuit)


def test_to_braket_raises_on_unsupported_gates():
    for num_qubits in range(3, 5):
        qubits = [LineQubit(int(qubit)) for qubit in range(num_qubits)]
        op = ops.MatrixGate(_rotation_of_pi_over_7(num_qubits)).on(*qubits)
        circuit = Circuit(op)
        with pytest.raises(ValueError):
            to_braket(circuit)


def test_to_from_braket_common_one_qubit_gates():
    """These gates should stay the same (i.e., not get decomposed) when
    converting Cirq -> Braket -> Cirq.
    """
    rots = [ops.rx, ops.ry, ops.rz]
    angles = [1 / 5, 3 / 5, -4 / 5]
    qubit = LineQubit(0)
    cirq_circuit = Circuit(
        # Paulis.
        ops.X(qubit),
        ops.Y(qubit),
        ops.Z(qubit),
        # Single-qubit rotations.
        [rot(angle).on(qubit) for rot, angle in zip(rots, angles)],
        # Rz alter egos.
        ops.T(qubit),
        ops.T(qubit) ** -1,
        ops.S(qubit),
        ops.S(qubit) ** -1,
        # Rx alter egos.
        ops.X(qubit) ** 0.5,
        ops.X(qubit) ** -0.5,
    )

    test_circuit = from_braket(to_braket(cirq_circuit))
    assert _equal(test_circuit, cirq_circuit, require_qubit_equality=True)


@pytest.mark.parametrize(
    "uncommon_gate",
    [
        ops.HPowGate(exponent=-1 / 14),
        cirq_ionq_ops.GPIGate(phi=1 / 14),
        cirq_ionq_ops.GPI2Gate(phi=1 / 14),
    ],
)
def test_to_from_braket_uncommon_one_qubit_gates(uncommon_gate):
    """These gates get decomposed when converting Cirq -> Braket -> Cirq, but
    the unitaries should be equal up to global phase.
    """
    cirq_circuit = Circuit(uncommon_gate.on(LineQubit(0)))
    test_circuit = from_braket(to_braket(cirq_circuit))
    testing.assert_allclose_up_to_global_phase(
        protocols.unitary(test_circuit),
        protocols.unitary(cirq_circuit),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "common_gate",
    [
        ops.CNOT,
        ops.CZ,
        ops.ISWAP,
        ops.XXPowGate(exponent=-0.2),
        ops.YYPowGate(exponent=0.3),
        ops.ZZPowGate(exponent=-0.1),
    ],
)
def test_to_from_braket_common_two_qubit_gates(common_gate):
    """These gates should stay the same (i.e., not get decomposed) when
    converting Cirq -> Braket -> Cirq.
    """
    cirq_circuit = Circuit(common_gate.on(*LineQubit.range(2)))
    test_circuit = from_braket(to_braket(cirq_circuit))
    testing.assert_allclose_up_to_global_phase(
        protocols.unitary(test_circuit),
        protocols.unitary(cirq_circuit),
        atol=1e-7,
    )

    # Cirq AAPowGate has a different global phase than braket AA which gets
    # lost in translation. Here, AA = XX, YY, or ZZ.
    if not isinstance(
        common_gate, (ops.XXPowGate, ops.YYPowGate, ops.ZZPowGate)
    ):
        assert _equal(test_circuit, cirq_circuit, require_qubit_equality=True)


@pytest.mark.parametrize(
    "uncommon_gate",
    [
        ops.CNotPowGate(exponent=-1 / 17),
        ops.CZPowGate(exponent=2 / 7),
        cirq_ionq_ops.MSGate(phi0=1 / 7, phi1=2 / 7),
    ],
)
def test_to_from_braket_uncommon_two_qubit_gates(uncommon_gate):
    """These gates get decomposed when converting Cirq -> Braket -> Cirq, but
    the unitaries should be equal up to global phase.
    """
    cirq_circuit = Circuit(uncommon_gate.on(*LineQubit.range(2)))
    test_circuit = from_braket(to_braket(cirq_circuit))

    testing.assert_allclose_up_to_global_phase(
        protocols.unitary(test_circuit),
        protocols.unitary(cirq_circuit),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "common_gate",
    [
        ops.TOFFOLI,
        ops.FREDKIN,
    ],
)
def test_to_from_braket_common_three_qubit_gates(common_gate):
    """These gates should stay the same (i.e., not get decomposed) when
    converting Cirq -> Braket -> Cirq.
    """
    cirq_circuit = Circuit(common_gate.on(*LineQubit.range(3)))
    test_circuit = from_braket(to_braket(cirq_circuit))
    testing.assert_allclose_up_to_global_phase(
        protocols.unitary(test_circuit),
        protocols.unitary(cirq_circuit),
        atol=1e-7,
    )

    assert _equal(test_circuit, cirq_circuit, require_qubit_equality=True)
