# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import cast, List, Optional, Union

import numpy as np

from cirq import Circuit, LineQubit, ops as cirq_ops, protocols
from cirq.linalg.decompositions import (
    deconstruct_single_qubit_matrix_into_angles,
    kak_decomposition,
)
from braket.circuits import (
    gates as braket_gates,
    Circuit as BKCircuit,
    Instruction,
)


def _raise_braket_to_cirq_error(instr: Instruction) -> None:
    raise ValueError(
        f"Unable to convert the instruction {instr} to Cirq. If you think "
        "this is a bug, you can open an issue on the Mitiq GitHub at "
        "https://github.com/unitaryfund/mitiq."
    )


def _raise_cirq_to_braket_error(op: cirq_ops.Operation) -> None:
    raise ValueError(
        f"Unable to convert {op} to Braket. If you think this is a bug, "
        "you can open an issue on the Mitiq GitHub at"
        " https://github.com/unitaryfund/mitiq."
    )


def from_braket(circuit: BKCircuit) -> Circuit:
    """Returns a Cirq circuit equivalent to the input Braket circuit.

    Note: The returned Cirq circuit acts on cirq.LineQubit's with indices equal
    to the qubit indices of the Braket circuit.

    Args:
        circuit: Braket circuit to convert to a Cirq circuit.
    """
    return Circuit(
        _translate_braket_instruction_to_cirq_operation(instr)
        for instr in circuit.instructions
    )


def to_braket(circuit: Circuit) -> BKCircuit:
    """Returns a Braket circuit equivalent to the input Cirq circuit.

    Args:
        circuit: Cirq circuit to convert to a Braket circuit.
    """
    return BKCircuit(
        _translate_cirq_operation_to_braket_instruction(op)
        for op in circuit.all_operations()
    )


def _translate_braket_instruction_to_cirq_operation(
    instr: Instruction,
) -> List[cirq_ops.Operation]:
    """Converts the braket instruction to an equivalent Cirq operation or list
    of Cirq operations.

    Args:
        instr: Braket instruction to convert.

    Raises:
        ValueError: If the instruction cannot be converted to Cirq.
    """
    nqubits = len(instr.target)

    if nqubits == 1:
        return _translate_one_qubit_braket_instruction_to_cirq_operation(instr)

    elif nqubits == 2:
        return _translate_two_qubit_braket_instruction_to_cirq_operation(instr)

    elif nqubits == 3:
        qubits = [LineQubit(int(qubit)) for qubit in instr.target]

        if isinstance(instr.operator, braket_gates.CCNot):
            return [cirq_ops.TOFFOLI.on(*qubits)]
        elif isinstance(instr.operator, braket_gates.CSwap):
            return [cirq_ops.FREDKIN.on(*qubits)]
        else:
            _raise_braket_to_cirq_error(instr)

    # Unknown instructions.
    else:
        _raise_braket_to_cirq_error(instr)

    return None  # type: ignore[return-value]  # pragma: no cover


def _translate_cirq_operation_to_braket_instruction(
    op: cirq_ops.Operation,
) -> List[Instruction]:
    """Converts the Cirq operation to an equivalent Braket instruction or list
    of instructions.

    Args:
        op: Cirq operation to convert.

    Raises:
        ValueError: If the operation cannot be converted to Braket.
    """
    nqubits = protocols.num_qubits(op)

    if nqubits == 1:
        return _translate_one_qubit_cirq_operation_to_braket_instruction(op)

    elif nqubits == 2:
        return _translate_two_qubit_cirq_operation_to_braket_instruction(op)

    elif nqubits == 3:
        qubits = [cast(LineQubit, q).x for q in op.qubits]

        if op == cirq_ops.TOFFOLI.on(*op.qubits):
            return [Instruction(braket_gates.CCNot(), qubits)]
        elif op == cirq_ops.FREDKIN.on(*op.qubits):
            return [Instruction(braket_gates.CSwap(), qubits)]
        else:
            _raise_cirq_to_braket_error(op)
    # Unsupported gates.
    else:
        _raise_cirq_to_braket_error(op)

    return None  # type: ignore[return-value]  # pragma: no cover


def _translate_one_qubit_braket_instruction_to_cirq_operation(
    instr: Instruction,
) -> List[cirq_ops.Operation]:
    """Converts the one-qubit braket instruction to Cirq.

    Args:
        instr: One-qubit Braket instruction to convert.

    Raises:
        ValueError: If the instruction cannot be converted to Cirq.
    """
    qubits = [LineQubit(int(qubit)) for qubit in instr.target]
    gate = instr.operator

    # One-qubit non-parameterized gates.
    if isinstance(gate, braket_gates.I):
        return [cirq_ops.I.on(*qubits)]
    elif isinstance(gate, braket_gates.X):
        return [cirq_ops.X.on(*qubits)]
    elif isinstance(gate, braket_gates.Y):
        return [cirq_ops.Y.on(*qubits)]
    elif isinstance(gate, braket_gates.Z):
        return [cirq_ops.Z.on(*qubits)]
    elif isinstance(gate, braket_gates.H):
        return [cirq_ops.H.on(*qubits)]
    elif isinstance(gate, braket_gates.S):
        return [cirq_ops.S.on(*qubits)]
    elif isinstance(gate, braket_gates.Si):
        return [protocols.inverse(cirq_ops.S.on(*qubits))]
    elif isinstance(gate, braket_gates.T):
        return [cirq_ops.T.on(*qubits)]
    elif isinstance(gate, braket_gates.Ti):
        return [protocols.inverse(cirq_ops.T.on(*qubits))]
    elif isinstance(gate, braket_gates.V):
        return [cirq_ops.X.on(*qubits) ** 0.5]
    elif isinstance(gate, braket_gates.Vi):
        return [cirq_ops.X.on(*qubits) ** -0.5]

    # One-qubit parameterized gates.
    elif isinstance(gate, braket_gates.Rx):
        return [cirq_ops.rx(gate.angle).on(*qubits)]
    elif isinstance(gate, braket_gates.Ry):
        return [cirq_ops.ry(gate.angle).on(*qubits)]
    elif isinstance(gate, braket_gates.Rz):
        return [cirq_ops.rz(gate.angle).on(*qubits)]
    elif isinstance(gate, braket_gates.PhaseShift):
        return [cirq_ops.Z.on(*qubits) ** (gate.angle / np.pi)]

    else:
        _raise_braket_to_cirq_error(instr)

    return None  # type: ignore[return-value]  # pragma: no cover


def _translate_two_qubit_braket_instruction_to_cirq_operation(
    instr: Instruction,
) -> List[cirq_ops.Operation]:
    """Converts the two-qubit braket instruction to Cirq.

    Args:
        instr: Two-qubit Braket instruction to convert.

    Raises:
        ValueError: If the instruction cannot be converted to Cirq.
    """
    qubits = [LineQubit(int(qubit)) for qubit in instr.target]
    gate = instr.operator

    # Two-qubit non-parameterized gates.
    if isinstance(gate, braket_gates.CNot):
        return [cirq_ops.CNOT.on(*qubits)]

    elif isinstance(gate, braket_gates.Swap):
        return [cirq_ops.SWAP.on(*qubits)]
    elif isinstance(gate, braket_gates.ISwap):
        return [cirq_ops.ISWAP.on(*qubits)]
    elif isinstance(gate, braket_gates.CZ):
        return [cirq_ops.CZ.on(*qubits)]
    elif isinstance(gate, braket_gates.CY):
        return [
            protocols.inverse(cirq_ops.S.on(qubits[1])),
            cirq_ops.CNOT.on(*qubits),
            cirq_ops.S.on(qubits[1]),
        ]

    # Two-qubit parameterized gates.
    elif isinstance(gate, braket_gates.CPhaseShift):
        return [cirq_ops.CZ.on(*qubits) ** (gate.angle / np.pi)]
    elif isinstance(gate, braket_gates.CPhaseShift00):
        return [
            cirq_ops.XX(*qubits),
            cirq_ops.CZ.on(*qubits) ** (gate.angle / np.pi),
            cirq_ops.XX(*qubits),
        ]
    elif isinstance(gate, braket_gates.CPhaseShift01):
        return [
            cirq_ops.X(qubits[0]),
            cirq_ops.CZ.on(*qubits) ** (gate.angle / np.pi),
            cirq_ops.X(qubits[0]),
        ]
    elif isinstance(gate, braket_gates.CPhaseShift10):
        return [
            cirq_ops.X(qubits[1]),
            cirq_ops.CZ.on(*qubits) ** (gate.angle / np.pi),
            cirq_ops.X(qubits[1]),
        ]
    elif isinstance(gate, braket_gates.PSwap):
        return [
            cirq_ops.SWAP.on(*qubits),
            cirq_ops.CNOT.on(*qubits),
            cirq_ops.Z.on(qubits[1]) ** (gate.angle / np.pi),
            cirq_ops.CNOT.on(*qubits),
        ]
    elif isinstance(gate, braket_gates.XX):
        return [
            cirq_ops.XXPowGate(
                exponent=gate.angle / np.pi, global_shift=-0.5
            ).on(*qubits)
        ]
    elif isinstance(gate, braket_gates.YY):
        return [
            cirq_ops.YYPowGate(
                exponent=gate.angle / np.pi, global_shift=-0.5
            ).on(*qubits)
        ]
    elif isinstance(gate, braket_gates.ZZ):
        return [
            cirq_ops.ZZPowGate(
                exponent=gate.angle / np.pi, global_shift=-0.5
            ).on(*qubits)
        ]
    elif isinstance(gate, braket_gates.XY):
        return [cirq_ops.ISwapPowGate(exponent=gate.angle / np.pi).on(*qubits)]

    else:
        _raise_braket_to_cirq_error(instr)

    return None  # type: ignore[return-value]  # pragma: no cover


def _translate_one_qubit_cirq_operation_to_braket_instruction(
    op: Union[np.ndarray, cirq_ops.Operation],
    target: Optional[int] = None,
) -> List[Instruction]:
    """Translates a one-qubit Cirq operation to a (sequence of) Braket
    instruction(s) according to the following rules:

    1. Attempts to find a "standard translation" from Cirq to Braket.
        - e.g., checks if `op` is Pauli-X and, if so, returns the Braket X.

    2. If (1) is not successful, decomposes the unitary of `op` to
    Rz(theta) Ry(phi) Rz(lambda) and returns the series of rotations as Braket
    instructions.

    Args:
        op: One-qubit Cirq operation to translate.
        target: Qubit index for the op to act on. Must be specified and if only
            if `op` is given as a numpy array.
    """
    # Translate qubit index.
    if not isinstance(op, np.ndarray):
        target = cast(LineQubit, op.qubits[0]).x

    if target is None:
        raise ValueError(
            "Arg `target` must be specified when `op` is a matrix."
        )

    # Check common single-qubit gates.
    if isinstance(op, cirq_ops.Operation):
        if isinstance(op.gate, cirq_ops.XPowGate):
            exponent = op.gate.exponent

            if np.isclose(exponent, 1.0) or np.isclose(exponent, -1.0):
                return [Instruction(braket_gates.X(), target)]
            elif np.isclose(exponent, 0.5):
                return [Instruction(braket_gates.V(), target)]
            elif np.isclose(exponent, -0.5):
                return [Instruction(braket_gates.Vi(), target)]

            return [Instruction(braket_gates.Rx(exponent * np.pi), target)]

        elif isinstance(op.gate, cirq_ops.YPowGate):
            exponent = op.gate.exponent

            if np.isclose(exponent, 1.0) or np.isclose(exponent, -1.0):
                return [Instruction(braket_gates.Y(), target)]

            return [Instruction(braket_gates.Ry(exponent * np.pi), target)]

        elif isinstance(op.gate, cirq_ops.ZPowGate):
            exponent = op.gate.exponent

            if np.isclose(exponent, 1.0) or np.isclose(exponent, -1.0):
                return [Instruction(braket_gates.Z(), target)]
            elif np.isclose(exponent, 0.5):
                return [Instruction(braket_gates.S(), target)]
            elif np.isclose(exponent, -0.5):
                return [Instruction(braket_gates.Si(), target)]
            elif np.isclose(exponent, 0.25):
                return [Instruction(braket_gates.T(), target)]
            elif np.isclose(exponent, -0.25):
                return [Instruction(braket_gates.Ti(), target)]

            return [Instruction(braket_gates.Rz(exponent * np.pi), target)]

        elif isinstance(op.gate, cirq_ops.HPowGate) and np.isclose(
            abs(op.gate.exponent), 1.0
        ):
            return [Instruction(braket_gates.H(), target)]

    # Arbitrary single-qubit unitary decomposition.
    # TODO: This does not account for global phase.
    if isinstance(op, cirq_ops.Operation):
        unitary_matrix = protocols.unitary(op)
    else:
        unitary_matrix = op

    a, b, c = deconstruct_single_qubit_matrix_into_angles(unitary_matrix)
    return [
        Instruction(braket_gates.Rz(a), target),
        Instruction(braket_gates.Ry(b), target),
        Instruction(braket_gates.Rz(c), target),
    ]


def _translate_two_qubit_cirq_operation_to_braket_instruction(
    op: cirq_ops.Operation,
) -> List[Instruction]:
    """Translates a two-qubit Cirq operation to a (sequence of) Braket
    instruction(s) according to the following rules:

    1. Attempts to find a "standard translation" from Cirq to Braket.
        - e.g., checks if `op` is a CNOT and, if so, returns the Braket CNOT.

    2. If (1) is not successful, performs a KAK decomposition of the unitary of
    `op` to obtain a circuit

        ──A1──X^0.5───@───X^a───X──────────────────@───B1───
                      │         │                  │
        ──A2──────────X───Y^b───@───X^-0.5───Z^c───X───B2────

    where A1, A2, B1, and B2 are arbitrary single-qubit unitaries and a, b, c
    are floats.

    Args:
        op: Two-qubit Cirq operation to translate.
    """
    # Translate qubit indices.
    q1, q2 = [cast(LineQubit, qubit).x for qubit in op.qubits]

    # Check common two-qubit gates.
    if isinstance(op.gate, cirq_ops.CNotPowGate) and np.isclose(
        abs(op.gate.exponent), 1.0
    ):
        return [Instruction(braket_gates.CNot(), [q1, q2])]
    elif isinstance(op.gate, cirq_ops.CZPowGate) and np.isclose(
        abs(op.gate.exponent), 1.0
    ):
        return [Instruction(braket_gates.CZ(), [q1, q2])]
    elif isinstance(op.gate, cirq_ops.ISwapPowGate) and np.isclose(
        op.gate.exponent, 1.0
    ):
        return [Instruction(braket_gates.ISwap(), [q1, q2])]
    elif isinstance(op.gate, cirq_ops.XXPowGate):
        return [
            Instruction(braket_gates.XX(op.gate.exponent * np.pi), [q1, q2])
        ]
    elif isinstance(op.gate, cirq_ops.YYPowGate):
        return [
            Instruction(braket_gates.YY(op.gate.exponent * np.pi), [q1, q2])
        ]
    elif isinstance(op.gate, cirq_ops.ZZPowGate):
        return [
            Instruction(braket_gates.ZZ(op.gate.exponent * np.pi), [q1, q2])
        ]

    # Arbitrary two-qubit unitary decomposition.
    kak = kak_decomposition(protocols.unitary(op))
    A1, A2 = kak.single_qubit_operations_before

    x, y, z = kak.interaction_coefficients
    a = x * -2 / np.pi + 0.5
    b = y * -2 / np.pi + 0.5
    c = z * -2 / np.pi + 0.5

    B1, B2 = kak.single_qubit_operations_after

    return [
        *_translate_one_qubit_cirq_operation_to_braket_instruction(A1, q1),
        *_translate_one_qubit_cirq_operation_to_braket_instruction(A2, q2),
        Instruction(braket_gates.Rx(0.5 * np.pi), q1),
        Instruction(braket_gates.CNot(), [q1, q2]),
        Instruction(braket_gates.Rx(a * np.pi), q1),
        Instruction(braket_gates.Ry(b * np.pi), q2),
        Instruction(braket_gates.CNot(), [q2, q1]),
        Instruction(braket_gates.Rx(-0.5 * np.pi), q2),
        Instruction(braket_gates.Rz(c * np.pi), q2),
        Instruction(braket_gates.CNot(), [q1, q2]),
        *_translate_one_qubit_cirq_operation_to_braket_instruction(B1, q1),
        *_translate_one_qubit_cirq_operation_to_braket_instruction(B2, q2),
    ]
