# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import random
from collections.abc import Callable
from functools import singledispatch

import cirq
from cirq import Circuit as _Circuit

from mitiq import QPROGRAM
from mitiq.interface import accept_qprogram_and_validate

# P, Q, R, S from https://arxiv.org/pdf/2301.02690.pdf
CNOT_twirling_gates = [
    (cirq.I, cirq.I, cirq.I, cirq.I),
    (cirq.I, cirq.X, cirq.I, cirq.X),
    (cirq.I, cirq.Y, cirq.Z, cirq.Y),
    (cirq.I, cirq.Z, cirq.Z, cirq.Z),
    (cirq.Y, cirq.I, cirq.Y, cirq.X),
    (cirq.Y, cirq.X, cirq.Y, cirq.I),
    (cirq.Y, cirq.Y, cirq.X, cirq.Z),
    (cirq.Y, cirq.Z, cirq.X, cirq.Y),
    (cirq.X, cirq.I, cirq.X, cirq.X),
    (cirq.X, cirq.X, cirq.X, cirq.I),
    (cirq.X, cirq.Y, cirq.Y, cirq.Z),
    (cirq.X, cirq.Z, cirq.Y, cirq.Y),
    (cirq.Z, cirq.I, cirq.Z, cirq.I),
    (cirq.Z, cirq.X, cirq.Z, cirq.X),
    (cirq.Z, cirq.Y, cirq.I, cirq.Y),
    (cirq.Z, cirq.Z, cirq.I, cirq.Z),
]
CZ_twirling_gates = [
    (cirq.I, cirq.I, cirq.I, cirq.I),
    (cirq.I, cirq.X, cirq.Z, cirq.X),
    (cirq.I, cirq.Y, cirq.Z, cirq.Y),
    (cirq.I, cirq.Z, cirq.I, cirq.Z),
    (cirq.X, cirq.I, cirq.X, cirq.Z),
    (cirq.X, cirq.X, cirq.Y, cirq.Y),
    (cirq.X, cirq.Y, cirq.Y, cirq.X),
    (cirq.X, cirq.Z, cirq.X, cirq.I),
    (cirq.Y, cirq.I, cirq.Y, cirq.Z),
    (cirq.Y, cirq.X, cirq.X, cirq.Y),
    (cirq.Y, cirq.Y, cirq.X, cirq.X),
    (cirq.Y, cirq.Z, cirq.Y, cirq.I),
    (cirq.Z, cirq.I, cirq.Z, cirq.I),
    (cirq.Z, cirq.X, cirq.I, cirq.X),
    (cirq.Z, cirq.Y, cirq.I, cirq.Y),
    (cirq.Z, cirq.Z, cirq.Z, cirq.Z),
]

CIRQ_NOISE_FUNCTION = Callable[[float], cirq.Gate]

CIRQ_NOISE_OP: dict[str, CIRQ_NOISE_FUNCTION] = {
    "bit-flip": cirq.bit_flip,
    "depolarize": cirq.depolarize,
}


def generate_pauli_twirl_variants(
    circuit: QPROGRAM,
    num_circuits: int = 10,
    noise_name: str | None = None,
    **kwargs: float,
) -> list[QPROGRAM]:
    r"""Return the Pauli twirled versions of the input circuit.

    Only the CNOT and CZ gates in an input circuit are Pauli twirled
    as specified in :cite:`Saki_2023_arxiv`.

    Args:
        circuit: The input circuit on which twirling is applied.
        num_circuits: Number of twirled variants of the circuits.
        noise_name: Name of the noisy operator acting on CNOT and CZ gates.
            This is useful if the user requires a noisy circuit after twirling.
            Values allowed: ["bit-flip", "depolarize"]

    Returns:
        A list of `num_circuits` twirled versions of `circuit`
    """
    CNOT_twirled_circuits = twirl_CNOT_gates(circuit, num_circuits)
    twirled_circuits = [
        twirl_CZ_gates(c, num_circuits=1)[0] for c in CNOT_twirled_circuits
    ]

    if noise_name is not None:
        twirled_circuits = [
            add_noise_to_two_qubit_gates(circuit, noise_name, **kwargs)
            for circuit in twirled_circuits
        ]

    return twirled_circuits


def add_noise_to_two_qubit_gates(
    circuit: QPROGRAM, noise_name: str, **kwargs: float
) -> QPROGRAM:
    """Add noise to CNOT and CZ gates on pre-twirled circuits.

    Args:
        circuit: Pre-twirled circuit
        noise_name: name of noise operator to apply after CNOT and CZ gates
    """
    # here we will validate if noise_op and kwargs are valid
    return _add_noise_to_two_qubit_gates(circuit, noise_name, **kwargs)


@singledispatch
def _add_noise_to_two_qubit_gates(
    circuit: QPROGRAM, noise_name: str, **kwargs: float
) -> QPROGRAM:
    raise NotImplementedError(
        f"Cannot add noise to Circuit of type {type(circuit)}."
    )


@_add_noise_to_two_qubit_gates.register
def _cirq(circuit: _Circuit, noise_name: str, **kwargs: float) -> _Circuit:
    noise_function = CIRQ_NOISE_OP[noise_name]
    noise_op = noise_function(**kwargs)  # type: ignore

    noisy_gates = [cirq.ops.CNOT, cirq.ops.CZ]

    noisy_circuit = cirq.Circuit()
    for moment in circuit:
        layer = cirq.Circuit()
        for op in moment:
            layer.append(op)
            if op.gate in noisy_gates:
                layer.append(noise_op.on_each(op.qubits))
        noisy_circuit += layer
    return noisy_circuit


if importlib.util.find_spec("pennylane") is not None:
    import pennylane as qml
    from pennylane.tape import QuantumTape

    PENNYLANE_NOISE_OP = {
        "bit-flip": qml.BitFlip,
        "depolarize": qml.DepolarizingChannel,
    }

    @_add_noise_to_two_qubit_gates.register
    def _pennylane(
        circuit: QuantumTape, noise_name: str, **kwargs: float
    ) -> QuantumTape:
        new_ops = []
        noise_function = PENNYLANE_NOISE_OP[noise_name]

        noisy_gates = ["CNOT", "CZ"]
        for op in circuit:
            new_ops.append(op)
            if op.name in noisy_gates:
                for wire in op.wires:
                    noise_op = noise_function(**kwargs, wires=wire)
                    new_ops.append(noise_op)

        return QuantumTape(
            ops=new_ops, measurements=circuit.measurements, shots=circuit.shots
        )


def twirl_CNOT_gates(circuit: QPROGRAM, num_circuits: int) -> list[QPROGRAM]:
    """Generate a list of circuits using Pauli twirling on CNOT gates.

    Args:
        circuit: The circuit to generate twirled versions of
        num_circuits: The number of sampled circuits to return
    """
    return [_twirl_CNOT_qprogram(circuit) for _ in range(num_circuits)]


@accept_qprogram_and_validate
def _twirl_CNOT_qprogram(circuit: cirq.Circuit) -> cirq.Circuit:
    return circuit.map_operations(_twirl_single_CNOT_gate)


def twirl_CZ_gates(circuit: QPROGRAM, num_circuits: int) -> list[QPROGRAM]:
    """Generate a list of circuits using Pauli twirling on CZ gates.

    Args:
        circuit: The circuit to generate twirled versions of
        num_circuits: The number of sampled circuits to return
    """
    return [_twirl_CZ_qprogram(circuit) for _ in range(num_circuits)]


@accept_qprogram_and_validate
def _twirl_CZ_qprogram(circuit: cirq.Circuit) -> cirq.Circuit:
    return circuit.map_operations(_twirl_single_CZ_gate)


def _twirl_single_CNOT_gate(op: cirq.Operation) -> cirq.OP_TREE:
    """Function which converts a CNOT gate to a logical equivalent. That is,
    it converts CNOT operations into the following, leaving all other cirq
    operations alone.

        --P---⏺---R--
              |
        --Q---⊕---S--

    Where P, Q, R, and S are Pauli operators sampled so as to not effect
    change on the underlying unitary.
    """
    if op.gate != cirq.CNOT:
        return op

    P, Q, R, S = random.choice(CNOT_twirling_gates)
    control_qubit, target_qubit = op.qubits
    return [
        P.on(control_qubit),
        Q.on(target_qubit),
        op,
        R.on(control_qubit),
        S.on(target_qubit),
    ]


def _twirl_single_CZ_gate(op: cirq.Operation) -> cirq.OP_TREE:
    """Function which converts a CZ gate to a logical equivalent. That is, it
    converts CZ operations into the following, leaving all other cirq
    operations alone.

        --P---⏺---R--
              |
        --Q---⏺---S--

    Where P, Q, R, and S are Pauli operators sampled so as to not effect
    change on the underlying unitary.
    """
    if op.gate != cirq.CZ:
        return op

    P, Q, R, S = random.choice(CZ_twirling_gates)
    control_qubit, target_qubit = op.qubits
    return [
        P.on(control_qubit),
        Q.on(target_qubit),
        op,
        R.on(control_qubit),
        S.on(target_qubit),
    ]
