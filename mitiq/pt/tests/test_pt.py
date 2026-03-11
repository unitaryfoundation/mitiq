# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import random

import cirq
import networkx as nx
import numpy as np
import pennylane as qml
import pytest
import qiskit
from pennylane.tape import QuantumTape

from mitiq.benchmarks import generate_mirror_circuit
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.pt.pt import (
    CIRQ_NOISE_OP,
    PENNYLANE_NOISE_OP,
    CNOT_twirling_gates,
    CZ_twirling_gates,
    generate_pauli_twirl_variants,
    twirl_CNOT_gates,
    twirl_CZ_gates,
)
from mitiq.utils import _equal

num_qubits = 2
qubits = cirq.LineQubit.range(num_qubits)
circuit = cirq.Circuit()
circuit.append(cirq.CNOT(*qubits))
circuit.append(cirq.CZ(*qubits))


def amp_damp_executor(circuit: cirq.Circuit, noise: float = 0.005) -> float:
    return compute_density_matrix(
        circuit, noise_model_function=cirq.amplitude_damp, noise_level=(noise,)
    )[0, 0].real


def test_twirl_CNOT_implements_same_unitary():
    num_circuits = 1
    twirled = twirl_CNOT_gates(circuit, num_circuits=num_circuits)
    assert len(twirled) == num_circuits
    original_unitary = cirq.unitary(circuit)
    twirled_unitary = cirq.unitary(twirled[0])
    assert np.array_equal(twirled_unitary, original_unitary) or np.array_equal(
        -1 * twirled_unitary, original_unitary
    )


def test_twirl_CZ_implements_same_unitary():
    num_circuits = 1
    twirled = twirl_CZ_gates(circuit, num_circuits=num_circuits)
    assert len(twirled) == num_circuits
    original_unitary = cirq.unitary(circuit)
    twirled_unitary = cirq.unitary(twirled[0])
    assert np.array_equal(twirled_unitary, original_unitary) or np.array_equal(
        -1 * twirled_unitary, original_unitary
    )


def test_CNOT_twirl_table():
    a, b = cirq.LineQubit.range(2)
    for P, Q, R, S in CNOT_twirling_gates:
        circuit = cirq.Circuit(
            P.on(a),
            Q.on(b),
            cirq.CNOT.on(a, b),
            R.on(a),
            S.on(b),
            cirq.CNOT.on(a, b),
        )
        assert np.allclose(cirq.unitary(circuit), np.eye(4)) or np.allclose(
            -1 * cirq.unitary(circuit), np.eye(4)
        )


def test_CZ_twirl_table():
    a, b = cirq.LineQubit.range(2)
    for P, Q, R, S in CZ_twirling_gates:
        circuit = cirq.Circuit(
            P.on(a),
            Q.on(b),
            cirq.CZ.on(a, b),
            R.on(a),
            S.on(b),
            cirq.CZ.on(a, b),
        )
        assert np.allclose(cirq.unitary(circuit), np.eye(4)) or np.allclose(
            -1 * cirq.unitary(circuit), np.eye(4)
        )


def test_twirl_CNOT_qiskit():
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    num_circuits = 10
    twirled = twirl_CNOT_gates(qc, num_circuits=num_circuits)
    assert len(twirled) == num_circuits
    random_index = random.randint(0, 9)
    assert isinstance(twirled[random_index], qiskit.QuantumCircuit)


def test_twirl_CNOT_increases_layer_count():
    num_qubits = 3
    num_layers = 10
    gateset = {cirq.X: 1, cirq.Y: 1, cirq.Z: 1, cirq.H: 1, cirq.CNOT: 2}
    circuit = cirq.testing.random_circuit(
        num_qubits, num_layers, op_density=0.8, gate_domain=gateset
    )
    num_CNOTS = sum([op.gate == cirq.CNOT for op in circuit.all_operations()])
    twirled = twirl_CNOT_gates(circuit, num_circuits=1)[0]
    num_gates_before = len(list(circuit.all_operations()))
    num_gates_after = len(list(twirled.all_operations()))
    if num_CNOTS:
        assert num_gates_after > num_gates_before
    else:
        assert num_gates_after == num_gates_before


def test_generate_pauli_twirl_variants():
    num_qubits = 3
    num_layers = 20
    circuit, _ = generate_mirror_circuit(
        nlayers=num_layers,
        two_qubit_gate_prob=1.0,
        connectivity_graph=nx.complete_graph(num_qubits),
    )
    num_circuits = 10
    twirled_output = generate_pauli_twirl_variants(circuit, num_circuits)
    assert len(twirled_output) == num_circuits


@pytest.mark.parametrize(
    "twirl_func",
    [generate_pauli_twirl_variants, twirl_CNOT_gates, twirl_CZ_gates],
)
def test_no_CNOT_CZ_circuit(twirl_func):
    num_qubits = 2
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.X.on_each(qubits))
    twirled_output = twirl_func(circuit, 5)
    assert len(twirled_output) == 5

    for i in range(5):
        assert _equal(circuit, twirled_output[i])


@pytest.mark.parametrize("noise_name", ["bit-flip", "depolarize"])
def test_noisy_cirq(noise_name):
    p = 0.01
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H.on(a), cirq.CNOT.on(a, b), cirq.CZ.on(a, b))
    twirled_circuit = generate_pauli_twirl_variants(
        circuit, num_circuits=1, noise_name=noise_name, p=p
    )[0]

    for i, current_moment in enumerate(twirled_circuit):
        for op in current_moment:
            if op.gate in [cirq.CNOT, cirq.CZ]:
                for next_op in twirled_circuit[i + 1]:
                    assert next_op.gate == CIRQ_NOISE_OP[noise_name](p=p)


@pytest.mark.parametrize("noise_name", ["bit-flip", "depolarize"])
def test_noisy_pennylane(noise_name):
    p = 0.01
    ops = [
        qml.Hadamard(0),
        qml.CNOT((0, 1)),
        qml.CZ((0, 1)),
    ]
    circuit = QuantumTape(ops)
    twirled_circuit = generate_pauli_twirl_variants(
        circuit, num_circuits=1, noise_name=noise_name, p=p
    )[0]

    noisy_gates = ["CNOT", "CZ"]

    for i, op in enumerate(twirled_circuit.operations):
        if op.name in noisy_gates:
            for j in range(1, len(op.wires) + 1):
                assert isinstance(
                    twirled_circuit.operations[i + j],
                    PENNYLANE_NOISE_OP[noise_name],
                )
