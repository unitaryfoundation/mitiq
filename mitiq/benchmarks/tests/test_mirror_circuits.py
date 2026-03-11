# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mirror circuits."""

import cirq
import networkx as nx
import pytest
from numpy import random

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks import mirror_circuits
from mitiq.utils import _equal

paulis = mirror_circuits.paulis
cliffords = mirror_circuits.cliffords
single_cliffords = []
for c in cliffords:
    single_cliffords.extend(c)
all_cliffords = single_cliffords.copy()
all_cliffords.append(cirq.CNOT)
all_gates = paulis + all_cliffords
all_gates.append(cirq.measure)


@pytest.mark.parametrize(
    "graph",
    [
        nx.Graph({3: (4,), 4: (3, 6), 5: (6,)}),
        nx.Graph({0: (99,), 10: (20,), 5: (6,)}),
    ],
)
def test_random_paulis(graph: nx.Graph):
    circuit = mirror_circuits.random_paulis(
        connectivity_graph=graph, random_state=random.RandomState()
    )
    assert isinstance(circuit, cirq.Circuit)
    assert len(circuit.all_qubits()) == len(graph.nodes)
    for qubit in circuit.all_qubits():
        assert qubit.x in graph.nodes
    assert len(list(circuit.all_operations())) == len(graph.nodes)
    assert set(op.gate for op in circuit.all_operations()).issubset(paulis)


def test_edge_grab():
    nqubits = [9, 10, 11]
    graphs = []
    for qubits in nqubits:
        graphs.append(nx.complete_graph(qubits))
    edges_to_remove = [
        [(1, 2), (4, 7), (3, 5)],
        [(2, 4), (5, 6), (7, 8)],
        [(3, 4), (5, 6), (8, 9)],
    ]

    for x in range(3):
        graph = graphs[x]
        for edge in edges_to_remove[x]:
            graph.remove_edge(*edge)
        selected_edges = mirror_circuits.edge_grab(
            0.3, graph, random_state=random.RandomState()
        )
        assert set(selected_edges.edges()).issubset(graph.edges())
        assert selected_edges.number_of_nodes() == nqubits[x]
        for node in selected_edges.nodes:
            assert len(list(selected_edges.neighbors(node))) in (0, 1)


def test_random_cliffords():
    nqubits = [9, 10, 11]
    edges = [
        [(1, 2), (4, 7), (3, 5)],
        [(2, 4), (5, 6), (7, 8)],
        [(3, 4), (5, 6), (8, 9)],
    ]
    for x in range(3):
        graph = nx.Graph()
        graph.add_nodes_from(range(nqubits[x]))
        graph.add_edges_from(edges[x])
        circuit = mirror_circuits.random_cliffords(
            graph, random_state=random.RandomState()
        )
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit.all_qubits()) == nqubits[x]
        two_q_gates = set()
        for a, b in edges[x]:
            two_q_gates.add(cirq.CNOT(cirq.LineQubit(a), cirq.LineQubit(b)))
        assert two_q_gates.issubset(circuit.all_operations())
        assert set(op.gate for op in circuit.all_operations()).issubset(
            all_cliffords
        )


def test_random_single_cliffords():
    qubits = [[0, 1, 2, 3, 4], [1, 3, 5, 7], [1, 2, 4, 6, 7, 8]]
    for x in range(3):
        graph = nx.Graph()
        graph.add_nodes_from(qubits[x])
        circuit = mirror_circuits.random_single_cliffords(
            graph, random_state=random.RandomState()
        )
        assert isinstance(circuit, cirq.Circuit)
        assert set(circuit.all_qubits()).issubset(
            [cirq.LineQubit(qubit) for qubit in qubits[x]]
        )
        assert set(op.gate for op in circuit.all_operations()).issubset(
            single_cliffords
        )


@pytest.mark.parametrize(
    "depth_twoqprob_graph",
    [
        (16, 0.3, nx.complete_graph(3)),
        (20, 0.4, nx.complete_graph(4)),
        (24, 0.5, nx.complete_graph(5)),
    ],
)
def test_generate_mirror_circuit(
    depth_twoqprob_graph: tuple[int, float, nx.Graph],
):
    depth, xi, connectivity_graph = depth_twoqprob_graph
    n = connectivity_graph.number_of_nodes()
    circ, _ = mirror_circuits.generate_mirror_circuit(
        depth, xi, connectivity_graph
    )
    assert isinstance(circ, cirq.Circuit)
    assert len(circ.all_qubits()) == n
    assert set(op.gate for op in circ.all_operations()).issubset(all_gates)
    circ.append(cirq.measure(*cirq.LineQubit.range(n)))
    result = (
        cirq.Simulator()
        .run(circ, repetitions=1_000)
        .multi_measurement_histogram(keys=circ.all_measurement_key_names())
    )
    assert (
        len(result.keys()) == 1
    )  # checks that the circuit only outputs one bitstring


@pytest.mark.parametrize("seed", (0, 3))
def test_mirror_circuit_seeding(seed: int):
    nlayers = 5
    two_qubit_gate_prob = 0.4
    connectivity_graph = nx.complete_graph(5)
    circuit, _ = mirror_circuits.generate_mirror_circuit(
        nlayers, two_qubit_gate_prob, connectivity_graph, seed=seed
    )
    for _ in range(5):
        circ, _ = mirror_circuits.generate_mirror_circuit(
            nlayers, two_qubit_gate_prob, connectivity_graph, seed=seed
        )
        assert _equal(
            circuit,
            circ,
            require_qubit_equality=True,
            require_measurement_equality=True,
        )


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_mirror_circuits_conversions(return_type: str):
    nlayers = 5
    two_qubit_gate_prob = 0.4
    connectivity_graph = nx.complete_graph(5)
    circuit, _ = mirror_circuits.generate_mirror_circuit(
        nlayers,
        two_qubit_gate_prob,
        connectivity_graph,
        return_type=return_type,
    )
    assert return_type in circuit.__module__


@pytest.mark.parametrize(
    "twoq_name_and_gate", [("CNOT", cirq.CNOT), ("CZ", cirq.CZ)]
)
def test_two_qubit_gate(twoq_name_and_gate: tuple[str, cirq.Gate]):
    twoq_name, twoq_gate = twoq_name_and_gate
    circuit, _ = mirror_circuits.generate_mirror_circuit(
        nlayers=2,
        two_qubit_gate_prob=1.0,
        connectivity_graph=nx.complete_graph(5),
        two_qubit_gate_name=twoq_name,
    )
    two_qubit_gates = {
        op.gate for op in circuit.all_operations() if len(op.qubits) == 2
    }
    assert two_qubit_gates == {twoq_gate}


def test_two_qubit_gate_unsupported():
    with pytest.raises(ValueError, match="Supported two-qubit gate names are"):
        mirror_circuits.generate_mirror_circuit(
            1, 1.0, nx.complete_graph(2), two_qubit_gate_name="bad_gate_name"
        )


def test_two_qubit_gate_bad_probability():
    with pytest.raises(
        ValueError, match="two_qubit_gate_prob must be between 0 and 1"
    ):
        mirror_circuits.generate_mirror_circuit(1, 2.0, nx.complete_graph(2))


def test_deterministic_correct_bitstrings():
    """For a fixed seed, correct bitstrings should be deterministic."""
    expected_correct_bitstrings = (
        [[0, 0], [1, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
        + [[0, 0], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1]]
        + [[1, 1], [1, 0], [1, 0], [1, 0], [0, 0], [1, 1], [0, 0], [0, 1]]
    )
    for j, expected in enumerate(expected_correct_bitstrings):
        _, bitstring = mirror_circuits.generate_mirror_circuit(
            nlayers=1,
            two_qubit_gate_prob=1.0,
            connectivity_graph=nx.complete_graph(2),
            seed=j,
        )
        assert bitstring == expected
