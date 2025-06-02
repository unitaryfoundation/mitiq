# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for creating mirror circuits as defined in
:cite:`Proctor_2021_NatPhys` for benchmarking quantum computers
(with error mitigation)."""

import cirq
import networkx as nx
from cirq.experiments.qubit_characterizations import _single_qubit_cliffords
from numpy import random

from mitiq import QPROGRAM, Bitstring
from mitiq.interface import convert_from_mitiq

single_q_cliffords = _single_qubit_cliffords()
cliffords = single_q_cliffords.c1_in_xy
paulis = [cirq.X, cirq.Y, cirq.Z, cirq.I]


def random_paulis(
    connectivity_graph: nx.Graph, random_state: random.RandomState
) -> cirq.Circuit:
    """Returns a circuit with randomly selected Pauli gates on each qubit.

    Args:
        connectivity_graph: Connectivity graph of device to run circuit on.
        random_state: Random state to select Paulis I, X, Y, Z uniformly at
            random.
    """
    return cirq.Circuit(
        paulis[random_state.randint(len(paulis))](cirq.LineQubit(x))
        for x in connectivity_graph.nodes
    )


def edge_grab(
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    random_state: random.RandomState,
) -> nx.Graph:
    """
    Args:
        two_qubit_gate_prob: Probability of an edge being chosen
            from the set of candidate edges.
        connectivity_graph: The connectivity graph for the backend
            on which the circuit will be run.
        random_state: Random state to select edges (uniformly at random).

    Returns:
        Returns a set of edges for which two qubit gates are to be
        applied given a two qubit gate density and the connectivity graph
        that must be satisfied.
    """
    connectivity_graph = connectivity_graph.copy()
    candidate_edges = nx.Graph()

    final_edges = nx.Graph()
    final_edges.add_nodes_from(connectivity_graph)

    while connectivity_graph.edges:
        num = random_state.randint(connectivity_graph.size())
        edges = list(connectivity_graph.edges)
        curr_edge = edges[num]
        candidate_edges.add_edge(*curr_edge)
        connectivity_graph.remove_nodes_from(curr_edge)

    for edge in candidate_edges.edges:
        if random_state.uniform(0.0, 1.0) < two_qubit_gate_prob:
            final_edges.add_edge(*edge)
    return final_edges


def random_cliffords(
    connectivity_graph: nx.Graph,
    random_state: random.RandomState,
    two_qubit_gate: cirq.Gate = cirq.CNOT,
) -> cirq.Circuit:
    """
    Args:
        connectivity_graph: A graph with the edges for which the
            two-qubit Clifford gate is to be applied.
        random_state: Random state to choose Cliffords (uniformly at random).
        two_qubit_gate: Two-qubit gate to use.

    Returns:
        A circuit with a two-qubit Clifford gate applied to each edge in
        edges, and a random single-qubit Clifford gate applied to every
        other qubit.
    """
    gates = [
        two_qubit_gate.on(cirq.LineQubit(a), cirq.LineQubit(b))
        for a, b in list(connectivity_graph.edges)
    ]
    qubits = nx.Graph()
    qubits.add_nodes_from(nx.isolates(connectivity_graph))
    gates.extend(
        list(random_single_cliffords(qubits, random_state).all_operations())
    )
    return cirq.Circuit(gates)


def random_single_cliffords(
    connectivity_graph: nx.Graph, random_state: random.RandomState
) -> cirq.Circuit:
    """
    Args:
        connectivity_graph: A graph with each node representing a qubit for
            which a random single-qubit Clifford gate is to be applied.
        random_state: Random state to choose Cliffords (uniformly at random).

    Returns:
        A circuit with a random single-qubit Clifford gate applied on each
        given qubit.
    """
    gates: list[cirq.Operation] = []
    for qubit in connectivity_graph.nodes:
        num = random_state.randint(len(cliffords))
        for clifford_gate in cliffords[num]:
            gates.append(clifford_gate(cirq.LineQubit(qubit)))
    return cirq.Circuit(gates)


def generate_mirror_circuit(
    nlayers: int,
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    two_qubit_gate_name: str = "CNOT",
    seed: int | None = None,
    return_type: str | None = None,
) -> tuple[QPROGRAM, Bitstring]:
    """
    Args:
        nlayers: The number of random Clifford layers to be generated.
        two_qubit_gate_prob: Probability of a two-qubit gate being applied.
        connectivity_graph: The connectivity graph of the backend
            on which the mirror circuit will be run. This is used
            to make sure 2-qubit gates are only applied to connected qubits.
        two_qubit_gate_name: Name of two-qubit gate to use. Options are "CNOT"
            and "CZ".
        seed: Seed for generating randomized mirror circuit.
        return_type: String which specifies the type of the
            returned circuit. See the keys of ``mitiq.SUPPORTED_PROGRAM_TYPES``
            for options. If ``None``, the returned circuit is a
            ``cirq.Circuit``.

    Returns:
        A randomized mirror circuit and the bitstring corresponding
        to a noise free result.
    """
    if not 0 <= two_qubit_gate_prob <= 1:
        raise ValueError("two_qubit_gate_prob must be between 0 and 1")

    supported_two_qubit_gates = {"CZ": cirq.CZ, "CNOT": cirq.CNOT}
    if two_qubit_gate_name not in supported_two_qubit_gates.keys():
        raise ValueError(
            f"Supported two-qubit gate names are "
            f"{tuple(supported_two_qubit_gates.keys())} but "
            f"{two_qubit_gate_name} was provided for `two_qubit_gate_name`."
        )
    two_qubit_gate = supported_two_qubit_gates[two_qubit_gate_name]

    random_state = random.RandomState(seed)

    single_qubit_cliffords = random_single_cliffords(
        connectivity_graph, random_state=random_state
    )

    forward_circuit = cirq.Circuit()

    quasi_inversion_circuit = cirq.Circuit()
    quasi_inverse_gates = []

    for _ in range(nlayers):
        forward_circuit.append(random_paulis(connectivity_graph, random_state))

        selected_edges = edge_grab(
            two_qubit_gate_prob, connectivity_graph, random_state
        )
        circ = random_cliffords(selected_edges, random_state, two_qubit_gate)
        forward_circuit.append(circ)

        quasi_inverse_gates.append(
            random_paulis(connectivity_graph, random_state)
        )
        quasi_inverse_gates.append(cirq.inverse(circ))

    quasi_inversion_circuit.append(
        gate for gate in reversed(quasi_inverse_gates)
    )

    rand_paulis = cirq.Circuit(random_paulis(connectivity_graph, random_state))
    circuit = (
        single_qubit_cliffords
        + forward_circuit
        + rand_paulis
        + quasi_inversion_circuit
        + cirq.inverse(single_qubit_cliffords)
    )

    # Compute the bitstring this circuit should sample.
    res = cirq.Simulator().run(
        circuit + cirq.measure(*sorted(circuit.all_qubits()))
    )
    bitstring = list(res.measurements.values())[0][0].tolist()

    return_type = "cirq" if not return_type else return_type
    return convert_from_mitiq(circuit, return_type), bitstring
