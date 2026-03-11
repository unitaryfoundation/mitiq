# Copyright (C) Unitary Foundation
# Portions of this code have been adapted from Cirq's qubit characterizations
# module.
# Original authors: Cirq developers: Xiao Mi, Dave Bacon, Craig Gidney,
# Ping Yeh, Matthew Neely.
# Code URL = ('https://github.com/quantumlib/Cirq/blob/main/cirq-core/cirq/experiments/qubit_characterizations.py').
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for generating randomized benchmarking circuits."""

import cirq
import numpy as np
from cirq.experiments.qubit_characterizations import (
    _find_inv_matrix,
    _reduce_gate_seq,
    _single_qubit_cliffords,
    _two_qubit_clifford,
    _two_qubit_clifford_matrices,
)

from mitiq import QPROGRAM
from mitiq.interface import convert_from_mitiq


def generate_rb_circuits(
    n_qubits: int,
    num_cliffords: int,
    trials: int = 1,
    return_type: str | None = None,
    seed: int | None = None,
) -> list[QPROGRAM]:
    """Returns a list of randomized benchmarking circuits, i.e. circuits that
    are equivalent to the identity.

    Args:
        n_qubits: The number of qubits. Can be either 1 or 2.
        num_cliffords: The number of Clifford group elements in the
            random circuits. This is proportional to the depth per circuit.
        trials: The number of random circuits at each num_cfd.
        return_type: String which specifies the type of the
            returned circuits. See the keys of
            ``mitiq.SUPPORTED_PROGRAM_TYPES`` for options. If ``None``, the
            returned circuits have type ``cirq.Circuit``.
        seed: A seed for generating randomized benchmarking circuits.

    Returns:
        A list of randomized benchmarking circuits.
    """
    if n_qubits not in (1, 2):
        raise ValueError(
            f"Only generates RB circuits on one or two qubits not {n_qubits}."
        )
    qubits = cirq.LineQubit.range(n_qubits)
    cliffords = _single_qubit_cliffords()
    rng = np.random.RandomState(seed)
    if n_qubits == 1:
        c1 = cliffords.c1_in_xy
        circuits = []
        clifford_group_size = 24
        for _ in range(trials):
            gate_ids = list(rng.choice(clifford_group_size, num_cliffords))
            gate_sequence = [
                gate for gate_id in gate_ids for gate in c1[gate_id]
            ]
            gate_sequence.append(_reduce_gate_seq(gate_sequence) ** -1)
            circuits.append(
                cirq.Circuit(gate(qubits[0]) for gate in gate_sequence)
            )

    else:
        clifford_group_size = 11520
        cfd_matrices = _two_qubit_clifford_matrices(
            qubits[0],
            qubits[1],
            cliffords,
        )
        circuits = []
        for _ in range(trials):
            idx_list = list(rng.choice(clifford_group_size, num_cliffords))
            circuit = cirq.Circuit()
            for idx in idx_list:
                circuit.append(
                    _two_qubit_clifford(qubits[0], qubits[1], idx, cliffords)
                )
            inv_idx = _find_inv_matrix(
                cirq.protocols.unitary(circuit), cfd_matrices
            )
            circuit.append(
                _two_qubit_clifford(qubits[0], qubits[1], inv_idx, cliffords)
            )

            circuits.append(circuit)

    return_type = "cirq" if not return_type else return_type
    return [convert_from_mitiq(circuit, return_type) for circuit in circuits]
