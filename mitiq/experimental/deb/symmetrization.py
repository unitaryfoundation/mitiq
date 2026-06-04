# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Circuit symmetrization for debiasing error mitigation."""

from typing import List

import cirq
import numpy as np

from mitiq import QPROGRAM


def construct_circuits(
    circuit: QPROGRAM,
    num_variants: int,
    random_state: int | None = None,
) -> List[cirq.Circuit]:
    """Generate Pauli-symmetrized circuit variants for debiasing.

    For each variant, random single-qubit Pauli operators are sampled from
    {I, X, Y, Z} per qubit and applied before and after the circuit.
    Since Paulis are self-inverse, they cancel exactly for an ideal circuit,
    but coherent errors get dressed differently in each variant and
    average out.

    Args:
        circuit: The input circuit to symmetrize.
        num_variants: Number of circuit variants to generate.
        random_state: Seed for random number generation.

    Returns:
        A list of symmetrized circuit variants.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Convert to Cirq circuit if needed
    if not isinstance(circuit, cirq.Circuit):
        circuit = cirq.Circuit(circuit)

    qubits = list(circuit.all_qubits())
    n_qubits = len(qubits)

    # Pauli operators as Cirq gates
    pauli_gates = {
        0: cirq.I,  # Identity
        1: cirq.X,
        2: cirq.Y,
        3: cirq.Z,
    }

    variants = []

    for _ in range(num_variants):
        # Sample random Pauli for each qubit
        pauli_indices = np.random.randint(0, 4, size=n_qubits)

        # Build symmetrized circuit
        sym_circuit = cirq.Circuit()

        # Prepend Pauli layer
        for i, qubit in enumerate(qubits):
            pauli_gate = pauli_gates[pauli_indices[i]]
            if pauli_gate != cirq.I:
                sym_circuit.append(pauli_gate(qubit))

        # Append original circuit
        sym_circuit += circuit

        # Append same Pauli layer (Paulis are self-inverse)
        for i, qubit in enumerate(qubits):
            pauli_gate = pauli_gates[pauli_indices[i]]
            if pauli_gate != cirq.I:
                sym_circuit.append(pauli_gate(qubit))

        variants.append(sym_circuit)

    return variants
