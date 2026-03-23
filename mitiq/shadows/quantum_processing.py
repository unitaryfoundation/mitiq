# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Quantum processing functions for classical shadows."""

from collections.abc import Callable, Sequence

import cirq
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from mitiq import MeasurementResult


def sample_random_pauli_bases(num_qubits: int, num_strings: int) -> list[str]:
    """Generate a list of random Pauli strings.

    Args:
        num_qubits: The number of qubits in the Pauli strings.
        num_strings: The number of Pauli strings to generate.

    Returns:
        A list of random Pauli strings.
    """

    # Sample random Pauli operators uniformly from ("X", "Y", "Z")
    pauli_bases = ("X", "Y", "Z")
    paulis = np.random.choice(pauli_bases, (num_strings, num_qubits))
    return ["".join(row) for row in paulis]


def get_rotated_circuits(
    circuit: cirq.Circuit,
    pauli_strings: list[str],
    qubits: Sequence[cirq.Qid] | None = None,
) -> list[cirq.Circuit]:
    """Returns a list of circuits measured in bases corresponding to
    ``pauli_strings``.

    Args:
        circuit: The circuit of interest.
        pauli_strings: The Pauli strings to measure.
        qubits: The qubits to measure. If None, all qubits in the circuit.

    Returns: The list of circuits with rotation and measurement gates appended.
    """
    circuits = []
    qubits = sorted(circuit.all_qubits()) if qubits is None else list(qubits)

    rotations = {
        "X": lambda q: cirq.H(q),
        "Y": lambda q: (cirq.S(q) ** -1, cirq.H(q)),
        "Z": lambda q: (),
    }

    for pauli_string in pauli_strings:
        if len(pauli_string) != len(qubits):
            raise ValueError("Pauli string length must match # of qubits.")

        circuit_in_pauli_basis = circuit.copy()

        for qubit, pauli in zip(qubits, pauli_string):
            if pauli not in rotations.keys():
                raise ValueError(f"Pauli must be X, Y, or Z. Got {pauli!r}.")

            circuit_in_pauli_basis.append(rotations[pauli](qubit))

        circuit_in_pauli_basis.append(cirq.measure(*qubits))
        circuits.append(circuit_in_pauli_basis)

    return circuits


def random_pauli_measurement(
    circuit: cirq.Circuit,
    num_measurements: int,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    qubits: list[cirq.Qid] | None = None,
) -> tuple[list[str], list[str]]:
    r"""This function performs random Pauli measurements on a given circuit and
    returns the outcomes. These outcomes are represented as a tuple of two
    lists of strings.

    Args:
        circuit: A Cirq circuit.
        num_measurements: The number of snapshots.
        executor: A callable that runs a circuit and returns a single
            bitstring.
        qubits: The qubits in the circuit to be measured. If None,
            all qubits in the circuit will be measured.

    Warning:
        The ``executor`` must return a ``MeasurementResult`` for a single shot,
        i.e., a single bitstring.

    Returns:
        Tuple containing two lists of strings, each of length equal to
        ``num_measurements``. Strings in the first list are sequences of
        0's and 1's, which represent qubit measurements outcomes in the
        computational basis (e.g. "01001"). Strings in the second list are
        sequences of Pauli-measurement performed on each qubit (e.g. "XZZYY").
    """

    qubits = sorted(list(circuit.all_qubits())) if qubits is None else qubits
    num_qubits = len(qubits)
    pauli_strings = sample_random_pauli_bases(num_qubits, num_measurements)

    # Rotate and attach measurement gates to the circuit
    rotated_circuits = get_rotated_circuits(
        circuit=circuit,
        pauli_strings=pauli_strings,
        qubits=qubits,
    )

    if tqdm is not None:
        rotated_circuits = tqdm(
            rotated_circuits,
            desc="Measurement",
            leave=False,
        )
    results = [
        executor(rotated_circuit) for rotated_circuit in rotated_circuits
    ]

    shadow_outcomes = []
    for result in results:
        bitstring = list(result.get_counts().keys())[0]
        if len(result.get_counts().keys()) > 1:
            raise ValueError(
                "The `executor` must return a `MeasurementResult` "
                "for a single shot"
            )
        shadow_outcomes.append(bitstring)

    return shadow_outcomes, pauli_strings
