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


def generate_random_pauli_strings(
    num_qubits: int, num_strings: int
) -> list[str]:
    """Generate a list of random Pauli strings.

    Args:
        num_qubits: The number of qubits in the Pauli strings.
        num_strings: The number of Pauli strings to generate.

    Returns:
        A list of random Pauli strings.
    """

    # Sample random Pauli operators uniformly from ("X", "Y", "Z")
    unitary_ensemble = ["X", "Y", "Z"]
    paulis = np.random.choice(unitary_ensemble, (num_strings, num_qubits))
    return ["".join(pauli) for pauli in paulis]


def get_rotated_circuits(
    circuit: cirq.Circuit,
    pauli_strings: list[str],
    add_measurements: bool = True,
    qubits: Sequence[cirq.Qid] | None = None,
) -> list[cirq.Circuit]:
    """Returns a list of circuits that are identical to the input circuit,
    except that each one has single-qubit Clifford gates followed by
    measurement gates that are designed to measure the input
    Pauli strings in the Z basis.

    Args:
        circuit: The circuit to measure.
        pauli_strings: The Pauli strings to measure in each output circuit.
        add_measurements: Whether to add measurement gates to the circuit.
        qubits: The qubits to measure. If None, all qubits in the circuit.
    Returns:
         The list of circuits with rotation and measurement gates appended.
    """
    qubits = sorted(list(circuit.all_qubits())) if qubits is None else qubits
    rotated_circuits = []
    for pauli_string in pauli_strings:
        rotated_circuit = circuit.copy()
        for qubit, pauli in zip(qubits, pauli_string):
            # Pauli X measurement is equivalent to H plus a Z measurement
            if pauli == "X":
                rotated_circuit.append(cirq.H(qubit))
            # Pauli X measurement is equivalent to S^-1*H plus a Z measurement
            elif pauli == "Y":
                rotated_circuit.append(cirq.S(qubit) ** -1)
                rotated_circuit.append(cirq.H(qubit))
            # Pauli Z measurement
            else:
                assert (
                    pauli == "Z"
                ), f"Pauli must be X, Y, Z. Got {pauli} instead."
        if add_measurements:
            rotated_circuit.append(cirq.measure(*qubits))
        rotated_circuits.append(rotated_circuit)
    return rotated_circuits


def random_pauli_measurement(
    circuit: cirq.Circuit,
    n_total_measurements: int,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    qubits: list[cirq.Qid] | None = None,
) -> tuple[list[str], list[str]]:
    r"""This function performs random Pauli measurements on a given circuit and
    returns the outcomes. These outcomes are represented as a tuple of two
    lists of strings.

    Args:
        circuit: A Cirq circuit.
        n_total_measurements: The number of snapshots.
        executor: A callable that runs a circuit and returns a single
            bitstring.
        qubits: The qubits in the circuit to be measured. If None,
            all qubits in the circuit will be measured.

    Warning:
        The ``executor`` must return a ``MeasurementResult`` for a single shot,
        i.e., a single bitstring.

    Returns:
        Tuple containing two lists of strings, each of length equal to
        ``n_total_measurements``. Strings in the first list are sequences of
        0's and 1's, which represent qubit measurements outcomes in the
        computational basis (e.g. "01001"). Strings in the second list are
        sequences of Pauli-measurement performed on each qubit (e.g. "XZZYY").
    """

    qubits = sorted(list(circuit.all_qubits())) if qubits is None else qubits
    num_qubits = len(qubits)
    pauli_strings = generate_random_pauli_strings(
        num_qubits, n_total_measurements
    )

    # Rotate and attach measurement gates to the circuit
    rotated_circuits = get_rotated_circuits(
        circuit=circuit,
        pauli_strings=pauli_strings,
        add_measurements=True,
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
