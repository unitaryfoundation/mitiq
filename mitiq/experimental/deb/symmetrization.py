# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Circuit construction utilities for debiasing by Pauli symmetrization."""

from dataclasses import dataclass
from typing import cast

import cirq
import numpy as np

from mitiq import QPROGRAM
from mitiq.interface.conversions import convert_from_mitiq, convert_to_mitiq

_PAULI_GATES = {
    "I": cirq.I,
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
}


@dataclass(frozen=True)
class PauliSymmetrization:
    """Container with a symmetrized circuit and its sampled Pauli layer.

    Args:
        circuit: Circuit with matching Pauli layers inserted around the
            non-measurement body of the input circuit.
        paulis: One Pauli label per qubit, ordered by ``qubits``.
        qubits: Sorted qubits used to construct the Pauli layer.
    """

    circuit: cirq.Circuit
    paulis: tuple[str, ...]
    qubits: tuple[cirq.Qid, ...]


def construct_circuits(
    circuit: QPROGRAM,
    num_variants: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> list[QPROGRAM]:
    """Returns Pauli-symmetrized variants of ``circuit``.

    Each variant samples one Pauli from ``{I, X, Y, Z}`` independently per
    qubit, prepends that Pauli layer to the circuit body, and appends the same
    layer immediately before terminal measurements. This keeps measurement
    operations terminal while exposing the randomized variants used by the
    high-level debiasing helpers.

    Args:
        circuit: Input quantum circuit supported by Mitiq.
        num_variants: Number of symmetrized variants to generate.
        random_state: Optional NumPy seed or generator for reproducibility.

    Raises:
        ValueError: If ``num_variants`` is not positive.

    Returns:
        A list of circuits in the same frontend format as ``circuit``.
    """
    symmetrizations, input_type = _construct_cirq_symmetrizations(
        circuit, num_variants, random_state=random_state
    )
    return [
        convert_from_mitiq(symmetrization.circuit, input_type)
        for symmetrization in symmetrizations
    ]


def _construct_cirq_symmetrizations(
    circuit: QPROGRAM,
    num_variants: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> tuple[list[PauliSymmetrization], str]:
    """Returns Cirq variants plus sampled Pauli metadata."""
    if num_variants <= 0:
        raise ValueError("num_variants must be a positive integer.")

    rng = (
        np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator)
        else random_state
    )

    cirq_circuit, input_type = convert_to_mitiq(circuit)
    qubits = tuple(sorted(cirq_circuit.all_qubits()))
    if not qubits:
        return [
            PauliSymmetrization(cirq_circuit.copy(), tuple(), tuple())
            for _ in range(num_variants)
        ], input_type

    body, terminal_measurements = _split_terminal_measurements(cirq_circuit)
    symmetrizations = []
    pauli_choices = tuple(_PAULI_GATES)

    for _ in range(num_variants):
        paulis = tuple(cast(str, rng.choice(pauli_choices)) for _ in qubits)
        layer = _pauli_layer(paulis, qubits)
        sym_circuit = cirq.Circuit()
        sym_circuit += layer
        sym_circuit += body
        sym_circuit += layer
        sym_circuit += terminal_measurements
        symmetrizations.append(
            PauliSymmetrization(sym_circuit, paulis, qubits)
        )

    return symmetrizations, input_type


def _pauli_layer(
    paulis: tuple[str, ...], qubits: tuple[cirq.Qid, ...]
) -> cirq.Circuit:
    """Builds a Cirq circuit containing a single Pauli layer."""
    return cirq.Circuit(
        _PAULI_GATES[pauli](qubit)
        for pauli, qubit in zip(paulis, qubits)
        if pauli != "I"
    )


def _split_terminal_measurements(
    circuit: cirq.Circuit,
) -> tuple[cirq.Circuit, cirq.Circuit]:
    """Splits a circuit into non-measurement body and terminal measurements."""
    body = cirq.Circuit()
    measurements = cirq.Circuit()
    seen_terminal_measurement = False

    for moment in circuit:
        body_ops = []
        measurement_ops = []
        for op in moment:
            if cirq.is_measurement(op):
                seen_terminal_measurement = True
                measurement_ops.append(op)
            else:
                if seen_terminal_measurement:
                    raise ValueError(
                        "Debiasing currently supports only terminal "
                        "measurements."
                    )
                body_ops.append(op)

        if body_ops:
            body.append(body_ops)
        if measurement_ops:
            measurements.append(measurement_ops)

    return body, measurements


def _average_distributions(
    distributions: list[dict[str, float]],
) -> dict[str, float]:
    """Averages normalized bitstring distributions."""
    if not distributions:
        raise ValueError("At least one distribution is required.")

    averaged: dict[str, float] = {}
    for distribution in distributions:
        for bitstring, probability in distribution.items():
            averaged[bitstring] = averaged.get(bitstring, 0.0) + probability

    scale = float(len(distributions))
    return {
        bitstring: probability / scale
        for bitstring, probability in sorted(averaged.items())
    }


def _counts_to_distribution(
    counts: dict[str, int | float],
) -> dict[str, float]:
    """Normalizes a count dictionary into probabilities."""
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Counts must contain a positive total weight.")
    return {
        bitstring: float(count) / total for bitstring, count in counts.items()
    }


def _distribution_to_counts(
    distribution: dict[str, float], shots: int
) -> dict[str, int]:
    """Converts a distribution to deterministic rounded counts."""
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")

    raw_counts = {
        bitstring: int(np.floor(probability * shots))
        for bitstring, probability in distribution.items()
    }
    remainder = shots - sum(raw_counts.values())
    if remainder <= 0:
        return raw_counts

    fractions = sorted(
        (
            (probability * shots - raw_counts[bitstring], bitstring)
            for bitstring, probability in distribution.items()
        ),
        reverse=True,
    )
    for _, bitstring in fractions[:remainder]:
        raw_counts[bitstring] += 1
    return raw_counts
