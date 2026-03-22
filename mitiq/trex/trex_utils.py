# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for TREX (Twirled Readout Error eXtinction)."""

from collections.abc import Sequence

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import MeasurementResult


def insert_x_before_first_measurement(
    circuit: cirq.Circuit,
    bitstring: npt.NDArray[np.int64],
    qubits: Sequence[cirq.Qid],
) -> cirq.Circuit:
    """Insert X gates before the first measurement moment based on a
    random bitstring.

    For each qubit where ``bitstring[i] == 1``, an X gate is inserted
    immediately before the first moment containing a measurement gate.
    This is designed for circuits with a single terminal measurement
    (as produced by ``PauliStringCollection.measure_in``).

    Args:
        circuit: A Cirq circuit containing measurement gates.
        bitstring: A binary array of length ``len(qubits)``. An X gate is
            inserted before measurement on qubit ``qubits[i]`` if
            ``bitstring[i] == 1``.
        qubits: The qubits corresponding to each bit in ``bitstring``.

    Returns:
        A new circuit with X gates inserted before measurements.
    """
    if len(bitstring) != len(qubits):
        raise ValueError(
            f"Length of bitstring ({len(bitstring)}) must match "
            f"number of qubits ({len(qubits)})."
        )

    x_ops = [
        cirq.X(qubit) for qubit, bit in zip(qubits, bitstring) if bit == 1
    ]

    if not x_ops:
        return circuit.copy()

    new_moments = []
    x_inserted = False
    for moment in circuit:
        has_measurement = any(
            cirq.is_measurement(op) for op in moment.operations
        )
        if has_measurement and not x_inserted:
            new_moments.append(cirq.Moment(x_ops))
            x_inserted = True
        new_moments.append(moment)

    if not x_inserted:
        new_moments.append(cirq.Moment(x_ops))

    return cirq.Circuit(new_moments)


def create_calibration_circuits(
    qubits: Sequence[cirq.Qid],
    randomization_strings: npt.NDArray[np.int64],
) -> list[cirq.Circuit]:
    """Create calibration circuits for all randomization strings.

    Args:
        qubits: The qubits to include in the calibration circuits.
        randomization_strings: 2D array of shape
            ``(num_randomizations, n_qubits)``.

    Returns:
        A list of calibration circuits, one per randomization string.
    """
    return [
        create_calibration_circuit(qubits, s) for s in randomization_strings
    ]


def create_calibration_circuit(
    qubits: Sequence[cirq.Qid],
    bitstring: npt.NDArray[np.int64],
) -> cirq.Circuit:
    """Create a calibration circuit for TREX.

    The calibration circuit prepares the ``|0...0>`` state, applies X gates
    according to ``bitstring``, and measures all qubits. After classical
    post-processing (XOR with ``bitstring``), the ideal result is all zeros.
    Deviations indicate readout errors.

    Args:
        qubits: The qubits to include in the calibration circuit.
        bitstring: A binary array of length ``len(qubits)``. An X gate is
            applied on qubit ``qubits[i]`` if ``bitstring[i] == 1``.

    Returns:
        A calibration circuit with X gates and measurements.
    """
    if len(bitstring) != len(qubits):
        raise ValueError(
            f"Length of bitstring ({len(bitstring)}) must match "
            f"number of qubits ({len(qubits)})."
        )

    sorted_qubits = sorted(qubits)
    ops = [
        cirq.X(qubit)
        for qubit, bit in zip(sorted_qubits, bitstring)
        if bit == 1
    ]

    circuit = cirq.Circuit()
    if ops:
        circuit.append(cirq.Moment(ops))
    circuit.append(cirq.measure(*sorted_qubits))
    return circuit


def xor_bitstrings(
    result: MeasurementResult,
    bitstring: npt.NDArray[np.int64],
) -> MeasurementResult:
    """Apply classical bit flip (XOR) to measurement results.

    This undoes the effect of X gates applied before measurement in
    the TREX protocol.

    Args:
        result: The measurement result to flip.
        bitstring: A binary array of length ``result.nqubits``. Bit ``i``
            of each measured bitstring is flipped if ``bitstring[i] == 1``.

    Returns:
        A new ``MeasurementResult`` with bits XOR'd by ``bitstring``.
    """
    if len(bitstring) != result.nqubits:
        raise ValueError(
            f"Length of bitstring ({len(bitstring)}) must match "
            f"number of qubits ({result.nqubits})."
        )

    flipped = result.asarray ^ bitstring
    return MeasurementResult(
        result=flipped.tolist(),
        qubit_indices=result.qubit_indices,
    )
