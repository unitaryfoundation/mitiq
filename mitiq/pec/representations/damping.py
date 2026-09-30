# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Functions related to representations with amplitude damping noise."""

import copy
from itertools import product

import numpy as np
import numpy.typing as npt
from cirq import (
    AmplitudeDampingChannel,
    Circuit,
    Operation,
    Qid,
    ResetChannel,
    Z,
    kraus,
    reset,
)

from mitiq import QPROGRAM
from mitiq.interface.conversions import (
    CircuitConversionError,
    UnsupportedCircuitError,
    append_cirq_circuit_to_qprogram,
    convert_to_mitiq,
)
from mitiq.pec.types import NoisyOperation, OperationRepresentation
from mitiq.utils import arbitrary_tensor_product

_RESET_NOT_SUPPORTED_ERROR = (
    "The quasi-probability representation of amplitude damping noise needs a "
    "reset operation, which cannot be expressed in a {} circuit. Convert "
    "`ideal_operation` to a circuit type which supports reset (Cirq, Qiskit "
    "or OpenQASM) before calling this function."
)


def _append_reset_to_qprogram(
    ideal_operation: QPROGRAM, qubit: Qid, circuit_type: str
) -> QPROGRAM:
    """Returns ``ideal_operation`` with a reset of ``qubit`` appended to it.

    Frontends which do not support reset behave in two different ways: some
    fail the conversion, others drop the instruction and return a circuit
    which no longer implements what was asked for. Both cases are turned into
    an ``UnsupportedCircuitError`` here, since a silently dropped reset would
    give a representation which does not sum to the ideal operation.

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to append to.
        qubit: The qubit to reset.
        circuit_type: The frontend of ``ideal_operation``, used in the error
            message.

    Returns:
        The input operation followed by a reset, in the input circuit type.
    """
    try:
        circuit_with_reset = append_cirq_circuit_to_qprogram(
            ideal_operation, Circuit(reset(qubit))
        )
    except CircuitConversionError as error:
        raise UnsupportedCircuitError(
            _RESET_NOT_SUPPORTED_ERROR.format(circuit_type)
        ) from error

    converted_circuit, _ = convert_to_mitiq(circuit_with_reset)
    if not any(
        isinstance(operation.gate, ResetChannel)
        for operation in converted_circuit.all_operations()
    ):
        raise UnsupportedCircuitError(
            _RESET_NOT_SUPPORTED_ERROR.format(circuit_type)
        )

    return circuit_with_reset


def represent_operation_with_amplitude_damping_noise(
    ideal_operation: QPROGRAM,
    noise_level: float,
    is_qubit_dependent: bool = True,
) -> OperationRepresentation:
    r"""Returns the quasi-probability representation of the input
    single-qubit ``ideal_operation`` with respect to a basis of noisy
    operations.

    Any ideal single-qubit unitary followed by local amplitude-damping noise
    of equal ``noise_level`` is assumed to be in the basis of implementable
    operations.

    The representation is based on the analytical result presented in
    Theorem 3 of :cite:`Takagi_2020_PRR`. The basis is the noisy operation
    itself, the noisy operation followed by a ``Z`` gate and a reset of the
    qubit to :math:`|0\rangle`. Its one-norm is
    :math:`(1 + \epsilon) / (1 - \epsilon)` for a noise level
    :math:`\epsilon`, a cost which that work shows is achievable and bounds
    from below by :math:`(\sqrt{1 - \epsilon} + \epsilon / 2) / (1 -
    \epsilon)`.

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level of each amplitude damping channel.
        is_qubit_dependent: If True, the representation corresponds to the
            operation on the specific qubits defined in `ideal_operation`.
            If False, the representation is valid for the same gate even if
            acting on different qubits from those specified in
            `ideal_operation`.

    Returns:
        The quasi-probability representation of the ``ideal_operation``.

    Raises:
        ValueError: If ``ideal_operation`` acts on more than one qubit.
        UnsupportedCircuitError: If the frontend of ``ideal_operation``
            cannot express the reset which the representation needs. Cirq,
            Qiskit and OpenQASM circuits support it. Braket, PennyLane,
            pyQuil and Qibo circuits do not.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final amplitude damping channel, is
        physically implementable.

    .. note::
        The noisy operations of the returned representation have the same
        circuit type as ``ideal_operation``.
    """
    circuit_copy = copy.deepcopy(ideal_operation)
    converted_circ, circuit_type = convert_to_mitiq(circuit_copy)

    qubits = converted_circ.all_qubits()

    if len(qubits) != 1:
        raise ValueError("Only single-qubit operations are supported.")

    q = tuple(qubits)[0]

    eta_0 = (1 + np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
    eta_1 = (1 - np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
    eta_2 = -noise_level / (1 - noise_level)
    etas = [eta_0, eta_1, eta_2]
    post_ops: list[list[Operation]] = [[], [Z(q)]]

    # Basis of implementable operations as circuits
    imp_op_circuits = [
        append_cirq_circuit_to_qprogram(ideal_operation, Circuit(op))
        for op in post_ops
    ]
    imp_op_circuits.append(
        _append_reset_to_qprogram(ideal_operation, q, circuit_type)
    )
    noisy_operations = [NoisyOperation(c) for c in imp_op_circuits]

    return OperationRepresentation(
        ideal_operation, noisy_operations, etas, is_qubit_dependent
    )


def amplitude_damping_kraus(
    noise_level: float,
    num_qubits: int,
) -> list[npt.NDArray[np.complex64]]:
    """Returns the Kraus operators of the tensor product of local
    depolarizing channels acting on each qubit.
    """
    local_noisy_op = AmplitudeDampingChannel(noise_level)
    local_kraus = list(kraus(local_noisy_op))
    return [
        arbitrary_tensor_product(*kraus_string)
        for kraus_string in product(local_kraus, repeat=num_qubits)
    ]
