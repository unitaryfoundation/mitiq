# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Functions related to representations with amplitude damping noise."""

from itertools import product

import numpy as np
import numpy.typing as npt
from cirq import (
    AmplitudeDampingChannel,
    Circuit,
    ResetChannel,
    Z,
    kraus,
    reset,
)

from mitiq import QPROGRAM
from mitiq.interface.conversions import (
    CircuitConversionError,
    append_cirq_circuit_to_qprogram,
    convert_to_mitiq,
)
from mitiq.pec.types import NoisyOperation, OperationRepresentation
from mitiq.utils import arbitrary_tensor_product


# Partial support for arbitrary QPROGRAM inputs (GitHub issue gh-702):
# extension to the remaining frontends is blocked on their conversion
# paths carrying the non-unitary reset operation — a converter-level gap
# for pyQuil and PennyLane, a language-level gap for Braket and Qibo
# (see the docstring note).
def _represent_operation_with_amplitude_damping_noise(
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

    The representation is based on the analytical result presented
    in :cite:`Takagi2020`.

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
        ValueError: If the input operation acts on more than one qubit.
        CircuitConversionError: If the frontend of ``ideal_operation``
            cannot faithfully represent the non-unitary reset operation
            contained in the basis of implementable operations.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final amplitude damping channel, is
        physically implementable.

    .. note::
        The basis of implementable operations contains a non-unitary reset
        operation, so this representation can only be returned for
        frontends whose Mitiq conversions preserve reset. These are
        currently Cirq and Qiskit circuits, as well as the OpenQASM 2.0
        strings returned by
        ``mitiq.interface.convert_from_mitiq(circuit, "openqasm")``.
        For the other frontends a ``CircuitConversionError`` is raised
        instead of returning a physically incorrect representation:
        the Cirq-to-pyQuil converter does not support
        ``cirq.ResetChannel`` (although pyQuil itself has a ``RESET``
        instruction), the Braket circuit model has no reset instruction,
        the Qibo OpenQASM parser rejects reset statements, and the
        PennyLane converter silently discards the reset.
    """
    converted_circ, input_circuit_type = convert_to_mitiq(ideal_operation)

    qubits = converted_circ.all_qubits()

    if len(qubits) == 1:
        q = tuple(qubits)[0]

        eta_0 = (1 + np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
        eta_1 = (1 - np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
        eta_2 = -noise_level / (1 - noise_level)
        etas = [eta_0, eta_1, eta_2]
        post_ops = [[], Z(q), reset(q)]

    else:
        raise ValueError("Only single-qubit operations are supported.")

    # Basis of implementable operations as circuits
    imp_op_circuits = [
        append_cirq_circuit_to_qprogram(
            ideal_operation,
            Circuit(op),
        )
        for op in post_ops
    ]
    noisy_operations = [NoisyOperation(c) for c in imp_op_circuits]

    # The last basis element ends with a non-unitary reset. Some frontend
    # conversions accept it but silently delete the reset, which would make
    # the returned representation wrong at any nonzero noise level. The
    # round trip through the native frontend must keep every reset and
    # keep the appended reset as the final operation.
    num_ideal_resets = sum(
        isinstance(op.gate, ResetChannel)
        for op in converted_circ.all_operations()
    )
    roundtrip_ops = list(noisy_operations[-1].circuit.all_operations())
    num_roundtrip_resets = sum(
        isinstance(op.gate, ResetChannel) for op in roundtrip_ops
    )
    if num_roundtrip_resets != num_ideal_resets + 1 or not isinstance(
        roundtrip_ops[-1].gate, ResetChannel
    ):
        raise CircuitConversionError(
            f"Conversion to the circuit type '{input_circuit_type}' does "
            "not preserve the non-unitary reset operation required by the "
            "amplitude damping representation."
        )

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
