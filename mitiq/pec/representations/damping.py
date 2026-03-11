# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Functions related to representations with amplitude damping noise."""

from itertools import product

import numpy as np
import numpy.typing as npt
from cirq import AmplitudeDampingChannel, Circuit, Z, kraus, reset

from mitiq.pec.types import NoisyOperation, OperationRepresentation
from mitiq.utils import arbitrary_tensor_product


# TODO: this may be extended to an arbitrary QPROGRAM (GitHub issue gh-702).
def _represent_operation_with_amplitude_damping_noise(
    ideal_operation: Circuit,
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

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final amplitude damping channel, is
        physically implementable.

    .. note::
        The input ``ideal_operation`` must be a ``cirq.Circuit``.
    """

    if not isinstance(ideal_operation, Circuit):
        raise NotImplementedError(
            "The input ideal_operation must be a cirq.Circuit.",
        )

    qubits = ideal_operation.all_qubits()

    if len(qubits) == 1:
        q = tuple(qubits)[0]

        eta_0 = (1 + np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
        eta_1 = (1 - np.sqrt(1 - noise_level)) / (2 * (1 - noise_level))
        eta_2 = -noise_level / (1 - noise_level)
        etas = [eta_0, eta_1, eta_2]
        post_ops = [[], Z(q), reset(q)]

    else:
        raise ValueError(  # pragma: no cover
            "Only single-qubit operations are supported."  # pragma: no cover
        )  # pragma: no cover

    # Basis of implementable operations as circuits
    imp_op_circuits = [ideal_operation + Circuit(op) for op in post_ops]
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
