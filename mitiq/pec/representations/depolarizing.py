# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Functions related to representations with depolarizing noise."""

import copy
from itertools import product

import numpy as np
import numpy.typing as npt
from cirq import (
    Circuit,
    DepolarizingChannel,
    Operation,
    X,
    Y,
    Z,
    is_measurement,
    kraus,
)

from mitiq import QPROGRAM
from mitiq.interface.conversions import (
    append_cirq_circuit_to_qprogram,
    convert_to_mitiq,
)
from mitiq.pec.types import NoisyOperation, OperationRepresentation
from mitiq.utils import arbitrary_tensor_product


def represent_operation_with_global_depolarizing_noise(
    ideal_operation: QPROGRAM,
    noise_level: float,
    is_qubit_dependent: bool = True,
) -> OperationRepresentation:
    r"""As described in :cite:`Temme_2017_PRL`, this function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its quasi-probability
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a depolarizing noise model and, more precicely,
    that the following noisy operations are implementable
    :math:`\mathcal{O}_{\alpha} = \mathcal{D} \circ \mathcal P_\alpha
    \circ \mathcal{U}`, where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    :math:`\mathcal{D}(\rho) = (1 - \epsilon) \rho + \epsilon I/2^k` is a
    depolarizing channel (:math:`\epsilon` is a simple function of
    ``noise_level``).

    For a single-qubit ``ideal_operation``, the representation is as
    follows:

    .. math::
         \mathcal{U}_{\beta} = \eta_1 \mathcal{O}_1 + \eta_2 \mathcal{O}_2 +
                               \eta_3 \mathcal{O}_3 + \eta_4 \mathcal{O}_4

    .. math::
        \eta_1 =1 + \frac{3}{4} \frac{\epsilon}{1- \epsilon},
        \qquad \mathcal{O}_1 = \mathcal{D} \circ \mathcal{I} \circ \mathcal{U}

        \eta_2 =- \frac{1}{4}\frac{\epsilon}{1- \epsilon} , \qquad
        \mathcal{O}_2 = \mathcal{D} \circ \mathcal{X} \circ \mathcal{U}

        \eta_3 =- \frac{1}{4}\frac{\epsilon}{1- \epsilon} , \qquad
        \mathcal{O}_3 = \mathcal{D} \circ \mathcal{Y} \circ \mathcal{U}

        \eta_4 =- \frac{1}{4}\frac{\epsilon}{1- \epsilon} , \qquad
        \mathcal{O}_4 = \mathcal{D} \circ \mathcal{Z} \circ \mathcal{U}

    It was proven in :cite:`Takagi_2020_PRR` that, under suitable assumptions,
    this representation is optimal (minimum 1-norm).

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level (as a float) of the depolarizing channel.
        is_qubit_dependent: If True, the representation corresponds to the
            operation on the specific qubits defined in `ideal_operation`.
            If False, the representation is valid for the same gate even if
            acting on different qubits from those specified in
            `ideal_operation`.

    Returns:
        The quasi-probability representation of the ``ideal_operation``.

    .. note::
        This representation is based on the ideal assumption that one
        can append Pauli gates to a noisy operation without introducing
        additional noise. For a backend which violates this assumption,
        it remains a good approximation for small values of ``noise_level``.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final depolarizing channel, is
        physically implementable.
    """
    circuit_copy = copy.deepcopy(ideal_operation)
    converted_circ, _ = convert_to_mitiq(circuit_copy)
    post_ops: list[list[Operation]]
    qubits = converted_circ.all_qubits()

    # The single-qubit case: linear combination of 1Q Paulis
    if len(qubits) == 1:
        q = tuple(qubits)[0]

        epsilon = 4 / 3 * noise_level
        alpha_pos = 1 + ((3 / 4) * epsilon / (1 - epsilon))
        alpha_neg = -(1 / 4) * epsilon / (1 - epsilon)

        alphas = [alpha_pos] + 3 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I
        post_ops += [[P(q)] for P in [X, Y, Z]]  # 1Q Paulis

    # The two-qubit case: linear combination of 2Q Paulis
    elif len(qubits) == 2:
        q0, q1 = qubits

        epsilon = 16 / 15 * noise_level
        alpha_pos = 1 + ((15 / 16) * epsilon / (1 - epsilon))
        alpha_neg = -(1 / 16) * epsilon / (1 - epsilon)

        alphas = [alpha_pos] + 15 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I x I
        post_ops += [[P(q0)] for P in [X, Y, Z]]  # 1Q Paulis for q0
        post_ops += [[P(q1)] for P in [X, Y, Z]]  # 1Q Paulis for q1
        post_ops += [
            [Pi(q0), Pj(q1)] for Pi in [X, Y, Z] for Pj in [X, Y, Z]
        ]  # 2Q Paulis

    else:
        raise ValueError(
            "Can only represent single- and two-qubit gates."
            "Consider pre-compiling your circuit."
        )

    # Basis of implementable operations as circuits

    imp_op_circuits = [
        append_cirq_circuit_to_qprogram(
            ideal_operation,
            Circuit(op),
        )
        for op in post_ops
    ]

    noisy_operations = [NoisyOperation(c) for c in imp_op_circuits]

    return OperationRepresentation(
        ideal_operation, noisy_operations, alphas, is_qubit_dependent
    )


def represent_operation_with_local_depolarizing_noise(
    ideal_operation: QPROGRAM,
    noise_level: float,
    is_qubit_dependent: bool = True,
) -> OperationRepresentation:
    r"""As described in :cite:`Temme_2017_PRL`, this function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its quasi-probability
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a (local) single-qubit depolarizing noise model even
    for multi-qubit operations. More precicely, it assumes that the following
    noisy operations are implementable :math:`\mathcal{O}_{\alpha} =
    \mathcal{D}^{\otimes k} \circ \mathcal P_\alpha \circ \mathcal{U}`,
    where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    :math:`\mathcal{D}(\rho) = (1 - \epsilon) \rho + \epsilon I/2` is a
    single-qubit depolarizing channel (:math:`\epsilon` is a simple function
    of ``noise_level``).

    More information about the quasi-probability representation for a
    depolarizing noise channel can be found in:
    :func:`represent_operation_with_global_depolarizing_noise`.

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level of each depolarizing channel.
        is_qubit_dependent: If True, the representation corresponds to the
            operation on the specific qubits defined in `ideal_operation`.
            If False, the representation is valid for the same gate even
            if acting on different qubits from those specified in
            `ideal_operation`.
    Returns:
        The quasi-probability representation of the ``ideal_operation``.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final depolarizing channel, is
        physically implementable.
    """
    circuit_copy = copy.deepcopy(ideal_operation)
    converted_circ, _ = convert_to_mitiq(circuit_copy)

    qubits = converted_circ.all_qubits()

    if len(qubits) == 1:
        return represent_operation_with_global_depolarizing_noise(
            ideal_operation,
            noise_level,
        )

    # The two-qubit case: tensor product of two depolarizing channels.
    elif len(qubits) == 2:
        q0, q1 = qubits

        # Single-qubit representation coefficients.
        epsilon = noise_level * 4 / 3
        c_neg = -(1 / 4) * epsilon / (1 - epsilon)
        c_pos = 1 - 3 * c_neg

        imp_op_circuits = []
        alphas = []

        # The zero-pauli term in the linear combination
        imp_op_circuits.append(converted_circ)
        alphas.append(c_pos * c_pos)

        # The single-pauli terms in the linear combination
        for qubit in qubits:
            for pauli in [X, Y, Z]:
                imp_op_circuits.append(
                    append_cirq_circuit_to_qprogram(
                        ideal_operation, Circuit(pauli(qubit))
                    )
                )
                alphas.append(c_neg * c_pos)

        # The two-pauli terms in the linear combination
        for pauli_0, pauli_1 in product([X, Y, Z], repeat=2):
            imp_op_circuits.append(
                append_cirq_circuit_to_qprogram(
                    ideal_operation,
                    Circuit(pauli_0(q0), pauli_1(q1)),
                )
            )
            alphas.append(c_neg * c_neg)

    else:
        raise ValueError(
            "Can only represent single- and two-qubit gates."
            "Consider pre-compiling your circuit."
        )

    noisy_operations = [NoisyOperation(c) for c in imp_op_circuits]

    return OperationRepresentation(
        ideal_operation, noisy_operations, alphas, is_qubit_dependent
    )


def represent_operations_in_circuit_with_global_depolarizing_noise(
    ideal_circuit: QPROGRAM, noise_level: float
) -> list[OperationRepresentation]:
    """Iterates over all unique operations of the input ``ideal_circuit`` and,
    for each of them, generates the corresponding quasi-probability
    representation (linear combination of implementable noisy operations).

    This function assumes that the same depolarizing noise channel of strength
    ``noise_level`` affects each implemented operation.

    This function internally calls
    :func:`represent_operation_with_global_depolarizing_noise` (more details
    about the quasi-probability representation can be found in its docstring).

    Args:
        ideal_circuit: The ideal circuit, whose ideal operations should be
            represented.
        noise_level: The (gate-independent) depolarizing noise level.

    Returns:
        The list of quasi-probability representations associated to
        the operations of the input ``ideal_circuit``.

    .. note::
        Measurement gates are ignored (not represented).

    .. note::
        The returned representations are always defined in terms of
        Cirq circuits, even if the input is not a ``cirq.Circuit``.
    """

    circ, _ = convert_to_mitiq(ideal_circuit)

    representations = []
    for op in set(circ.all_operations()):
        if is_measurement(op):
            continue
        representations.append(
            represent_operation_with_global_depolarizing_noise(
                Circuit(op),
                noise_level,
            )
        )
    return representations


def represent_operations_in_circuit_with_local_depolarizing_noise(
    ideal_circuit: QPROGRAM, noise_level: float
) -> list[OperationRepresentation]:
    """Iterates over all unique operations of the input ``ideal_circuit`` and,
    for each of them, generates the corresponding quasi-probability
    representation (linear combination of implementable noisy operations).

    This function assumes that the tensor product of ``k`` single-qubit
    depolarizing channels affects each implemented operation, where
    ``k`` is the number of qubits associated to the operation.

    This function internally calls
    :func:`represent_operation_with_local_depolarizing_noise` (more details
    about the quasi-probability representation can be found in its docstring).

    Args:
        ideal_circuit: The ideal circuit, whose ideal operations should be
            represented.
        noise_level: The (gate-independent) depolarizing noise level.

    Returns:
        The list of quasi-probability representations associated to
        the operations of the input ``ideal_circuit``.

    .. note::
        Measurement gates are ignored (not represented).

    .. note::
        The returned representations are always defined in terms of
        Cirq circuits, even if the input is not a ``cirq.Circuit``.
    """
    circ, _ = convert_to_mitiq(ideal_circuit)

    representations = []
    for op in set(circ.all_operations()):
        if is_measurement(op):
            continue
        representations.append(
            represent_operation_with_local_depolarizing_noise(
                Circuit(op),
                noise_level,
            )
        )
    return representations


def global_depolarizing_kraus(
    noise_level: float,
    num_qubits: int,
) -> list[npt.NDArray[np.complex64]]:
    """Returns the kraus operators of a global depolarizing channel at a
    given noise level.
    """
    noisy_op = DepolarizingChannel(noise_level, num_qubits)
    return list(kraus(noisy_op))


def local_depolarizing_kraus(
    noise_level: float,
    num_qubits: int,
) -> list[npt.NDArray[np.complex64]]:
    """Returns the kraus operators of the tensor product of local
    depolarizing channels acting on each qubit.
    """
    local_kraus = global_depolarizing_kraus(noise_level, num_qubits=1)
    return [
        arbitrary_tensor_product(*kraus_string)
        for kraus_string in product(local_kraus, repeat=num_qubits)
    ]
