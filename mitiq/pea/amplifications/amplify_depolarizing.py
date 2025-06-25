# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions related to amplifications with depolarizing noise."""

import copy
from itertools import product

from cirq import Circuit, Operation, X, Y, Z, is_measurement

from mitiq import QPROGRAM
from mitiq.interface.conversions import (
    accept_any_qprogram_as_input,
    append_cirq_circuit_to_qprogram,
    convert_to_mitiq,
)
from mitiq.pec.types import NoisyOperation, OperationRepresentation


@accept_any_qprogram_as_input
def amplify_noisy_op_with_global_depolarizing_noise(
    ideal_operation: QPROGRAM,
    noise_level: float,
    is_qubit_dependent: bool = True,
) -> OperationRepresentation:
    r"""As described in :cite:`Kim_2023_Nature`, this function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its noise-amplified
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a depolarizing noise model and, more precisely,
    that the following noisy operations are implementable
    :math:`\mathcal{O}_{\alpha} = \mathcal{D} \circ \mathcal P_\alpha
    \circ \mathcal{U}`, where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    :math:`\mathcal{D}(\rho) = (1 - \epsilon) \rho + \epsilon I/2^k` is a
    depolarizing channel (:math:`\epsilon` is a simple function of
    ``noise_level``).

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level (as a float) of the depolarizing channel.
        is_qubit_dependent: If True, the representation corresponds to the
            operation on the specific qubits defined in `ideal_operation`.
            If False, the representation is valid for the same gate even if
            acting on different qubits from those specified in
            `ideal_operation`.

    Returns:
        The noise-amplified representation of the ``ideal_operation``.
    """
    circuit_copy = copy.deepcopy(ideal_operation)
    converted_circ, _ = convert_to_mitiq(circuit_copy)
    post_ops: list[list[Operation]]
    qubits = converted_circ.all_qubits()

    # The single-qubit case: linear combination of Paulis on one qubit
    if len(qubits) == 1:
        q = tuple(qubits)[0]

        alpha_pos = 1.0 - noise_level
        alpha_neg = noise_level / 3

        alphas = [alpha_pos] + 3 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I
        post_ops += [[P(q)] for P in [X, Y, Z]]  # Paulis on one qubit

    # The two-qubit case: linear combination of Paulis on each qubit
    elif len(qubits) == 2:
        q0, q1 = qubits

        alpha_pos = 1.0 - noise_level
        alpha_neg = noise_level / 15

        alphas = [alpha_pos] + 15 * [alpha_neg]
        post_ops = [[]]  # for alpha_pos, we do nothing, rather than I x I
        post_ops += [[P(q0)] for P in [X, Y, Z]]  # Paulis on q0
        post_ops += [[P(q1)] for P in [X, Y, Z]]  # Paulis on q1
        post_ops += [
            [Pi(q0), Pj(q1)] for Pi in [X, Y, Z] for Pj in [X, Y, Z]
        ]  # Paulis on q0 and q1

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


@accept_any_qprogram_as_input
def amplify_noisy_op_with_local_depolarizing_noise(
    ideal_operation: QPROGRAM,
    noise_level: float,
    is_qubit_dependent: bool = True,
) -> OperationRepresentation:
    r"""As described in :cite:`Kim_2023_Nature`, this function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its noise-amplified
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a (local) single-qubit depolarizing noise model even
    for multi-qubit operations. More precisely, it assumes that the following
    noisy operations are implementable :math:`\mathcal{O}_{\alpha} =
    \mathcal{D}^{\otimes k} \circ \mathcal P_\alpha \circ \mathcal{U}`,
    where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation`` acting on :math:`k` qubits,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and
    :math:`\mathcal{D}(\rho) = (1 - \epsilon) \rho + \epsilon I/2` is a
    single-qubit depolarizing channel (:math:`\epsilon` is a simple function
    of ``noise_level``).

    More information about the noise-amplified representation for a
    depolarizing noise channel can be found in:
    :func:`amplify_operation_with_global_depolarizing_noise`.

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        noise_level: The noise level of each depolarizing channel.
        is_qubit_dependent: If True, the representation corresponds to the
            operation on the specific qubits defined in `ideal_operation`.
            If False, the representation is valid for the same gate even
            if acting on different qubits from those specified in
            `ideal_operation`.
    Returns:
        The noise-amplified representation of the ``ideal_operation``.

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
        return amplify_noisy_op_with_global_depolarizing_noise(
            ideal_operation,
            noise_level,
        )

    # The two-qubit case: tensor product of two depolarizing channels.
    elif len(qubits) == 2:
        q0, q1 = qubits

        # Single-qubit amplification coefficients.
        c_neg = noise_level / 3
        c_pos = 1.0 - noise_level

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


def amplify_noisy_ops_in_circuit_with_global_depolarizing_noise(
    ideal_circuit: QPROGRAM, noise_level: float
) -> list[OperationRepresentation]:
    """Iterates over all unique operations of the input ``ideal_circuit`` and,
    for each of them, generates the corresponding noise-amplified
    representation (linear combination of implementable noisy operations).

    This function assumes that the same depolarizing noise channel of strength
    ``noise_level`` affects each implemented operation.

    Args:
        ideal_circuit: The ideal circuit, whose ideal operations should be
            represented.
        noise_level: The (gate-independent) depolarizing noise level.

    Returns:
        The list of quasi-probability amplifications associated to
        the operations of the input ``ideal_circuit``.

    .. note::
        Measurement gates are ignored.

    .. note::
        The returned amplifications are always defined in terms of
        Cirq circuits, even if the input is not a ``cirq.Circuit``.
    """

    circ, _ = convert_to_mitiq(ideal_circuit)

    amplifications = []
    for op in set(circ.all_operations()):
        if is_measurement(op):
            continue
        amplifications.append(
            amplify_noisy_op_with_global_depolarizing_noise(
                Circuit(op),
                noise_level,
            )
        )
    return amplifications


def amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
    ideal_circuit: QPROGRAM, noise_level: float
) -> list[OperationRepresentation]:
    """Iterates over all unique operations of the input ``ideal_circuit`` and,
    for each of them, generates the corresponding quasi-probability
    amplification (linear combination of implementable noisy operations).

    This function assumes that the tensor product of ``k`` single-qubit
    depolarizing channels affects each implemented operation, where
    ``k`` is the number of qubits associated to the operation.


    Args:
        ideal_circuit: The ideal circuit, whose ideal operations should be
            represented.
        noise_level: The (gate-independent) depolarizing noise level.

    Returns:
        The list of quasi-probability amplifications associated to
        the operations of the input ``ideal_circuit``.

    .. note::
        Measurement gates are ignored.

    .. warning::
        The returned amplifications are always defined in terms of
        Cirq circuits, even if the input is not a ``cirq.Circuit``.
    """
    circ, _ = convert_to_mitiq(ideal_circuit)

    amplifications = []
    for op in set(circ.all_operations()):
        if is_measurement(op):
            continue
        amplifications.append(
            amplify_noisy_op_with_local_depolarizing_noise(
                Circuit(op),
                noise_level,
            )
        )
    return amplifications
