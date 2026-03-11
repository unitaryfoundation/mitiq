# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Function to generate representations with biased noise."""

import copy

from cirq import Circuit, Operation, X, Y, Z

from mitiq import QPROGRAM
from mitiq.interface.conversions import (
    append_cirq_circuit_to_qprogram,
    convert_to_mitiq,
)
from mitiq.pec import NoisyOperation, OperationRepresentation


def represent_operation_with_local_biased_noise(
    ideal_operation: QPROGRAM,
    epsilon: float,
    eta: float,
    is_qubit_dependent: bool = True,
) -> OperationRepresentation:
    r"""This function maps an
    ``ideal_operation`` :math:`\mathcal{U}` into its quasi-probability
    representation, which is a linear combination of noisy implementable
    operations :math:`\sum_\alpha \eta_{\alpha} \mathcal{O}_{\alpha}`.

    This function assumes a combined depolarizing and dephasing noise model
    with a bias factor :math:`\eta` (see :cite:`Strikis_2021_PRXQuantum`)
    and that the following noisy operations are implementable
    :math:`\mathcal{O}_{\alpha} = \mathcal{D} \circ \mathcal P_\alpha`
    where :math:`\mathcal{U}` is the unitary associated
    to the input ``ideal_operation``,
    :math:`\mathcal{P}_\alpha` is a Pauli operation and

    .. math::
        \mathcal{D}(\epsilon) = (1 - \epsilon)[\mathbb{1}] +
        \epsilon(\frac{\eta}{\eta + 1} \mathcal{Z}
        + \frac{1}{3}\frac{1}{\eta + 1}(\mathcal{X} + \mathcal{Y}
        + \mathcal{Z}))

    is the combined (biased) dephasing and depolarizing channel acting on a
    single qubit. For multi-qubit operations, we use a noise channel that is
    the tensor product of the local single-qubit channels.

    Args:
        ideal_operation: The ideal operation (as a QPROGRAM) to represent.
        epsilon: The local noise severity (as a float) of the combined channel.
        eta: The noise bias between combined dephasing and depolarizing
            channels with :math:`\eta = 0` describing a fully depolarizing
            channel and :math:`\eta = \infty` describing a fully dephasing
            channel.
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
        it remains a good approximation for small values of ``epsilon``.

    .. note::
        The input ``ideal_operation`` is typically a QPROGRAM with a single
        gate but could also correspond to a sequence of more gates.
        This is possible as long as the unitary associated to the input
        QPROGRAM, followed by a single final biased noise channel, is
        physically implementable.
    """
    circuit_copy = copy.deepcopy(ideal_operation)
    converted_circ, _ = convert_to_mitiq(circuit_copy)
    post_ops: list[list[Operation]]
    qubits = converted_circ.all_qubits()

    # Calculate coefficients in basis expansion in terms of eta and epsilon
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))
    alpha = (a**2 + a * b - 2 * c**2) / (
        a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2
    )
    beta = (-a * b - b**2 + 2 * c**2) / (
        a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2
    )
    gamma = -c / (a**2 + 2 * a * b + b**2 - 4 * c**2)
    if len(qubits) == 1:
        q = tuple(qubits)[0]
        alphas = [alpha] + 2 * [gamma] + [beta]
        post_ops = [[]]  # for eta_1, we do nothing, rather than I
        post_ops += [[P(q)] for P in [X, Y, Z]]  # 1Q Paulis

        # The two-qubit case: linear combination of 2Q Paulis
    elif len(qubits) == 2:
        q0, q1 = qubits

        alphas = (
            [alpha**2]
            + 2 * [alpha * gamma]
            + [alpha * beta]
            + 2 * [alpha * gamma]
            + [alpha * beta]
            + 2 * [gamma**2]
            + [beta * gamma]
            + 2 * [gamma**2]
            + 3 * [beta * gamma]
            + [beta**2]
        )
        post_ops = [[]]  # for eta_1, we do nothing, rather than I x I
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
    # Basis of implementable operations as circuits.
    imp_op_circuits = [
        append_cirq_circuit_to_qprogram(ideal_operation, Circuit(op))
        for op in post_ops
    ]

    # Build basis expansion.
    noisy_operations = [NoisyOperation(c) for c in imp_op_circuits]
    return OperationRepresentation(
        ideal_operation, noisy_operations, alphas, is_qubit_dependent
    )
