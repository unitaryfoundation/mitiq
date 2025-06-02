# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Built-in rules determining what DDD sequence should be applied in a given
slack window.
"""

from itertools import cycle

import numpy as np
from cirq import (
    Circuit,
    Gate,
    I,
    LineQubit,
    X,
    Y,
    allclose_up_to_global_phase,
    unitary,
)


def general_rule(
    slack_length: int, gates: list[Gate], spacing: int = -1
) -> Circuit:
    """Returns a digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
        gates: A list of single qubit Cirq gates to build the rule. E.g. [X, X]
            is the xx sequence, [X, Y, X, Y] is the xyxy sequence.
            - Note: To repeat the sequence, specify a repeated gateset.
    Returns:
        A digital dynamical decoupling sequence, as a Cirq circuit.

    Example:
        When ``slack_length = 8`` and ``gates = [X, X]`` the spacing defaults
        to 2 and the rule returns the sequence::

            ──I──I──X──I──I──X──I──I──

        When ``slack_length = 9`` and ``gates [X, Y, X, Y]`` the spacing
        defaults to 1 and the rule returns the sequence::

            ──I──X──I──Y──I──X──I──Y──I──
    """
    if len(gates) < 2:
        raise ValueError("Gateset too short to make a ddd sequence.")
    if slack_length < 2 or slack_length < len(gates):
        return Circuit()
    num_decoupling_gates = len(gates)
    slack_difference = slack_length - num_decoupling_gates
    if spacing < 0:
        spacing = slack_difference // (num_decoupling_gates + 1)
    slack_remainder = slack_length - (
        spacing * (num_decoupling_gates + 1) + num_decoupling_gates
    )
    if slack_remainder < 0:
        return Circuit()
    q = LineQubit(0)
    slack_gates = [I(q) for _ in range(spacing)]
    sequence = Circuit(slack_gates)
    for gate in gates:
        sequence.append(gate.on(q))
        sequence.append(slack_gates)

    if not allclose_up_to_global_phase(np.eye(2), unitary(sequence)):
        raise ValueError("Sequence is not equivalent to the identity!")
    for i in range(slack_remainder):
        sequence.append(I(q)) if i % 2 else sequence.insert(0, I(q))
    return sequence


def xx(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns an XX digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
            E.g. given slack_length = 8 the spacing defaults to 2 and this
            rule returns the sequence:
            ──I──I──X──I──I──X──I──I──.
    Returns:
        An XX digital dynamical decoupling sequence, as a Cirq circuit.
    """
    xx_sequence = general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[X, X],
    )
    return xx_sequence


def xyxy(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns an XYXY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
            E.g. given slack_length = 9 the spacing defaults to 1 and this
            rule returns the sequence:
            ──I──X──I──Y──I──X──I──Y──I──.
    Returns:
        An XYXY digital dynamical decoupling sequence, as a Cirq circuit.
    """
    xyxy_sequence = general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[X, Y, X, Y],
    )
    return xyxy_sequence


def yy(slack_length: int, spacing: int = -1) -> Circuit:
    """Returns a YY digital dynamical decoupling sequence, based on inputs.

    Args:
        slack_length: Length of idle window to fill.
        spacing: How many identity spacing gates to apply between dynamical
            decoupling gates, as a non-negative int. Negative int corresponds
            to default. Defaults to maximal spacing that fits a single sequence
            in the given slack window.
            E.g. given slack_length = 8 the spacing defaults to 2 and
            this rule returns the sequence:
            ──I──I──Y──I──I──Y──I──I──.
    Returns:
        An YY digital dynamical decoupling sequence, as a Cirq circuit.
    """
    yy_sequence = general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=[Y, Y],
    )
    return yy_sequence


def repeated_rule(slack_length: int, gates: list[Gate]) -> Circuit:
    """Returns a general digital dynamical decoupling sequence that repeats
    until the slack is filled without spacing, up to a complete repetition.

    Args:
        slack_length: Length of idle window to fill.
        gates: A list of single qubit Cirq gates to build the rule. E.g. [X, X]
            is the xx sequence, [X, Y, X, Y] is the xyxy sequence.
    Returns:
        A repeated digital dynamical decoupling sequence, as a Cirq circuit.

    Warning:
        Where :func:`.general_rule()` fills a slack window with a single
        sequence, this rule attempts to fill every moment with sequence
        repetitions (up to a complete repetition of the gate set).
        E.g. given ``slack_length = 8`` and ``gates = [X, Y, X, Y]``, this
        rule returns the sequence::

            ──X──Y──X──Y──X──Y──X──Y──
    """
    num_decoupling_gates = len(gates)
    if num_decoupling_gates > slack_length:
        return Circuit()
    sequence = general_rule(
        slack_length=slack_length,
        spacing=0,
        gates=[
            gate
            for (_, gate) in zip(
                range(slack_length - (slack_length % num_decoupling_gates)),
                cycle(gates),
            )
        ],
    )
    return sequence
