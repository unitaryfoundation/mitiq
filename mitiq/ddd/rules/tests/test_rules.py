# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for DDD rules."""

import pytest
from cirq import CNOT, Circuit, I, LineQubit, X, Y, Z, bit_flip

from mitiq.ddd.rules.rules import general_rule, repeated_rule, xx, xyxy, yy
from mitiq.utils import _equal


@pytest.mark.parametrize(
    "slack_length",
    [4, 5, 8, 13, 21, 34],
)
@pytest.mark.parametrize(
    "gates",
    [
        [X, X],
        [X, Y, X, Y],
        [Y, Y],
        [X, Y, Z],
    ],
)
def test_general_sequences(slack_length, gates):
    sequence = general_rule(
        slack_length=slack_length,
        gates=gates,
    )
    gate_set = {X, Y, Z}
    seq_gates = [op.gate for op in sequence.all_operations()]
    assert len(sequence) == slack_length
    assert gates == [gate for gate in seq_gates if gate in gate_set]


@pytest.mark.parametrize(
    "slack_length",
    [5, 8, 13, 21, 34],
)
@pytest.mark.parametrize(
    "rule",
    [
        xx,
        xyxy,
        yy,
    ],
)
def test_built_in_sequences(rule, slack_length):
    name = rule.__name__
    sequence = rule(slack_length)
    gates = [X if i == "x" else Y for i in name]
    gate_set = {X, Y}
    seq_gates = [op.gate for op in sequence.all_operations()]
    assert len(sequence) == slack_length
    assert gates == [gate for gate in seq_gates if gate in gate_set]


@pytest.mark.parametrize(
    ("slack_length", "rule", "sequence"),
    [
        (
            5,
            xx,
            Circuit(
                [
                    X(LineQubit(0)) if i % 2 else I(LineQubit(0))
                    for i in range(5)
                ]
            ),
        ),
        (
            5,
            yy,
            Circuit(
                [
                    Y(LineQubit(0)) if i % 2 else I(LineQubit(0))
                    for i in range(5)
                ]
            ),
        ),
        (
            4,
            xyxy,
            Circuit(
                [
                    Y(LineQubit(0)) if i % 2 else X(LineQubit(0))
                    for i in range(4)
                ]
            ),
        ),
    ],
)
def test_exact_sequences(slack_length, rule, sequence):
    sequence_to_test = rule(slack_length)
    assert _equal(sequence_to_test, sequence)


@pytest.mark.parametrize(
    "slack_length",
    [1, 2, 3, 5, 8, 13, 21, 34],
)
@pytest.mark.parametrize("spacing", [i for i in range(5, 7)])
def test_rule_failures(slack_length, spacing):
    num_decoupling_gates = 3
    if slack_length < num_decoupling_gates:
        sequence = general_rule(
            slack_length=slack_length,
            spacing=spacing,
            gates=[X, Y, Z],
        )
        assert len(sequence) == 0
    elif slack_length < (
        (num_decoupling_gates + 1) * spacing + num_decoupling_gates
    ):
        sequence = general_rule(
            slack_length=slack_length,
            spacing=spacing,
            gates=[X, Y, Z],
        )
        assert len(sequence) == 0
    else:
        sequence = general_rule(
            slack_length=slack_length,
            spacing=spacing,
            gates=[X, Y, Z],
        )
        assert len(sequence) == slack_length


@pytest.mark.parametrize(
    "slack_length",
    [1, 2, 3, 5],
)
@pytest.mark.parametrize(
    "gates",
    [
        [X],
        [Y],
        [Z],
    ],
)
def test_general_for_incomplete_rule(slack_length, gates):
    with pytest.raises(ValueError, match="too short to make a ddd sequence"):
        general_rule(
            slack_length=slack_length,
            gates=gates,
        )


@pytest.mark.parametrize(
    "slack_length",
    [3, 5],
)
@pytest.mark.parametrize(
    "gates",
    [
        [CNOT, X, Y],
    ],
)
def test_general_for_multi_qubit_gate(slack_length, gates):
    with pytest.raises(ValueError, match="Wrong number of qubits"):
        general_rule(
            slack_length=slack_length,
            gates=gates,
        )


@pytest.mark.parametrize(
    "slack_length",
    [4, 5, 8, 13, 21, 34],
)
@pytest.mark.parametrize(
    "gates",
    [
        [X, X],
        [X, Y, X, Y],
        [Y, Y],
        [X, Y, Z],
    ],
)
def test_repeated_sequences(slack_length, gates):
    sequence = repeated_rule(
        slack_length=slack_length,
        gates=gates,
    )
    num_reps = slack_length // len(gates)
    gate_set = {X, Y, Z}
    seq_gates = [op.gate for op in sequence.all_operations()]
    assert len(sequence) == slack_length
    assert gates * num_reps == [gate for gate in seq_gates if gate in gate_set]


@pytest.mark.parametrize(
    "slack_length",
    [2, 3],
)
@pytest.mark.parametrize(
    "gates",
    [
        [X, Y, X, Y],
        [Y, Y, Y, Y],
    ],
)
def test_short_repeated_sequences(slack_length, gates):
    sequence = repeated_rule(
        slack_length=slack_length,
        gates=gates,
    )
    assert len(sequence) == 0


@pytest.mark.parametrize(
    "gates",
    [
        [bit_flip(p=0.1), bit_flip(p=0.1)],
        [X, X, X],
    ],
)
def test_not_unitary(gates):
    if bit_flip(p=0.1) in gates:
        with pytest.raises(TypeError, match="cirq.unitary failed"):
            general_rule(slack_length=17, gates=gates)
    else:
        with pytest.raises(
            ValueError, match="is not equivalent to the identity"
        ):
            general_rule(slack_length=17, gates=gates)
