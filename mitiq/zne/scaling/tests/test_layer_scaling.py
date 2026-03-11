# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scaling by layer."""

import random
from itertools import product
from unittest.mock import patch

import pytest
from cirq import Circuit, LineQubit, ops

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.zne.scaling import get_layer_folding, layer_folding


def test_layer_folding_with_measurements():
    # Test circuit
    # 0: ───H───M───────
    #
    # 1: ───H───@───M───
    #           │
    # 2: ───────X───M───
    q = LineQubit.range(3)
    circuit = Circuit(
        ops.H(q[0]),
        ops.H(q[1]),
        ops.CNOT(*q[1:]),
        ops.measure_each(*q),
    )
    folded_circuit = layer_folding(circuit, [1] * len(circuit))

    expected_folded_circuit = Circuit(
        [ops.H(q[0])] * 3,
        [ops.H(q[1])] * 3,
        [ops.CNOT(*q[1:])] * 3,
        ops.measure_each(*q),
    )
    assert folded_circuit == expected_folded_circuit


def test_layer_folding():
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───X───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )

    # Iterate over every possible combination of layerwise folds for a maximum
    # number of 5-folds.
    total_folds = 5
    for i1, i2, i3 in product(range(total_folds), repeat=3):
        folded_circuit = layer_folding(circ, [i1, i2, i3])

        # For a given layer, the number of copies on a layer will be
        # 2n + 1 where "n" is the number of folds to perform.
        a, b, c = LineQubit.range(3)
        correct = Circuit(
            # Layer-1
            [ops.H.on_each(*(a, b, c))] * (2 * i1 + 1),
            # Layer-2
            [ops.CNOT.on(a, b)] * (2 * i2 + 1),
            [ops.X.on(c)] * (2 * i2 + 1),
            # Layer-3
            [ops.TOFFOLI.on(*(a, b, c))] * (2 * i3 + 1),
        )
        assert folded_circuit == correct


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_layer_folding_all_qprograms(circuit_type):
    """This test only ensures proper depth of layer-folded non-cirq circuits
    as the mitiq conversion functions alter structure/gate composition."""
    qreg = LineQubit.range(3)
    circuit = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    num_layers = len(circuit)
    circuit = convert_from_mitiq(circuit, circuit_type)
    layers_to_fold = random.choices(range(5), k=num_layers)
    folded_circuit = layer_folding(circuit, layers_to_fold)
    folded_mitiq_circuit = convert_to_mitiq(folded_circuit)[0]
    num_ideal_layers = sum(2 * n + 1 for n in layers_to_fold)
    if circuit_type == "pyquil":
        # this block is needed for pyquil because of some quirks that pop up
        # when converting to and from pyquil that does not make exact equality.
        assert len(folded_mitiq_circuit) >= num_ideal_layers
    else:
        assert len(folded_mitiq_circuit) == num_ideal_layers


@patch("mitiq.zne.scaling.layer_scaling.layer_folding")
def test_get_layer_folding(mock_layer_folding):
    a, b = LineQubit.range(2)
    circuit = Circuit(ops.X(a), ops.CNOT(a, b), ops.Y(b))
    layer_index = 1
    scale_factor = 3

    folding_func = get_layer_folding(layer_index)
    folding_func(circuit, scale_factor)

    mock_layer_folding.assert_called_once_with(
        circuit, layers_to_fold=[0, 1, 0]
    )
