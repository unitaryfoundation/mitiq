# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for scaling noise by inserting identity layers."""

import pytest
from cirq import Circuit, LineQubit, ops

from mitiq.utils import _equal
from mitiq.zne.scaling.identity_insertion import (
    UnscalableCircuitError,
    _calculate_id_layers,
    insert_id_layers,
)


@pytest.mark.parametrize("scale_factor", (1, 2, 3, 4, 5, 6))
def test_id_layers_whole_scale_factor(scale_factor):
    """Tests if n-1 identity layers are inserted uniformly when
    the intended scale factor is n."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    scaled_circ = insert_id_layers(circ, scale_factor=scale_factor)
    num_layers = scale_factor - 1
    expected_circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.I.on_each(*qreg)] * num_layers,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.I.on_each(*qreg)] * num_layers,
        [ops.TOFFOLI.on(*qreg)],
        [ops.I.on_each(*qreg)] * num_layers,
    )
    assert _equal(scaled_circ, expected_circ)


def test_scale_with_intermediate_measurements_raises_error():
    """Tests scaling function raise an error on circuits with
    intermediate measurements.
    """
    qbit = LineQubit(0)
    circ = Circuit([ops.H.on(qbit)], [ops.measure(qbit)], [ops.T.on(qbit)])
    with pytest.raises(
        UnscalableCircuitError,
        match="Circuit contains intermediate measurements",
    ):
        insert_id_layers(circ, scale_factor=3.0)


def test_scaling_with_terminal_measurement():
    """Checks if the circuit with a terminal measurement is
    scaled with identity layers as expected.
    """
    qbit = LineQubit(0)
    input_circ = Circuit(
        [ops.H.on(qbit)], [ops.T.on(qbit)], [ops.measure(qbit)]
    )
    scaled_circ = insert_id_layers(input_circ, scale_factor=3.0)
    expected_circ = Circuit(
        [ops.H.on(qbit)],
        [ops.I.on(qbit)] * 2,
        [ops.T.on(qbit)],
        [ops.I.on(qbit)] * 2,
        [ops.measure(qbit)],
    )
    assert _equal(scaled_circ, expected_circ)


def test_calculate_id_layers_diff_scale_factor():
    """Checks if the partial layers to be inserted are 0 for a full
    scale factor and may or may not be 0 otherwise.

    Also checks if an error is raised for an invalid scale factor
    value.
    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    circ_depth = len(circ)
    full_scale_factor = 3
    id_layers_full_scale = _calculate_id_layers(circ_depth, full_scale_factor)
    num_partial_layers_full_scale_factor = id_layers_full_scale[-1]
    assert num_partial_layers_full_scale_factor == 0

    float_scale_factor_list = [1.3, 2.6, 3.77, 4.8, 5.9]
    for i in float_scale_factor_list:
        float_scale_factor = i
        id_layers_float_scale = _calculate_id_layers(
            circ_depth, float_scale_factor
        )

        num_partial_layers_float_scale_factor = id_layers_float_scale[-1]
        assert num_partial_layers_float_scale_factor >= 0

    bad_scale_factor_list = [-1.3, 0, -3, 0.76]
    for i in bad_scale_factor_list:
        with pytest.raises(ValueError, match="Requires scale_factor >= 1"):
            _calculate_id_layers(circ_depth, i)
            insert_id_layers(circ, i)


@pytest.mark.parametrize(
    "intended_scale_factor",
    (1, 1.1, 1.3, 1.7, 1.9, 2, 3.1, 3.6, 3.9, 3, 4, 5, 6),
)
def test_compare_scale_factor(intended_scale_factor):
    """tests if the intended scale factor is approximately close to the
    actual scale factor.
    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.X.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    scaled = insert_id_layers(circ, intended_scale_factor)
    achieved_scale_factor = len(scaled) / len(circ)
    assert achieved_scale_factor <= intended_scale_factor
    assert achieved_scale_factor >= intended_scale_factor - 1
