# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mirror quantum volume circuits."""

import cirq
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.mirror_qv_circuits import generate_mirror_qv_circuit


@pytest.mark.parametrize(
    "depth_num",
    [1, 2, 3, 4, 5, 6, 7, 8],
)
def test_generate_mirror_qv_circuit(depth_num):
    """Check the circuit output."""
    test_circ = generate_mirror_qv_circuit(4, depth_num)

    # check bitstring is all 0's
    bit_test = cirq.Simulator().run(
        test_circ + cirq.measure(test_circ.all_qubits()), repetitions=1000
    )
    test_bitstring = list(bit_test.measurements.values())[0][0].tolist()
    expected_bitstring = [0] * 4
    assert test_bitstring == expected_bitstring


def test_bad_depth_number():
    """Check if an unacceptable depth number rasies an error."""
    for n in (-1, 0):
        with pytest.raises(
            ValueError, match="{} is invalid for the generated circuit depth."
        ):
            generate_mirror_qv_circuit(3, n)


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_volume_conversion(return_type):
    """Check generated circuit's return type."""
    circuit = generate_mirror_qv_circuit(4, 3, return_type=return_type)
    assert return_type in circuit.__module__


def test_generate_model_circuit_with_seed():
    """Test that a model circuit is determined by its seed."""
    circuit_1 = generate_mirror_qv_circuit(4, 3, seed=1)
    circuit_2 = generate_mirror_qv_circuit(4, 3, seed=1)
    circuit_3 = generate_mirror_qv_circuit(4, 3, seed=2)

    assert cirq.approx_eq(circuit_1, circuit_2, atol=1e-12)
    assert circuit_2 != circuit_3


def test_circuit_decomposition():
    """Test that decomposed circuit consists of gates in default cirq gatest.
    As defined in cirq.protocols.decompose_protocol, this default gateset is
        ops.XPowGate,
        ops.YPowGate,
        ops.ZPowGate,
        ops.CZPowGate,
        ops.MeasurementGate,
        ops.GlobalPhaseGate
    """
    circuit = generate_mirror_qv_circuit(4, 3, decompose=True)
    for op in [operation for moment in circuit for operation in moment]:
        assert op in cirq.protocols.decompose_protocol.DECOMPOSE_TARGET_GATESET
