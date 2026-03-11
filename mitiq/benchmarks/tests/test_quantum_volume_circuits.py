# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for quantum volume circuits. The Cirq functions that do the main work
are tested here:
cirq-core/cirq/contrib/quantum_volume/quantum_volume_test.py

Tests below check that generate_quantum_volume_circuit() works as a wrapper and
fits with Mitiq's interface.
"""

import cirq
import pytest
from cirq import protocols

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.quantum_volume_circuits import (
    compute_heavy_bitstrings,
    generate_quantum_volume_circuit,
)


def test_generate_model_circuit_no_seed():
    """Test that random circuit of the right length is generated."""
    circuit, _ = generate_quantum_volume_circuit(3, 3)
    assert len(circuit) == 3


def test_generate_model_circuit_with_seed():
    """Test that a model circuit is determined by its seed."""
    circuit_1, _ = generate_quantum_volume_circuit(3, 3, seed=1)
    circuit_2, _ = generate_quantum_volume_circuit(3, 3, seed=1)
    circuit_3, _ = generate_quantum_volume_circuit(3, 3, seed=2)

    assert circuit_1 == circuit_2
    assert circuit_2 != circuit_3


def test_compute_heavy_bitstrings():
    """Test that the heavy bitstrings can be computed from a given circuit."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.X(a), cirq.Y(b)]),
            cirq.Moment([]),
            cirq.Moment([cirq.CNOT(a, c)]),
            cirq.Moment([cirq.Z(a), cirq.H(b)]),
        ]
    )
    true_heavy_set = [[1, 0, 1], [1, 1, 1]]
    computed_heavy_set = compute_heavy_bitstrings(model_circuit, 3)
    assert computed_heavy_set == true_heavy_set


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
    circuit, _ = generate_quantum_volume_circuit(3, 3, decompose=True)
    for op in [operation for moment in circuit for operation in moment]:
        assert op in protocols.decompose_protocol.DECOMPOSE_TARGET_GATESET


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_volume_conversion(return_type):
    circuit, _ = generate_quantum_volume_circuit(3, 3, return_type=return_type)
    assert return_type in circuit.__module__
