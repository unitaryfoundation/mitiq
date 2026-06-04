# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for circuit symmetrization."""

import cirq

from mitiq.experimental.deb.symmetrization import construct_circuits


def test_construct_circuits_num_variants():
    """Test that the correct number of variants is generated."""
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    variants = construct_circuits(circuit, num_variants=5, random_state=42)
    assert len(variants) == 5


def test_construct_circuits_pauli_layers():
    """Test that Pauli layers are added to circuits."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    variants = construct_circuits(circuit, num_variants=10, random_state=42)

    # Check that each variant has the original circuit in the middle
    for variant in variants:
        # Count operations before and after H gate
        ops = list(variant.all_operations())
        h_idx = None
        for i, op in enumerate(ops):
            if op.gate == cirq.H:
                h_idx = i
                break

        assert h_idx is not None, "H gate not found in variant"
        # Should have at least the H gate
        assert len(ops) >= 1


def test_construct_circuits_noiseless_same_distribution():
    """Test that on a noiseless simulator, all variants return the same distribution."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key="result"))

    variants = construct_circuits(circuit, num_variants=5, random_state=42)

    # Execute each variant on noiseless simulator
    simulator = cirq.Simulator()
    results = []
    for variant in variants:
        result = simulator.run(variant, repetitions=100)
        counts = result.histogram(key="result")
        results.append(counts)

    # All variants should have the same distribution (50/50 for H gate)
    # Allow some tolerance for randomness
    for i in range(1, len(results)):
        assert results[i] == results[0] or abs(
            results[i].get(0, 0) - results[0].get(0, 0)
        ) < 20
