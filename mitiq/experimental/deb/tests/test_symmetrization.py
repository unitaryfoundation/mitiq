# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import cirq
import numpy as np
import pytest

from mitiq.experimental.deb.symmetrization import (
    _construct_variants,
    _permute,
    construct_circuits,
)

qubits = cirq.LineQubit.range(3)
base_circuit = cirq.Circuit(
    [cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.X(qubits[2])]
)


def _gate_multiset(circuit: cirq.Circuit) -> Counter:
    return Counter(str(op.gate) for op in circuit.all_operations())


def test_construct_circuits_returns_requested_number():
    variants = construct_circuits(base_circuit, num_variants=7, random_state=0)
    assert len(variants) == 7


def test_construct_circuits_is_reproducible():
    first = construct_circuits(base_circuit, 5, random_state=42)
    second = construct_circuits(base_circuit, 5, random_state=42)
    assert first == second


def test_different_seeds_give_different_variants():
    first = construct_circuits(base_circuit, 6, random_state=1)
    second = construct_circuits(base_circuit, 6, random_state=2)
    assert first != second


def test_variants_relabel_the_same_qubits_and_gates():
    # A permutation only relabels qubits, so the qubit set and the multiset of
    # gates are unchanged.
    variants = construct_circuits(base_circuit, 8, random_state=1)
    base_qubits = set(base_circuit.all_qubits())
    base_gates = _gate_multiset(base_circuit)
    for variant in variants:
        assert set(variant.all_qubits()) == base_qubits
        assert _gate_multiset(variant) == base_gates


def test_permutation_actually_changes_some_variants():
    variants = construct_circuits(base_circuit, 10, random_state=4)
    assert any(variant != base_circuit for variant in variants)


def test_variants_are_unitarily_equivalent():
    # Each variant is a relabeling of the circuit, so undoing its permutation
    # recovers the original unitary exactly.
    sorted_qubits = sorted(base_circuit.all_qubits())
    base_unitary = cirq.unitary(base_circuit)
    for variant, permutation in _construct_variants(base_circuit, 8, 1):
        inverse = [0] * len(permutation)
        for i, target in enumerate(permutation):
            inverse[target] = i
        restored = _permute(variant, sorted_qubits, inverse)
        assert np.allclose(cirq.unitary(restored), base_unitary)


def test_single_qubit_circuit_has_only_the_identity_permutation():
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    variants = construct_circuits(circuit, 5, random_state=0)
    assert all(variant == circuit for variant in variants)


def test_non_positive_num_variants_raises():
    with pytest.raises(ValueError):
        construct_circuits(base_circuit, num_variants=0)
