# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import cirq
import pytest

from mitiq.experimental.deb.symmetrization import construct_circuits

qubits = cirq.LineQubit.range(2)
base_circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])


def test_construct_circuits_returns_requested_number():
    variants = construct_circuits(base_circuit, num_variants=7, random_state=0)
    assert len(variants) == 7


def test_construct_circuits_is_reproducible():
    first = construct_circuits(base_circuit, 5, random_state=42)
    second = construct_circuits(base_circuit, 5, random_state=42)
    assert first == second


def test_different_seeds_give_different_variants():
    first = construct_circuits(base_circuit, 5, random_state=1)
    second = construct_circuits(base_circuit, 5, random_state=2)
    assert first != second


def test_variants_keep_the_original_operations():
    variants = construct_circuits(base_circuit, 5, random_state=1)
    base_ops = Counter(str(op) for op in base_circuit.all_operations())
    for variant in variants:
        variant_ops = Counter(str(op) for op in variant.all_operations())
        assert variant_ops & base_ops == base_ops


def test_added_gates_are_symmetric_paulis():
    variants = construct_circuits(base_circuit, 10, random_state=3)
    base_ops = Counter(str(op) for op in base_circuit.all_operations())
    pauli_gates = (cirq.X, cirq.Y, cirq.Z)
    for variant in variants:
        extra = Counter(str(op) for op in variant.all_operations()) - base_ops
        # The pre- and post-layers are identical, so each added Pauli on a
        # given qubit appears exactly twice.
        assert all(count == 2 for count in extra.values())
        added = [
            op for op in variant.all_operations() if str(op) not in base_ops
        ]
        assert all(op.gate in pauli_gates for op in added)


def test_non_positive_num_variants_raises():
    with pytest.raises(ValueError):
        construct_circuits(base_circuit, num_variants=0)
