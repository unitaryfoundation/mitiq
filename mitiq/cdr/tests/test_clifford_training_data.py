# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for generating (near) Clifford circuits."""

import cirq
import numpy as np
import pytest
from cirq.circuits import Circuit

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.cdr._testing import random_x_z_cnot_circuit
from mitiq.cdr.clifford_training_data import (
    _map_to_near_clifford,
    _replace,
    _select,
    generate_training_circuits,
)
from mitiq.cdr.clifford_utils import count_non_cliffords, is_clifford
from mitiq.interface import convert_from_mitiq


def test_generate_training_circuits():
    circuit = random_x_z_cnot_circuit(
        cirq.LineQubit.range(3), n_moments=5, random_state=1
    )
    assert not is_clifford(circuit)

    (clifford_circuit,) = generate_training_circuits(
        circuit, num_training_circuits=1, fraction_non_clifford=0.0
    )
    assert is_clifford(clifford_circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_generate_training_circuits_any_qprogram(circuit_type):
    circuit = random_x_z_cnot_circuit(
        cirq.LineQubit.range(3), n_moments=5, random_state=1
    )
    circuit = convert_from_mitiq(circuit, circuit_type)

    (clifford_circuit,) = generate_training_circuits(
        circuit, num_training_circuits=1, fraction_non_clifford=0.0
    )
    assert is_clifford(clifford_circuit)


@pytest.mark.parametrize("method", ("uniform", "gaussian"))
def test_select_all(method):
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]
    indices = _select(
        ops, 0.0, method=method, random_state=np.random.RandomState(1)
    )
    assert np.allclose(indices, np.array(list(range(len(ops)))))


@pytest.mark.parametrize("method", ("uniform", "gaussian"))
def test_select_some(method):
    n = 10  # Number to select.
    q = cirq.GridQubit(1, 1)
    ops = [cirq.ops.rz(a).on(q) for a in np.random.randn(n)]
    indices = _select(ops, fraction_non_clifford=0.5, method=method)
    assert len(indices) == n // 2


def test_select_bad_method():
    with pytest.raises(ValueError, match="Arg `method_select` must be"):
        _select([], fraction_non_clifford=0.0, method="unknown method")


@pytest.mark.parametrize("method", ("closest", "uniform", "gaussian"))
def test_replace(method):
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(0.01).on(q), cirq.ops.rz(-0.77).on(q)]

    new_ops = _replace(
        non_clifford_ops=ops,
        method=method,
        random_state=np.random.RandomState(1),
    )

    assert len(new_ops) == len(ops)
    assert all(cirq.has_stabilizer_effect(op) for op in new_ops)


def test_map_to_near_clifford():
    q = cirq.LineQubit(0)
    ops = [cirq.ops.rz(np.pi / 2.0 + 0.01).on(q), cirq.ops.rz(-0.22).on(q)]

    new_ops = _map_to_near_clifford(
        ops,
        fraction_non_clifford=0.0,
        method_select="uniform",
        method_replace="uniform",
        random_state=np.random.RandomState(2),
    )
    expected_ops = [cirq.rz(np.pi * 1).on(q), cirq.rz(np.pi * 1.5).on(q)]
    assert new_ops == expected_ops


def test_generate_training_circuits_bad_methods():
    with pytest.raises(ValueError):
        generate_training_circuits(
            Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_training_circuits=1,
            fraction_non_clifford=0.0,
            method_select="unknown select method",
        )

    with pytest.raises(ValueError):
        generate_training_circuits(
            Circuit(cirq.ops.rx(0.5).on(cirq.LineQubit(0))),
            num_training_circuits=1,
            fraction_non_clifford=0.0,
            method_replace="unknown replace method",
        )


def test_generate_training_circuits_with_clifford_circuit():
    circuit = Circuit(cirq.ops.rx(0.0).on(cirq.LineQubit(0)))
    assert generate_training_circuits(
        circuit,
        num_training_circuits=2,
        fraction_non_clifford=0.0,
    ) == [circuit, circuit]


@pytest.mark.parametrize("method_select", ["uniform", "gaussian"])
@pytest.mark.parametrize("method_replace", ["uniform", "gaussian", "closest"])
@pytest.mark.parametrize(
    "kwargs", [{}, {"sigma_select": 0.5, "sigma_replace": 0.5}]
)
def test_generate_training_circuits_mega(
    method_select, method_replace, kwargs
):
    circuit = random_x_z_cnot_circuit(qubits=4, n_moments=10, random_state=1)
    num_train = 10
    fraction_non_clifford = 0.1

    train_circuits = generate_training_circuits(
        circuit,
        num_training_circuits=num_train,
        fraction_non_clifford=0.1,
        random_state=np.random.RandomState(13),
        method_select=method_select,
        method_replace=method_replace,
        **kwargs,
    )
    assert len(train_circuits) == num_train

    for train_circuit in train_circuits:
        assert set(train_circuit.all_qubits()) == set(circuit.all_qubits())
        assert count_non_cliffords(train_circuit) == int(
            round(fraction_non_clifford * count_non_cliffords(circuit))
        )


@pytest.mark.parametrize("method", ["uniform", "gaussian"])
def test_select(method):
    q = cirq.NamedQubit("q")
    non_clifford_ops = [
        cirq.ops.rz(a).on(q) for a in np.linspace(1, 2, 10) / np.e
    ]
    fraction_non_clifford = 0.4

    indices = _select(
        non_clifford_ops,
        fraction_non_clifford,
        method=method,
        sigma=0.5,
        random_state=np.random.RandomState(1),
    )
    assert all(isinstance(index, int) for index in indices)
    assert len(indices) == int(
        round((1.0 - fraction_non_clifford) * len(non_clifford_ops))
    )
