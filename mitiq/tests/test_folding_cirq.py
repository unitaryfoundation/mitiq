"""Unit tests for folding Cirq circuits."""

from copy import deepcopy

import numpy as np
import pytest
from cirq import (Circuit, GridQubit, LineQubit, ops, inverse)

from mitiq.utils import (_equal, random_circuit)
from mitiq.folding_cirq import (_is_measurement,
                                _pop_measurements,
                                _append_measurements,
                                _update_moment_indices,
                                _fold_gate_at_index_in_moment,
                                _fold_gates_in_moment,
                                fold_gates,
                                fold_moments,
                                _fold_all_gates_locally,
                                fold_gates_from_left,
                                fold_gates_from_right,
                                fold_gates_at_random,
                                fold_local,
                                fold_global)


def test_is_measurement():
    """Tests for checking if operations are measurements."""
    # Test circuit:
    # 0: ───H───X───Z───
    qbit = LineQubit(0)
    circ = Circuit(
        [ops.H.on(qbit), ops.X.on(qbit), ops.Z.on(qbit), ops.measure(qbit)]
    )
    for (i, op) in enumerate(circ.all_operations()):
        if i == 3:
            assert _is_measurement(op)
        else:
            assert not _is_measurement(op)


def test_pop_measurements_and_add_measurements():
    """Tests popping measurements from a circuit.."""
    # Test circuit:
    # 0: ───H───T───@───M───
    #               │   │
    # 1: ───H───M───┼───┼───
    #               │   │
    # 2: ───H───────X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(qreg)],
        [ops.T.on(qreg[0])],
        [ops.measure(qreg[1])],
        [ops.CNOT.on(qreg[0], qreg[2])],
        [ops.measure(qreg[0], qreg[2])]
    )
    copy = deepcopy(circ)
    measurements = _pop_measurements(copy)
    correct = Circuit(
        [ops.H.on_each(qreg)],
        [ops.T.on(qreg[0])],
        [ops.CNOT.on(qreg[0], qreg[2])],
    )
    assert _equal(copy, correct)
    _append_measurements(copy, measurements)
    assert _equal(copy, circ)


def test_update_moment_indices():
    """Tests indices of moments are properly updated."""
    moment_indices = {i: i for i in range(5)}
    _update_moment_indices(moment_indices, 3)
    assert moment_indices == {0: 0, 1: 1, 2: 2, 3: 5, 4: 6}
    _update_moment_indices(moment_indices, 0)
    assert moment_indices == {0: 2, 1: 3, 2: 4, 3: 7, 4: 8}
    _update_moment_indices(moment_indices, 4)
    assert moment_indices == {0: 2, 1: 3, 2: 4, 3: 7, 4: 10}
    with pytest.raises(ValueError):
        _update_moment_indices(moment_indices, 6)


def test_fold_gate_at_index_in_moment_one_qubit():
    """Tests local folding with a moment, index for a one qubit circuit."""
    # Test circuit:
    # 0: ───H───X───Z───
    qbit = LineQubit(0)
    circ = Circuit(
        [ops.H.on(qbit), ops.X.on(qbit), ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=0)
    assert folded == Circuit(
        [ops.H.on(qbit)] * 3 + [ops.X.on(qbit)] + [ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=0)
    assert folded == Circuit(
        [ops.H.on(qbit)] + [ops.X.on(qbit)] * 3 + [ops.Z.on(qbit)]
    )
    # Fold the zeroth operation in the second moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=2, gate_index=0)
    assert folded == Circuit(
        [ops.H.on(qbit)] + [ops.X.on(qbit)] + [ops.Z.on(qbit)] * 3
    )
    # Make sure the original circuit wasn't modified
    old = Circuit(
        [ops.H.on(qbit), ops.X.on(qbit), ops.Z.on(qbit)]
    )
    assert _equal(circ, old)


def test_fold_gate_at_index_in_moment_two_qubits():
    """Tests local folding with a moment, index for a two qubit circuit with single qubit gates."""
    # Test circuit:
    # 0: ───H───T───
    #
    # 1: ───H───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.T.on_each(*qreg)]
    )

    # Fold the zeroth operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=0)
    correct = Circuit(
        [ops.H.on(qreg[0]), ops.H.on(qreg[0])**-1] + list(circ.all_operations())
    )
    assert _equal(folded, correct)

    # Fold the first operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=1)
    correct = Circuit(
        [ops.H.on(qreg[1]), ops.H.on(qreg[1])**-1] + list(circ.all_operations())
    )
    assert _equal(folded, correct)

    # Fold the zeroth operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=0)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.T.on(qreg[0]), ops.T.on(qreg[0])**-1, ops.T.on(qreg[0])],
        [ops.T.on(qreg[1])]
    )
    assert _equal(folded, correct)

    # Fold the first operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=1)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.T.on(qreg[1]), ops.T.on(qreg[1]) ** -1, ops.T.on(qreg[1])],
        [ops.T.on(qreg[0])]
    )
    assert _equal(folded, correct)

    # Make sure the original circuit wasn't modified
    assert Circuit(
        [ops.H.on_each(*qreg), ops.T.on_each(*qreg)]
    )


def test_fold_gate_at_index_in_moment_two_qubit_gates():
    """Tests local folding with a moment, index for a two qubit circuit with two qubit gates."""
    # Test circuit:
    # 0: ───H───@───
    #           │
    # 1: ───────X───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on(qreg[0]), ops.CNOT.on(*qreg)]
    )

    # Fold the zeroth operation in the zeroth moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=0, gate_index=0)
    correct = Circuit(
        [ops.H.on(qreg[0])**-1, ops.H.on(qreg[0])] + list(circ.all_operations())
    )
    assert _equal(folded, correct)

    # Fold the zeroth operation in the first moment
    folded = deepcopy(circ)
    _fold_gate_at_index_in_moment(folded, moment_index=1, gate_index=0)
    correct = Circuit(
        list(circ.all_operations()) + [ops.CNOT.on(*qreg)] * 2
    )
    assert _equal(folded, correct)

    # Make sure the original circuit wasn't modified
    old = Circuit(
        [ops.H.on(qreg[0]), ops.CNOT.on(*qreg)]
    )
    assert _equal(circ, old)


def test_fold_gate_at_index_in_moment_empty_circuit():
    """Tests local folding with a moment, index with an empty circuit."""
    circ = Circuit()

    # Fold the zeroth operation in the zeroth moment
    with pytest.raises(IndexError):
        _fold_gate_at_index_in_moment(circ, moment_index=0, gate_index=0)


def test_fold_gate_at_index_in_moment_bad_moment():
    """Tests local folding with a moment index not in the input circuit."""
    qreg = [GridQubit(x, y) for x in range(2) for y in range(2)]
    circ = Circuit(
        [ops.H.on_each(*qreg)]
    )
    with pytest.raises(IndexError):
        _fold_gate_at_index_in_moment(circ, 1, 0)


def test_fold_gates_in_moment_single_qubit_gates():
    """Tests folding gates at given indices within a moment."""
    # Test circuit:
    # 0: ───H───T───
    #
    # 1: ───H───T───
    #
    # 2: ───H───T───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.T.on_each(*qreg)]
    )

    # Fold all gates in the zeroth moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=0, gate_indices=[0, 1, 2])
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3 + [ops.T.on_each(*qreg)]
    )
    assert _equal(folded, correct)

    # Fold a subset of gates in the first moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=1, gate_indices=[0, 2])
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.T.on(qreg[0]), ops.T.on(qreg[0])**-1],
        [ops.T.on(qreg[2]), ops.T.on(qreg[2])**-1],
        [ops.T.on_each(*qreg)]
    )
    assert _equal(folded, correct)


def test_fold_gates_in_moment_multi_qubit_gates():
    """Tests folding gates at given indices within a moment."""
    # Test circuit:
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )

    # Fold the CNOT gate in the first moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=1, gate_indices=[0])
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )
    assert _equal(folded, correct)

    # Fold the T gate in the first moment
    folded = deepcopy(circ)
    _fold_gates_in_moment(folded, moment_index=1, gate_indices=[0, 1])
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2])**-1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )
    assert _equal(folded, correct)

    # Fold the Toffoli gate in the second moment
    _fold_gates_in_moment(circ, moment_index=2, gate_indices=[0])
    correct = Circuit(
        [ops.H.on_each(*qreg)] +
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3
    )
    assert _equal(circ, correct)


def test_fold_gates():
    """Test folding gates at specified indices within specified moments."""
    # Test circuit:
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )
    folded = fold_gates(circ, moment_indices=[0, 1], gate_indices=[[0, 1, 2], [1]])
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )
    assert _equal(folded, correct)


def test_fold_moments():
    """Tests folding moments in a circuit."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )

    # Fold a single moment
    folded = fold_moments(circ, moment_indices=[0])
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 2,
        list(circ.all_operations())
    )
    assert _equal(folded, correct)

    # Fold another single moment
    folded = fold_moments(circ, moment_indices=[2])
    correct = Circuit(
        list(circ.all_operations()),
        [ops.TOFFOLI.on(*qreg)] * 2
    )
    assert _equal(folded, correct)

    # Fold two moments
    folded = fold_moments(circ, moment_indices=[0, 2])
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 2,
        list(circ.all_operations()),
        [ops.TOFFOLI.on(*qreg)] * 2
    )
    assert _equal(folded, correct)

    # Fold three moments
    folded = fold_moments(circ, moment_indices=[0, 1, 2])
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3
    )
    assert _equal(folded, correct)

    # Make sure the original circuit wasn't modified
    old = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )
    assert _equal(circ, old)


def test_fold_all_gates_locally():
    # Test circuit:
    # 0: ───H───@───
    #           │
    # 1: ───H───X───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on(qreg[0])],
        [ops.CNOT.on(*qreg)]
    )
    folded = deepcopy(circ)
    _fold_all_gates_locally(folded)
    correct = Circuit(
        [ops.H.on(qreg[0])] * 3,
        [ops.CNOT.on(*qreg)] * 3,
    )
    assert _equal(folded, correct)


def test_fold_all_gates_locally_three_qubits():
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )
    _fold_all_gates_locally(circ)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2])**-1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3
    )
    assert _equal(circ, correct)


def test_fold_from_left_two_qubits():
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[1])]
    )

    # Intermediate stretch factor
    folded = fold_gates_from_left(circ, stretch=2.5)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(*qreg)] * 3,
        [ops.T.on(qreg[1])]
    )
    assert _equal(folded, correct)

    # Full stretch factor
    folded = fold_gates_from_left(circ, stretch=3)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(*qreg)] * 3,
        [ops.T.on(qreg[1]), ops.T.on(qreg[1])**-1, ops.T.on(qreg[1])]
    )
    assert _equal(folded, correct)


def test_fold_from_left_three_qubits():
    """Unit test for folding gates from left to stretch a circuit."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )

    folded = fold_gates_from_left(circ, stretch=2)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 2,
        list(circ.all_operations())
    )
    assert _equal(folded, correct)


def test_fold_from_left_no_stretch():
    """Unit test for folding gates from left for a stretch factor of one."""
    circuit = random_circuit(depth=100)
    folded = fold_gates_from_left(circuit, stretch=1)
    assert _equal(folded, circuit)
    assert not (folded is circuit)


def test_fold_from_left_bad_stretch():
    """Tests that a ValueError is raised for an invalid stretch factor."""
    circuit = random_circuit(100)
    with pytest.raises(ValueError):
        fold_gates_from_left(circuit, stretch=10)


def test_fold_from_left_with_terminal_measurements_min_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    folded = fold_gates_from_left(circ, stretch=1.)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    assert _equal(folded, correct)


def test_fold_from_left_with_terminal_measurements_max_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    folded = fold_gates_from_left(circ, stretch=3.)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2])**-1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
        [ops.measure_each(*qreg)]
    )
    assert _equal(folded, correct)


@pytest.mark.parametrize(["fold_method"],
                         [[fold_gates_from_left],
                          [fold_gates_from_right],
                          [fold_gates_at_random]])
def test_fold_with_intermediate_measurements_raises_error(fold_method):
    """Tests folding from left with intermediate measurements."""
    # Test circuit
    #
    qbit = LineQubit(0)
    circ = Circuit(
        [ops.H.on(qbit)],
        [ops.measure(qbit)],
        [ops.T.on(qbit)]
    )
    with pytest.raises(ValueError):
        fold_method(circ, stretch=2.)


def test_fold_from_right_basic():
    """Tests folding gates from the right for a two-qubit circuit."""
    # Test circuit:
    # 0: ───H───@───────
    #           │
    # 1: ───H───X───T───
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[1])]
    )

    # Small stretch factor
    folded = fold_gates_from_right(circ, stretch=1.5)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(*qreg)],
        [ops.T.on(qreg[1]), ops.T.on(qreg[1])**-1, ops.T.on(qreg[1])]
    )
    assert _equal(folded, correct)

    # Intermediate stretch factor
    folded = fold_gates_from_right(circ, stretch=2.5)
    correct = Circuit(
        [ops.H.on(qreg[0])] * 3,
        [ops.H.on(qreg[1])],
        [ops.CNOT.on(*qreg)] * 3,
        [ops.T.on(qreg[1]), ops.T.on(qreg[1])**-1, ops.T.on(qreg[1])]
    )
    assert _equal(folded, correct)


def test_fold_from_right_max_stretch():
    """Tests that folding from right = folding from left with maximum stretch."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )

    left_folded = fold_gates_from_left(circ, stretch=3.)
    right_folded = fold_gates_from_right(circ, stretch=3.)
    assert _equal(left_folded, right_folded)


def test_fold_from_right_with_terminal_measurements_min_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    folded = fold_gates_from_right(circ, stretch=1.)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    assert _equal(folded, correct)


def test_fold_from_right_with_terminal_measurements_max_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    folded = fold_gates_from_right(circ, stretch=3.)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2])**-1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
        [ops.measure_each(*qreg)]
    )
    assert _equal(folded, correct)


def test_fold_gates_at_random_no_stretch():
    """Tests folded circuit is identical for a stretch factor of one."""
    circuit = random_circuit(10)
    folded = fold_gates_at_random(circuit, stretch=1, seed=None)
    assert _equal(folded, circuit)


def test_fold_gates_at_random_seed_one_qubit():
    """Test for folding gates at random on a one qubit circuit with a seed for repeated behavior."""
    qubit = LineQubit(0)
    circuit = Circuit(
        [ops.X.on(qubit), ops.Y.on(qubit), ops.Z.on(qubit)]
    )
    # Small stretch
    folded = fold_gates_at_random(circuit, stretch=1.4, seed=1)
    correct = Circuit(
        [ops.X.on(qubit)],
        [ops.Y.on(qubit)] * 3,
        [ops.Z.on(qubit)]
    )
    assert _equal(folded, correct)

    # Medium stretch, fold two gates
    folded = fold_gates_at_random(circuit, stretch=2.5, seed=1)
    correct = Circuit(
        [ops.X.on(qubit)],
        [ops.Y.on(qubit)] * 3,
        [ops.Z.on(qubit)] * 3,
    )
    assert _equal(folded, correct)

    # Max stretch, fold three gates
    folded = fold_gates_at_random(circuit, stretch=3, seed=1)
    correct = Circuit(
        [ops.X.on(qubit)] * 3,
        [ops.Y.on(qubit)] * 3,
        [ops.Z.on(qubit)] * 3,
    )
    assert _equal(folded, correct)


def test_fold_random_min_stretch():
    """Tests that folding at random with min stretch returns a copy of the input circuit."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )

    folded = fold_gates_at_random(circ, stretch=1, seed=1)
    assert _equal(folded, circ)
    assert folded is not circ


def test_fold_random_max_stretch():
    """Tests that folding at random with max stretch folds all gates on a multi-qubit circuit."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )

    folded = fold_gates_at_random(circ, stretch=3, seed=1)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2])**-1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3
    )
    assert _equal(folded, correct)


def test_fold_random_bad_stretch():
    """Tests that an error is raised when a bad stretch is provided."""
    with pytest.raises(ValueError):
        fold_gates_at_random(Circuit(), stretch=4)


def test_fold_random_no_repeats():
    """Tests folding at random to ensure that no gates are folded twice and folded gates
    are not folded again.
    """
    # Test circuit:
    # 0: ───H───@───Y───@───
    #           │       │
    # 1: ───────X───X───@───
    # Note that each gate only occurs once and is self-inverse.
    # This allows us to check that no gates are folded more than once
    qreg = LineQubit.range(2)
    circ = Circuit(
        [ops.H.on_each(qreg[0])],
        [ops.CNOT.on(*qreg)],
        [ops.X.on(qreg[1])],
        [ops.Y.on(qreg[0])],
        [ops.CZ.on(*qreg)]
    )
    circuit_ops = set(circ.all_operations())

    for stretch in np.linspace(1., 3., 5):
        folded = fold_gates_at_random(circ, stretch=stretch, seed=1)
        gates = list(folded.all_operations())
        counts = {gate: gates.count(gate) for gate in circuit_ops}
        assert all(count <= 3 for count in counts.values())


def test_fold_random_with_terminal_measurements_min_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    folded = fold_gates_at_random(circ, stretch=1.)
    correct = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    assert _equal(folded, correct)


def test_fold_random_with_terminal_measurements_max_stretch():
    """Tests folding from left with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    folded = fold_gates_at_random(circ, stretch=3.)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2])**-1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3,
        [ops.measure_each(*qreg)]
    )
    assert _equal(folded, correct)


def test_fold_local_small_stretch_from_left():
    """Test for local folding with stretch < 3."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )

    folded = fold_local(circ, stretch=2, fold_method=fold_gates_from_left)
    correct = Circuit(
        [ops.H.on_each(*qreg)] * 2,
        list(circ.all_operations())
    )
    assert _equal(folded, correct)


def test_fold_local_stretch_three_from_left():
    """Test for local folding with stretch > 3."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )
    folded = fold_local(circ, stretch=3, fold_method=fold_gates_from_left)
    assert _equal(folded, fold_gates_from_left(circ, stretch=3))


def test_fold_local_big_stretch_from_left():
    """Test for local folding with stretch > 3."""
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg), ops.CNOT.on(qreg[0], qreg[1]), ops.T.on(qreg[2]), ops.TOFFOLI.on(*qreg)]
    )
    folded = fold_local(circ, stretch=4, fold_method=fold_gates_from_left)
    correct = Circuit(
        [ops.H.on(qreg[0])] * 7,
        [ops.H.on(qreg[1])] * 5,
        [ops.H.on(qreg[2])] * 3,
        [ops.CNOT.on(qreg[0], qreg[1])] * 3,
        [ops.T.on(qreg[2]), ops.T.on(qreg[2]) ** -1, ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)] * 3
    )
    assert _equal(folded, correct)


def test_global_fold_min_stretch():
    """Tests that global fold with stretch = 1 is the same circuit."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )

    folded = fold_global(circ, 1.)
    assert _equal(folded, circ)
    assert folded is not circ


def test_global_fold_min_stretch_with_terminal_measurements():
    """Tests that global fold with stretch = 1 is the same circuit."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        [ops.measure_each(*qreg)]
    )
    folded = fold_global(circ, stretch=1.)
    assert _equal(folded, circ)
    assert folded is not circ


def test_global_fold_raises_error_intermediate_measurements():
    """Tests than an error is raised when trying to globally fold
    a circuit with intermediate measurements.
    """
    qbit = GridQubit(0, 0)
    circ = Circuit([ops.H.on(qbit), ops.measure(qbit), ops.H.on(qbit)])
    with pytest.raises(ValueError):
        fold_global(circ, stretch=3)


def test_global_fold_stretch_factor_of_three():
    """Tests global folding with the stretch as a factor of 3."""
    # Test circuit
    # 0: ───H───@───@───
    #           │   │
    # 1: ───H───X───@───
    #               │
    # 2: ───H───T───X───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)]
    )
    folded = fold_global(circ, stretch=3.)
    correct = Circuit(circ, inverse(circ), circ)
    assert _equal(folded, correct)


def test_global_fold_stretch_factor_of_three_with_terminal_measurements():
    """Tests global folding with the stretch as a factor of 3 for a circuit with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    meas = Circuit([ops.measure_each(*qreg)])
    folded = fold_global(circ + meas, stretch=3.)
    correct = Circuit(circ, inverse(circ), circ, meas)
    assert _equal(folded, correct)


def test_global_fold_stretch_factor_nine_with_terminal_measurements():
    """Tests global folding with the stretch as a factor of 9 for a circuit with terminal measurements."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    meas = Circuit([ops.measure_each(*qreg)])
    folded = fold_global(circ + meas, stretch=9.)
    correct = Circuit([circ, inverse(circ)] * 4, [circ], [meas])
    assert _equal(folded, correct)


def test_global_fold_stretch_factor_eight_terminal_measurements():
    """Tests global folding with a stretch factor not a multiple of three so that local folding is also called."""
    # Test circuit
    # 0: ───H───@───@───M───
    #           │   │
    # 1: ───H───X───@───M───
    #               │
    # 2: ───H───T───X───M───
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(*qreg)],
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
    )
    meas = Circuit(ops.measure_each(*qreg))
    folded = fold_global(circ + meas, stretch=3.5)
    correct = Circuit(
        circ,
        inverse(circ),
        circ,
        inverse(Circuit([ops.CNOT.on(qreg[0], qreg[1])],
                [ops.T.on(qreg[2])],
                [ops.TOFFOLI.on(*qreg)])),
        [ops.CNOT.on(qreg[0], qreg[1])],
        [ops.T.on(qreg[2])],
        [ops.TOFFOLI.on(*qreg)],
        meas
    )
    assert _equal(folded, correct)
