# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utility functions."""

from copy import deepcopy

import cirq
import numpy as np
import pytest
from cirq import (
    CNOT,
    Circuit,
    ControlledGate,
    H,
    LineQubit,
    MeasurementGate,
    S,
    T,
    X,
    Y,
    Z,
    depolarize,
    ops,
)

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.interface import compare_cost, convert_from_mitiq
from mitiq.utils import (
    _append_measurements,
    _are_close_dict,
    _circuit_to_choi,
    _equal,
    _is_measurement,
    _max_ent_state_circuit,
    _operation_to_choi,
    _pop_measurements,
    _simplify_circuit_exponents,
    _simplify_gate_exponent,
    arbitrary_tensor_product,
    matrix_kronecker_product,
    matrix_to_vector,
    operator_ptm_vector_rep,
    qem_methods,
    vector_to_matrix,
)


def test_arbitrary_tensor_product():
    terms = [np.random.rand(dim, dim) for dim in range(1, 4)]
    expected = np.kron(np.kron(terms[0], terms[1]), terms[2])
    assert np.allclose(arbitrary_tensor_product(*terms), expected)
    # Check limit cases
    one_term = np.random.rand(5, 5)
    assert np.allclose(arbitrary_tensor_product(one_term), one_term)
    assert np.allclose(arbitrary_tensor_product(2.0, one_term), 2.0 * one_term)
    assert np.allclose(arbitrary_tensor_product(3.0, 4.0), 12.0)
    with pytest.raises(TypeError, match="requires at least one argument"):
        assert np.allclose(arbitrary_tensor_product(), one_term)


def test_matrix_to_vector():
    for d in [1, 2, 3, 4]:
        mat = np.random.rand(d, d)
        assert matrix_to_vector(mat).shape == (d**2,)
        assert (vector_to_matrix(matrix_to_vector(mat)) == mat).all


def test_vector_to_matrix():
    for d in [1, 2, 3, 4]:
        vec = np.random.rand(d**2)
        assert vector_to_matrix(vec).shape == (d, d)
        assert (matrix_to_vector(vector_to_matrix(vec)) == vec).all


@pytest.mark.parametrize("require_qubit_equality", [True, False])
def test_circuit_equality_identical_qubits(require_qubit_equality):
    qreg = cirq.NamedQubit.range(5, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qreg))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qreg))
    assert circA is not circB
    assert _equal(circA, circB, require_qubit_equality=require_qubit_equality)


@pytest.mark.parametrize("require_qubit_equality", [True, False])
def test_circuit_equality_nonidentical_but_equal_qubits(
    require_qubit_equality,
):
    n = 5
    qregA = cirq.NamedQubit.range(n, prefix="q_")
    qregB = cirq.NamedQubit.range(n, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert circA is not circB
    assert _equal(circA, circB, require_qubit_equality=require_qubit_equality)


def test_circuit_equality_linequbit_gridqubit_equal_indices():
    n = 10
    qregA = cirq.LineQubit.range(n)
    qregB = [cirq.GridQubit(x, 0) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_linequbit_gridqubit_unequal_indices():
    n = 10
    qregA = cirq.LineQubit.range(n)
    qregB = [cirq.GridQubit(x + 3, 0) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_linequbit_namedqubit_equal_indices():
    n = 8
    qregA = cirq.LineQubit.range(n)
    qregB = cirq.NamedQubit.range(n, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_linequbit_namedqubit_unequal_indices():
    n = 11
    qregA = cirq.LineQubit.range(n)
    qregB = [cirq.NamedQubit(str(x + 10)) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_gridqubit_namedqubit_equal_indices():
    n = 8
    qregA = [cirq.GridQubit(0, x) for x in range(n)]
    qregB = cirq.NamedQubit.range(n, prefix="q_")
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_gridqubit_namedqubit_unequal_indices():
    n = 5
    qregA = [cirq.GridQubit(x + 2, 0) for x in range(n)]
    qregB = [cirq.NamedQubit(str(x + 10)) for x in range(n)]
    circA = cirq.Circuit(cirq.ops.H.on_each(*qregA))
    circB = cirq.Circuit(cirq.ops.H.on_each(*qregB))
    assert _equal(circA, circB, require_qubit_equality=False)
    assert not _equal(circA, circB, require_qubit_equality=True)


def test_circuit_equality_unequal_measurement_keys_terminal_measurements():
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="one") for q in qreg)

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="two") for q in qreg)

    assert _equal(circ1, circ2, require_measurement_equality=False)
    assert not _equal(circ1, circ2, require_measurement_equality=True)


@pytest.mark.parametrize("require_measurement_equality", [True, False])
def test_circuit_equality_equal_measurement_keys_terminal_measurements(
    require_measurement_equality,
):
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="z") for q in qreg)

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="z") for q in qreg)

    assert _equal(
        circ1, circ2, require_measurement_equality=require_measurement_equality
    )


def test_circuit_equality_unequal_measurement_keys_nonterminal_measurements():
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    end_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=5, op_density=0.99, random_state=2
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="one") for q in qreg)
    circ1 += end_circuit

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="two") for q in qreg)
    circ2 += end_circuit

    assert _equal(circ1, circ2, require_measurement_equality=False)
    assert not _equal(circ1, circ2, require_measurement_equality=True)


@pytest.mark.parametrize("require_measurement_equality", [True, False])
def test_circuit_equality_equal_measurement_keys_nonterminal_measurements(
    require_measurement_equality,
):
    base_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=10, op_density=0.99, random_state=1
    )
    end_circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=5, op_density=0.99, random_state=2
    )
    qreg = list(base_circuit.all_qubits())

    circ1 = deepcopy(base_circuit)
    circ1.append(cirq.measure(q, key="z") for q in qreg)
    circ1 += end_circuit

    circ2 = deepcopy(base_circuit)
    circ2.append(cirq.measure(q, key="z") for q in qreg)
    circ2 += end_circuit

    assert _equal(
        circ1, circ2, require_measurement_equality=require_measurement_equality
    )


def test_is_measurement():
    """Tests for checking if operations are measurements."""
    # Test circuit:
    # 0: ───H───X───Z───
    qbit = LineQubit(0)
    circ = Circuit(
        [ops.H.on(qbit), ops.X.on(qbit), ops.Z.on(qbit), ops.measure(qbit)]
    )
    for i, op in enumerate(circ.all_operations()):
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
        [ops.measure(qreg[0], qreg[2])],
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


@pytest.mark.parametrize("gate", [X**3, Y**-3, Z**-1, H**-1])
def test_simplify_gate_exponent(gate):
    # Check exponent is simplified to 1
    assert _simplify_gate_exponent(gate).exponent == 1
    # Check simplified gate is equivalent to the input
    assert _simplify_gate_exponent(gate) == gate


@pytest.mark.parametrize("gate", [T**-1, S**-1, MeasurementGate(1)])
def test_simplify_gate_exponent_with_gates_that_cannot_be_simplified(gate):
    # Check the gate is not simplified (same representation)
    assert _simplify_gate_exponent(gate).__repr__() == gate.__repr__()


def test_simplify_circuit_exponents_controlled_gate():
    circuit = Circuit(
        ControlledGate(CNOT, num_controls=1).on(*LineQubit.range(3))
    )
    copy = circuit.copy()

    _simplify_circuit_exponents(circuit)
    assert _equal(circuit, copy)


def test_simplify_circuit_exponents():
    qreg = LineQubit.range(2)
    circuit = Circuit([H.on(qreg[0]), CNOT.on(*qreg), Z.on(qreg[1])])

    # Invert circuit
    inverse_circuit = cirq.inverse(circuit)
    inverse_repr = inverse_circuit.__repr__()
    inverse_qasm = inverse_circuit._to_qasm_output().__str__()

    # Expected circuit after simplification
    expected_inv = Circuit([Z.on(qreg[1]), CNOT.on(*qreg), H.on(qreg[0])])
    expected_repr = expected_inv.__repr__()
    expected_qasm = expected_inv._to_qasm_output().__str__()

    # Check inverse_circuit is logically equivalent to expected_inverse
    # but they have a different representation
    assert inverse_circuit == expected_inv
    assert inverse_repr != expected_repr
    assert inverse_qasm != expected_qasm

    # Simplify the circuit
    _simplify_circuit_exponents(inverse_circuit)

    # Check inverse_circuit has the expected simplified representation
    simplified_repr = inverse_circuit.__repr__()
    simplified_qasm = inverse_circuit._to_qasm_output().__str__()
    assert inverse_circuit == expected_inv
    assert simplified_repr == expected_repr
    assert simplified_qasm == expected_qasm


def test_simplify_circuit_exponents_with_non_self_inverse_gates():
    qreg = LineQubit.range(2)
    # Make a circuit with gates which are not self-inverse
    circuit = Circuit([S.on(qreg[0]), T.on(qreg[1])])

    inverse_circuit = cirq.inverse(circuit)
    inverse_repr = inverse_circuit.__repr__()
    inverse_qasm = inverse_circuit._to_qasm_output().__str__()

    # Simplify the circuit (it should not change this circuit)
    _simplify_circuit_exponents(inverse_circuit)

    # Check inverse_circuit did not change
    simplified_repr = inverse_circuit.__repr__()
    simplified_qasm = inverse_circuit._to_qasm_output().__str__()
    assert simplified_repr == inverse_repr
    assert simplified_qasm == inverse_qasm


def test_are_close_dict():
    """Tests the _are_close_dict function."""
    dict1 = {"a": 1, "b": 0.0}
    dict2 = {"a": 1, "b": 0.0 + 1.0e-10}
    assert _are_close_dict(dict1, dict2)
    assert _are_close_dict(dict2, dict1)
    dict2 = {"b": 0.0 + 1.0e-10, "a": 1}
    assert _are_close_dict(dict1, dict2)
    assert _are_close_dict(dict2, dict1)
    dict2 = {"a": 1, "b": 1.0}
    assert not _are_close_dict(dict1, dict2)
    assert not _are_close_dict(dict2, dict1)
    dict2 = {"b": 1, "a": 0.0}
    assert not _are_close_dict(dict1, dict2)
    assert not _are_close_dict(dict2, dict1)
    dict2 = {"a": 1, "b": 0.0, "c": 1}
    assert not _are_close_dict(dict1, dict2)
    assert not _are_close_dict(dict2, dict1)


def test_max_ent_state_circuit():
    """Tests 1-qubit and 2-qubit maximally entangled states are generated."""
    two_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    four_state = np.array(3 * [1, 0, 0, 0, 0] + [1]) / 2.0
    assert np.allclose(
        _max_ent_state_circuit(2).final_state_vector(), two_state
    )
    assert np.allclose(
        _max_ent_state_circuit(4).final_state_vector(), four_state
    )


def test_circuit_to_choi_and_operation_to_choi():
    """Tests the Choi matrix of a depolarizing channel is recovered."""
    # Define first the expected result
    base_noise = 0.01
    max_ent_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    identity_part = np.outer(max_ent_state, max_ent_state)
    mixed_part = np.eye(4) / 4.0
    epsilon = base_noise * 4.0 / 3.0
    choi = (1.0 - epsilon) * identity_part + epsilon * mixed_part
    # Choi matrix of the double application of a depolarizing channel
    choi_twice = sum(
        [
            ((1.0 - epsilon) ** 2 * identity_part),
            (2 * epsilon - epsilon**2) * mixed_part,
        ]
    )

    # Evaluate the Choi matrix of one or two depolarizing channels
    q = LineQubit(0)
    noisy_operation = depolarize(base_noise).on(q)
    noisy_sequence = [noisy_operation, noisy_operation]
    assert np.allclose(choi, _operation_to_choi(noisy_operation))
    noisy_circuit = Circuit(noisy_operation)
    noisy_circuit_twice = Circuit(noisy_sequence)
    assert np.allclose(choi, _circuit_to_choi(noisy_circuit))
    assert np.allclose(choi_twice, _circuit_to_choi(noisy_circuit_twice))


def test_kronecker_product():
    matrices = [np.array([[1, 2], [3, 4]]), np.array([[0, 1], [1, 0]])]
    expected_result = np.array(
        [[0, 1, 0, 2], [1, 0, 2, 0], [0, 3, 0, 4], [3, 0, 4, 0]]
    )
    np.testing.assert_array_equal(
        matrix_kronecker_product(matrices), expected_result
    )


def test_operator_ptm_vector_rep():
    opt = cirq.I._unitary_() / np.sqrt(2)
    expected_result = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        operator_ptm_vector_rep(opt), expected_result
    )


def test_operator_ptm_vector_rep_raised_error():
    with pytest.raises(TypeError, match="Input must be a square matrix"):
        assert np.allclose(
            operator_ptm_vector_rep(np.array([1.0, 0.0, 0.0, 0.0]))
        )


def test_qem_methods_basic():
    for module, name in qem_methods().items():
        prefix, suffix = module.split(".")
        assert prefix == "mitiq"
        assert len(suffix) <= 3


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_compare_cost_basic(circuit_type):
    q0, q1 = cirq.LineQubit.range(2)
    base = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1),
    )
    qem_circuits = [
        cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1),
        ),
        cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.H(q1),
            cirq.measure(q0, q1),
        ),
    ]
    base_native = convert_from_mitiq(base, circuit_type)
    qem_native = [convert_from_mitiq(c, circuit_type) for c in qem_circuits]
    cost = compare_cost(base_native, qem_native)
    assert cost["extra_circuits"] == 1
    assert cost["gate_overhead"] == {"1q": 2, "2q": 1, "nq": 0}
    assert "shots_per_circuit" not in cost


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_compare_cost_with_shots(circuit_type):
    q0, q1 = cirq.LineQubit.range(2)
    base = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1),
    )
    qem_circuits = [
        cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1),
        ),
        cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.H(q1),
            cirq.measure(q0, q1),
        ),
    ]
    base_native = convert_from_mitiq(base, circuit_type)
    qem_native = [convert_from_mitiq(c, circuit_type) for c in qem_circuits]
    cost = compare_cost(base_native, qem_native, shots=100)
    assert cost["shots_per_circuit"] == 50
