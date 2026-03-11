# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import numpy as np
import pyquil
import pytest
import qiskit
from cirq import Circuit

from mitiq.pec.types import NoisyBasis, NoisyOperation, OperationRepresentation
from mitiq.utils import _equal

icirq = Circuit(cirq.I(cirq.LineQubit(0)))
xcirq = Circuit(cirq.X(cirq.LineQubit(0)))
ycirq = Circuit(cirq.Y(cirq.LineQubit(0)))
zcirq = Circuit(cirq.Z(cirq.LineQubit(0)))
hcirq = Circuit(cirq.H(cirq.LineQubit(0)))
cnotcirq = Circuit(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))


def test_init_with_cirq_circuit():
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation(zcirq, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)

    assert noisy_op.qubits == (cirq.LineQubit(0),)
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(zcirq, noisy_op.circuit)
    assert _equal(noisy_op._native_circuit, noisy_op.circuit)


@pytest.mark.parametrize(
    "qubit",
    (cirq.LineQubit(0), cirq.GridQubit(1, 2), cirq.NamedQubit("Qubit")),
)
def test_init_with_different_qubits(qubit):
    ideal_op = Circuit(cirq.H.on(qubit))
    real = np.zeros(shape=(4, 4))
    noisy_op = NoisyOperation(ideal_op, real)

    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(
        noisy_op.circuit,
        cirq.Circuit(ideal_op),
        require_qubit_equality=True,
    )
    assert noisy_op.qubits == (qubit,)
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real

    assert noisy_op._native_type == "cirq"
    assert _equal(cirq.Circuit(ideal_op), noisy_op.circuit)
    assert _equal(noisy_op._native_circuit, noisy_op.circuit)


def test_init_with_cirq_input():
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(cirq.H.on(qreg[0]), cirq.CNOT.on(*qreg))
    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)

    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op.circuit, circ, require_qubit_equality=True)
    assert set(noisy_op.qubits) == set(qreg)
    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


def test_init_with_qiskit_circuit():
    qreg = qiskit.QuantumRegister(2)
    circ = qiskit.QuantumCircuit(qreg)
    _ = circ.h(qreg[0])
    _ = circ.cx(*qreg)

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op._circuit, cirq_circ)
    assert _equal(noisy_op.circuit, cirq_circ)

    assert noisy_op.native_circuit == circ
    assert noisy_op._native_circuit == circ
    assert noisy_op._native_type == "qiskit"

    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


@pytest.mark.parametrize(
    "gate",
    (
        cirq.H,
        cirq.H(cirq.LineQubit(0)),
        qiskit.circuit.library.HGate,
        qiskit.circuit.library.CHGate,
        pyquil.gates.H,
    ),
)
def test_init_with_gates_raises_error(gate):
    rng = np.random.RandomState(seed=1)
    with pytest.raises(TypeError, match="Failed to convert to an internal"):
        NoisyOperation(circuit=gate, channel_matrix=rng.rand(4, 4))


def test_init_with_pyquil_program():
    circ = pyquil.Program(pyquil.gates.H(0), pyquil.gates.CNOT(0, 1))

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    real = np.zeros(shape=(16, 16))
    noisy_op = NoisyOperation(circ, real)
    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op._circuit, cirq_circ)
    assert _equal(noisy_op.circuit, cirq_circ)

    assert noisy_op.native_circuit == circ
    assert noisy_op._native_circuit == circ
    assert noisy_op._native_type == "pyquil"

    assert np.allclose(noisy_op.channel_matrix, real)
    assert noisy_op.channel_matrix is not real


def test_init_dimension_mismatch_error():
    ideal = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    real = np.zeros(shape=(3, 3))
    with pytest.raises(ValueError, match="has shape"):
        NoisyOperation(ideal, real)


def test_unknown_channel_matrix():
    qreg = qiskit.QuantumRegister(2)
    circ = qiskit.QuantumCircuit(qreg)
    _ = circ.h(qreg[0])
    _ = circ.cx(*qreg)

    cirq_qreg = cirq.LineQubit.range(2)
    cirq_circ = cirq.Circuit(cirq.H.on(cirq_qreg[0]), cirq.CNOT.on(*cirq_qreg))

    noisy_op = NoisyOperation(circ)
    assert isinstance(noisy_op._circuit, cirq.Circuit)
    assert _equal(noisy_op._circuit, cirq_circ)
    assert _equal(noisy_op.circuit, cirq_circ)

    assert noisy_op.native_circuit == circ
    assert noisy_op._native_circuit == circ
    assert noisy_op._native_type == "qiskit"

    with pytest.raises(ValueError, match="The channel matrix is unknown."):
        _ = noisy_op.channel_matrix


def test_add_simple():
    circuit1 = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))])
    circuit2 = cirq.Circuit([cirq.Y.on(cirq.NamedQubit("Q"))])

    super_op1 = np.random.rand(4, 4)
    super_op2 = np.random.rand(4, 4)

    noisy_op1 = NoisyOperation(circuit1, super_op1)
    noisy_op2 = NoisyOperation(circuit2, super_op2)

    noisy_op = noisy_op1 + noisy_op2

    correct = cirq.Circuit(
        [cirq.X.on(cirq.NamedQubit("Q")), cirq.Y.on(cirq.NamedQubit("Q"))],
    )

    assert _equal(noisy_op._circuit, correct, require_qubit_equality=True)
    assert np.allclose(noisy_op.channel_matrix, super_op2 @ super_op1)


def test_add_pyquil_noisy_operations():
    ideal = pyquil.Program(pyquil.gates.X(0))
    real = np.random.rand(4, 4)

    noisy_op1 = NoisyOperation(ideal, real)
    noisy_op2 = NoisyOperation(ideal, real)

    noisy_op = noisy_op1 + noisy_op2

    correct = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))] * 2)

    assert _equal(noisy_op._circuit, correct, require_qubit_equality=False)
    assert np.allclose(noisy_op.channel_matrix, real @ real)


def test_add_qiskit_noisy_operations():
    qreg = qiskit.QuantumRegister(1)
    ideal = qiskit.QuantumCircuit(qreg)
    _ = ideal.x(qreg)
    real = np.random.rand(4, 4)

    noisy_op1 = NoisyOperation(ideal, real)
    noisy_op2 = NoisyOperation(ideal, real)

    noisy_op = noisy_op1 + noisy_op2

    correct = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))] * 2)

    assert _equal(noisy_op._circuit, correct, require_qubit_equality=False)
    assert np.allclose(noisy_op.channel_matrix, real @ real)


def test_add_bad_type():
    ideal = cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))])
    real = np.random.rand(4, 4)

    noisy_op = NoisyOperation(ideal, real)

    with pytest.raises(ValueError, match="must be a NoisyOperation"):
        noisy_op + ideal


def test_add_noisy_operation_no_channel_matrix():
    noisy_op1 = NoisyOperation(cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))]))
    noisy_op2 = NoisyOperation(
        cirq.Circuit([cirq.X.on(cirq.NamedQubit("Q"))]),
        channel_matrix=np.random.rand(4, 4),
    )

    with pytest.raises(ValueError):
        (noisy_op1 + noisy_op2).channel_matrix


def test_noisy_operation_str():
    noisy_op = NoisyOperation(circuit=icirq, channel_matrix=np.identity(4))
    assert isinstance(noisy_op.__str__(), str)


def test_noisy_basis_deprecation_error():
    with pytest.raises(NotImplementedError, match="NoisyBasis"):
        NoisyBasis()
    with pytest.raises(NotImplementedError, match="NoisyBasis"):
        NoisyBasis(zcirq, xcirq)


def get_test_representation():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation(
        circuit=zcirq, channel_matrix=np.zeros(shape=(4, 4))
    )

    decomp = OperationRepresentation(
        ideal,
        [noisy_xop, noisy_zop],
        [0.5, -0.5],
    )
    return ideal, noisy_xop, noisy_zop, decomp


def test_representation_simple():
    ideal, noisy_xop, noisy_zop, decomp = get_test_representation()

    assert _equal(decomp.ideal, ideal)
    assert decomp.coeffs == [0.5, -0.5]
    assert np.allclose(decomp.distribution, np.array([0.5, 0.5]))
    assert np.isclose(decomp.norm, 1.0)
    assert isinstance(decomp.basis_expansion[0][0], float)
    assert set(decomp.noisy_operations) == {noisy_xop, noisy_zop}


def test_representation_bad_type():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )

    with pytest.raises(TypeError, match="All elements of `noisy_operations`"):
        OperationRepresentation(
            ideal=ideal,
            noisy_operations=[0.1],
            coeffs=[0.1],
        )
    with pytest.raises(TypeError, match="All elements of `coeffs` must"):
        OperationRepresentation(
            ideal=ideal,
            noisy_operations=[noisy_xop],
            coeffs=["x"],
        )


def test_representation_sample():
    _, noisy_xop, noisy_zop, decomp = get_test_representation()

    for _ in range(10):
        noisy_op, sign, coeff = decomp.sample()
        assert sign in (-1, 1)
        assert coeff in (-0.5, 0.5)
        assert noisy_op in (noisy_xop, noisy_zop)
        case_one = noisy_op == noisy_xop and coeff == 0.5
        case_two = noisy_op == noisy_zop and coeff == -0.5
        assert case_one or case_two


def test_representation_sample_seed():
    _, noisy_xop, noisy_zop, decomp = get_test_representation()

    seed1 = np.random.RandomState(seed=1)
    seed2 = np.random.RandomState(seed=1)
    for _ in range(10):
        _, sign1, coeff1 = decomp.sample(random_state=seed1)
        _, sign2, coeff2 = decomp.sample(random_state=seed2)

        assert sign1 == sign2
        assert np.isclose(coeff1, coeff2)


def test_representation_sample_bad_seed_type():
    _, _, _, decomp = get_test_representation()
    with pytest.raises(TypeError, match="should be of type"):
        decomp.sample(random_state="8")


def test_representation_sample_zero_coefficient():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation(
        circuit=zcirq, channel_matrix=np.zeros(shape=(4, 4))
    )

    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop, noisy_zop],
        coeffs=[0.5, 0.0],  # 0 term should never be sampled.
    )

    random_state = np.random.RandomState(seed=1)
    for _ in range(500):
        noisy_op, sign, coeff = decomp.sample(random_state=random_state)
        assert sign == 1
        assert coeff == 0.5
        assert np.allclose(
            cirq.unitary(noisy_op.circuit),
            cirq.unitary(cirq.X),
        )


def test_print_cirq_operation_representation():
    ideal = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

    noisy_xop = NoisyOperation(
        circuit=xcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    noisy_zop = NoisyOperation(
        circuit=zcirq, channel_matrix=np.zeros(shape=(4, 4))
    )
    # Positive first coefficient
    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop, noisy_zop],
        coeffs=[0.5, 0.5],
    )
    expected = r"0: ───H─── = 0.500*(0: ───X───)+0.500*(0: ───Z───)"
    assert str(decomp) == expected
    # Negative first coefficient
    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop, noisy_zop],
        coeffs=[-0.5, 1.5],
    )
    expected = r"0: ───H─── = -0.500*(0: ───X───)+1.500*(0: ───Z───)"
    assert str(decomp) == expected
    # Empty representation
    decomp = OperationRepresentation(ideal, [], [])
    expected = r"0: ───H─── = 0.000"
    assert str(decomp) == expected
    # Small coefficient approximation
    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop, noisy_zop],
        coeffs=[1.00001, 0.00001],
    )
    expected = r"0: ───H─── = 1.000*(0: ───X───)"
    assert str(decomp) == expected
    # Small coefficient approximation different position
    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop, noisy_zop],
        coeffs=[0.00001, 1.00001],
    )
    expected = r"0: ───H─── = 1.000*(0: ───Z───)"
    # Small coefficient approximation different position
    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop],
        coeffs=[0.00001],
    )
    expected = r"0: ───H─── = 0.000"
    assert str(decomp) == expected


def test_print_operation_representation_two_qubits():
    qreg = cirq.LineQubit.range(2)
    ideal = cirq.Circuit(cirq.CNOT(*qreg))

    noisy_a = NoisyOperation(
        circuit=cirq.Circuit(
            cirq.H.on_each(qreg), cirq.CNOT(*qreg), cirq.H.on_each(qreg)
        )
    )
    noisy_b = NoisyOperation(
        circuit=cirq.Circuit(
            cirq.Z.on_each(qreg),
            cirq.CNOT(*qreg),
            cirq.Z.on_each(qreg),
        )
    )
    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_a, noisy_b],
        coeffs=[0.5, 0.5],
    )
    expected = f"""
0: ───@───
      │
1: ───X─── ={" "}

0.500
0: ───H───@───H───
          │
1: ───H───X───H───

+0.500
0: ───Z───@───Z───
          │
1: ───Z───X───Z───"""
    # Remove initial newline
    expected = expected[1:]
    assert str(decomp) == expected


def test_print_operation_representation_two_qubits_neg():
    qreg = cirq.LineQubit.range(2)
    ideal = cirq.Circuit(cirq.CNOT(*qreg))

    noisy_a = NoisyOperation(
        circuit=cirq.Circuit(
            cirq.H.on_each(qreg[0]), cirq.CNOT(*qreg), cirq.H.on_each(qreg[1])
        )
    )
    noisy_b = NoisyOperation(circuit=cirq.Circuit(cirq.Z.on_each(*qreg)))
    decomp = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_a, noisy_b],
        coeffs=[-0.5, 1.5],
    )
    expected = f"""
0: ───@───
      │
1: ───X─── ={" "}

-0.500
0: ───H───@───────
          │
1: ───────X───H───

+1.500
0: ───Z───

1: ───Z───"""

    # Remove initial newline
    expected = expected[1:]
    assert str(decomp) == expected


def test_equal_method_of_representations():
    q = cirq.LineQubit(0)
    ideal = cirq.Circuit(cirq.H(q))
    noisy_xop_a = NoisyOperation(
        circuit=cirq.Circuit(cirq.X(q)),
        channel_matrix=np.zeros(shape=(4, 4)),
    )
    noisy_zop_a = NoisyOperation(
        circuit=cirq.Circuit(cirq.Z(q)),
        channel_matrix=np.zeros(shape=(4, 4)),
    )
    rep_a = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop_a, noisy_zop_a],
        coeffs=[0.5, 0.5],
    )
    noisy_xop_b = NoisyOperation(
        circuit=cirq.Circuit(cirq.X(q)),
        channel_matrix=np.ones(shape=(4, 4)),
    )
    noisy_zop_b = NoisyOperation(
        circuit=cirq.Circuit(cirq.Z(q)),
        channel_matrix=np.ones(shape=(4, 4)),
    )
    rep_b = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop_b, noisy_zop_b],
        coeffs=[0.5, 0.5],
    )
    # Equal representation up to real superoperators
    assert rep_a == rep_b
    # Different ideal
    ideal_b = cirq.Circuit(cirq.X(q))
    rep_b = OperationRepresentation(
        ideal=ideal_b,
        noisy_operations=[noisy_xop_b, noisy_zop_b],
        coeffs=[0.5, 0.5],
    )
    assert rep_a != rep_b
    # Different type
    q_b = qiskit.QuantumRegister(1)
    ideal_b = qiskit.QuantumCircuit(q_b)
    ideal_b.x(q_b)
    noisy_opx = NoisyOperation(ideal_b)
    rep_b = OperationRepresentation(
        ideal=ideal_b,
        noisy_operations=[noisy_opx, noisy_opx],
        coeffs=[0.5, 0.5],
    )
    assert rep_a != rep_b
    # Different length
    rep_b = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop_b],
        coeffs=[0.5],
    )
    assert rep_a != rep_b
    # Different operations
    noisy_diff = NoisyOperation(circuit=cirq.Circuit(cirq.H(q)))
    rep_b = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop_b, noisy_diff],
        coeffs=[0.5, 0.5],
    )
    assert rep_a != rep_b
    # Different coefficients
    rep_b = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop_b, noisy_zop_b],
        coeffs=[0.7, 0.5],
    )
    assert rep_a != rep_b
    # Different value of is_qubit_dependent
    rep_b = OperationRepresentation(
        ideal=ideal,
        noisy_operations=[noisy_xop_a, noisy_zop_a],
        coeffs=[0.5, 0.5],
        is_qubit_dependent=False,
    )
    assert rep_a != rep_b


def test_operation_representation_warnings():
    with pytest.warns(UserWarning, match="different from 1"):
        OperationRepresentation(
            ideal=xcirq,
            noisy_operations=[NoisyOperation(xcirq), NoisyOperation(zcirq)],
            coeffs=[0.5, 0.1],
        )


def test_different_qubits_error():
    """Ideal operation and noisy operations must have equal qubits."""

    with pytest.raises(ValueError, match="must act on the same qubits"):
        OperationRepresentation(
            ideal=cirq.Circuit(cirq.X(cirq.NamedQubit("a"))),
            noisy_operations=[NoisyOperation(xcirq), NoisyOperation(zcirq)],
            coeffs=[0.5, 0.5],
        )


def test_different_length_error():
    """The number of coefficients must be equal to the number of noisy
    operations.
    """
    with pytest.raises(ValueError, match="must have equal length"):
        OperationRepresentation(
            ideal=cirq.Circuit(cirq.X(cirq.LineQubit(0))),
            noisy_operations=[NoisyOperation(xcirq), NoisyOperation(zcirq)],
            coeffs=[0.5, 0.5, 0.4],
        )
