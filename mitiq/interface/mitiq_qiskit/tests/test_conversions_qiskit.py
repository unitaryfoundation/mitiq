# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for conversions between Mitiq circuits and Qiskit circuits."""

import copy

import cirq
import numpy as np
import pytest
import qiskit
from qiskit import qasm2

from mitiq.interface import convert_to_mitiq
from mitiq.interface.mitiq_qiskit.conversions import (
    _add_identity_to_idle,
    _map_bit_index,
    _measurement_order,
    _remove_identity_from_idle,
    _remove_qasm_barriers,
    _transform_registers,
    from_qasm,
    from_qiskit,
    to_qasm,
    to_qiskit,
)
from mitiq.utils import _equal


def _multi_reg_circuits():
    """Returns a circuit with multiple registers
    and an equivalent circuit with a single register."""
    qregs = [qiskit.QuantumRegister(1, f"q{j}") for j in range(4)]
    circuit_multi_reg = qiskit.QuantumCircuit(*qregs)
    for q in qregs[1:]:
        circuit_multi_reg.x(q)
        circuit_multi_reg.x(3)
    qreg = qiskit.QuantumRegister(4, "q")
    circuit_single_reg = qiskit.QuantumCircuit(qreg)
    for q in qreg[1:]:
        circuit_single_reg.x(q)
        circuit_single_reg.x(3)

    return circuit_multi_reg, circuit_single_reg


def test_remove_qasm_barriers():
    assert (
        _remove_qasm_barriers(
            """
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
include "barrier.inc";
include ";barrier.inc";
gate majority a,b,c
{
  cx c,b;
  cx c,a;
  ccx a,b,c;
}
gate barrier1 a,a,a,a,a{
    barrier x,y,z;
    barrier1;
    barrier;
}
gate unmaj a,b,c
{
  ccx a,b,c;
  cx c,a;
  cx a,b;
}
qreg cin[1];
qreg a[4];
qreg b[4];
// barrier;
qreg cout[1]; barrier x,y,z;
creg ans[5];
// set input states
x a[0]; // a = 0001
x b;    // b = 1111
// add a to b, storing result in b
majority cin[0],b[0],a[0];
majority a[0],b[1],a[1];
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
"""
        )
        == """
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
include "barrier.inc";
include ";barrier.inc";
gate majority a,b,c
{
  cx c,b;
  cx c,a;
  ccx a,b,c;
}
gate barrier1 a,a,a,a,a{
    barrier1;
}
gate unmaj a,b,c
{
  ccx a,b,c;
  cx c,a;
  cx a,b;
}
qreg cin[1];
qreg a[4];
qreg b[4];
// barrier;
qreg cout[1];
creg ans[5];
// set input states
x a[0]; // a = 0001
x b;    // b = 1111
// add a to b, storing result in b
majority cin[0],b[0],a[0];
majority a[0],b[1],a[1];
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
"""
    )


def test_bell_state_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qiskit_circuit = to_qiskit(cirq_circuit)  # Qiskit from Cirq
    circuit_cirq = from_qiskit(qiskit_circuit)  # Cirq from Qiskit
    assert _equal(cirq_circuit, circuit_cirq)


def test_bell_state_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
    with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])]
    )
    qasm = to_qasm(cirq_circuit)  # Qasm from Cirq
    circuit_cirq = from_qasm(qasm)
    assert _equal(cirq_circuit, circuit_cirq)


def test_random_circuit_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a random two-qubit circuit.
    """
    cirq_circuit = cirq.testing.random_circuit(
        qubits=2, n_moments=10, op_density=0.99, random_state=1
    )
    qiskit_circuit = to_qiskit(cirq_circuit)
    circuit_cirq = from_qiskit(qiskit_circuit)
    assert cirq.equal_up_to_global_phase(
        cirq_circuit.unitary(), circuit_cirq.unitary()
    )


def test_random_circuit_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
    with a random one-qubit circuit.
    """
    cirq_circuit = cirq.testing.random_circuit(
        qubits=2, n_moments=10, op_density=0.99, random_state=2
    )
    qasm = to_qasm(cirq_circuit)
    circuit_cirq = from_qasm(qasm)
    assert cirq.equal_up_to_global_phase(
        cirq_circuit.unitary(), circuit_cirq.unitary()
    )


def test_convert_with_qft():
    """Tests converting a Qiskit circuit with a QFT to a Cirq circuit."""
    circuit = qiskit.QuantumCircuit(1)
    circuit &= qiskit.circuit.library.QFT(1)
    circuit.measure_all()
    qft_cirq = from_qiskit(circuit)
    qreg = cirq.LineQubit.range(1)
    cirq_circuit = cirq.Circuit(
        [cirq.ops.H.on(qreg[0]), cirq.ops.measure(qreg[0], key="meas")]
    )
    assert cirq.equal_up_to_global_phase(
        cirq_circuit.unitary(), qft_cirq.unitary()
    )


def test_convert_qiskit_reset_to_cirq():
    """Tests converting a Qiskit circuit with a reset to a Cirq circuit."""
    qreg = qiskit.QuantumRegister(1)
    circuit = qiskit.QuantumCircuit(qreg)
    circuit.reset(qreg[0])
    cirq_circuit = from_qiskit(circuit)
    assert len(cirq_circuit) == 1
    assert isinstance(cirq_circuit[0].operations[0].gate, cirq.ResetChannel)


@pytest.mark.parametrize("as_qasm", (True, False))
def test_convert_with_barrier(as_qasm):
    """Tests converting a Qiskit circuit with a barrier to a Cirq circuit."""
    n = 5
    qiskit_circuit = qiskit.QuantumCircuit(qiskit.QuantumRegister(n))
    qiskit_circuit.barrier()

    if as_qasm:
        cirq_circuit = from_qasm(qasm2.dumps(qiskit_circuit))
    else:
        cirq_circuit = from_qiskit(qiskit_circuit)

    assert _equal(cirq_circuit, cirq.Circuit())


@pytest.mark.parametrize("as_qasm", (True, False))
def test_convert_with_multiple_barriers(as_qasm):
    """Tests converting a Qiskit circuit with barriers to a Cirq circuit."""
    n = 1
    num_ops = 10

    qreg = qiskit.QuantumRegister(n)
    qiskit_circuit = qiskit.QuantumCircuit(qreg)
    for _ in range(num_ops):
        qiskit_circuit.h(qreg)
        qiskit_circuit.barrier()

    if as_qasm:
        cirq_circuit = from_qasm(qasm2.dumps(qiskit_circuit))
    else:
        cirq_circuit = from_qiskit(qiskit_circuit)

    qbit = cirq.LineQubit(0)
    correct = cirq.Circuit(cirq.ops.H.on(qbit) for _ in range(num_ops))
    assert _equal(cirq_circuit, correct)


@pytest.mark.parametrize("reg_sizes", [[2, 4, 1, 6], [5, 4, 2], [6]])
def test_map_bit_index(reg_sizes):
    expected_register_index = 0
    expected_mapped_index = 0
    for bit_index in range(sum(reg_sizes)):
        register_index, mapped_index = _map_bit_index(bit_index, reg_sizes)

        assert register_index == expected_register_index
        assert mapped_index == expected_mapped_index

        expected_mapped_index += 1
        if bit_index == sum(reg_sizes[: expected_register_index + 1]) - 1:
            expected_register_index += 1
            expected_mapped_index = 0


@pytest.mark.parametrize("nqubits", [1, 5])
@pytest.mark.parametrize("with_ops", [True, False])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_qregs_one_qubit_ops(nqubits, with_ops, measure):
    qreg = qiskit.QuantumRegister(nqubits)
    circ = qiskit.QuantumCircuit(qreg)
    if with_ops:
        circ.h(qreg)
    if measure:
        circ.add_register(qiskit.ClassicalRegister(nqubits))
        circ.measure(qreg, circ.cregs[0])

    orig = circ.copy()
    assert circ.qregs == [qreg]

    new_qregs = [qiskit.QuantumRegister(1) for _ in range(nqubits)]
    circ = _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert circ.cregs == orig.cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("nqubits", [1, 5])
@pytest.mark.parametrize("with_ops", [True, False])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_circuit_with_multiple_qregs(nqubits, with_ops, measure):
    qreg_1 = qiskit.QuantumRegister(nqubits)
    qreg_2 = qiskit.QuantumRegister(nqubits)
    circ = qiskit.QuantumCircuit(qreg_1, qreg_2)
    if with_ops:
        circ.h(qreg_1)
        circ.h(qreg_2)
    if measure:
        circ.add_register(qiskit.ClassicalRegister(nqubits))
        circ.measure(qreg_1, circ.cregs[0])
        circ.measure(qreg_2, circ.cregs[0])

    orig = circ.copy()
    assert circ.qregs == [qreg_1, qreg_2]

    new_qregs_1 = [qiskit.QuantumRegister(1) for _ in range(nqubits)]
    new_qregs_2 = [qiskit.QuantumRegister(1) for _ in range(nqubits)]
    circ = _transform_registers(circ, new_qregs=new_qregs_1 + new_qregs_2)

    assert circ.qregs == new_qregs_1 + new_qregs_2
    assert circ.cregs == orig.cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("new_reg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
def test_transform_qregs_two_qubit_ops(new_reg_sizes):
    nqubits = sum(new_reg_sizes)
    circ = to_qiskit(
        cirq.testing.random_circuit(
            nqubits, n_moments=5, op_density=1, random_state=1
        )
    )
    orig = circ.copy()

    new_qregs = [qiskit.QuantumRegister(s) for s in new_reg_sizes]
    circ = _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert circ.cregs == orig.cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("new_reg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_qregs_random_circuit(new_reg_sizes, measure):
    nbits = sum(new_reg_sizes)
    circ = to_qiskit(
        cirq.testing.random_circuit(
            nbits, n_moments=5, op_density=1, random_state=10
        )
    )
    creg = qiskit.ClassicalRegister(nbits)
    circ.add_register(creg)
    if measure:
        circ.measure(circ.qregs[0], creg)
    orig = circ.copy()

    new_qregs = [qiskit.QuantumRegister(s) for s in new_reg_sizes]
    circ = _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


def test_transform_qregs_no_new_qregs():
    qreg = qiskit.QuantumRegister(5)
    circ = qiskit.QuantumCircuit(qreg)
    circ = _transform_registers(circ, new_qregs=None)
    assert circ.qregs == [qreg]


def test_transform_registers_too_few_qubits():
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(2))
    new_qregs = [qiskit.QuantumRegister(1)]

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs=new_qregs)


def test_transform_registers_adds_idle_qubits():
    """Tests transforming registers in a circuit with n qubits to a circuit
    with m > n qubits.
    """
    qreg = qiskit.QuantumRegister(1)
    creg = qiskit.ClassicalRegister(1)
    circuit = qiskit.QuantumCircuit(qreg, creg)
    circuit.x(qreg[0])
    circuit.measure(qreg[0], creg[0])

    assert len(circuit.qregs) == 1
    assert circuit.num_qubits == 1
    old_data = copy.deepcopy(circuit.data)

    circuit = _transform_registers(
        circuit, new_qregs=[qreg, qiskit.QuantumRegister(4)]
    )

    assert len(circuit.qregs) == 2
    assert circuit.num_qubits == 5
    assert circuit.data == old_data


def test_transform_registers_wrong_reg_number():
    nqubits = 2
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(nqubits))
    new_qregs = [qiskit.QuantumRegister(1) for _ in range(2 * nqubits)]
    circ.add_register(*new_qregs)

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs=new_qregs)


@pytest.mark.parametrize("size", [5])
def test_measurement_order(size):
    q, c = qiskit.QuantumRegister(size), qiskit.ClassicalRegister(size)
    circuit = qiskit.QuantumCircuit(q, c)

    index_order = [int(i) for i in np.random.RandomState(1).permutation(size)]
    for i in index_order:
        circuit.measure(q[i], c[i])

    order = _measurement_order(circuit)
    assert order == [(q[i], c[i]) for i in index_order]


def test_add_identity_to_idle():
    circuit = qiskit.QuantumCircuit(9)
    circuit.x([0, 8])
    circuit.cx(0, 8)
    expected_idle_qubits = circuit.qubits[1:-1]

    idle_qubits = _add_identity_to_idle(circuit)
    id_qubits = []
    for op in circuit.get_instructions("id"):
        qubits = op.qubits
        for qubit in qubits:
            id_qubits.append(qubit)
    assert idle_qubits == set(expected_idle_qubits)
    assert set(id_qubits) == idle_qubits


def test_remove_identity_from_idle():
    idle_indices = set(range(1, 8))
    circuit = qiskit.QuantumCircuit(9)
    circuit.x([0, 8])
    circuit.cx(0, 8)
    _remove_identity_from_idle(circuit, idle_indices)
    id_indices = []
    for op in circuit.get_instructions("id"):
        qubits = op.qubits
        for qubit in qubits:
            id_indices.append(qubit.index)
    assert id_indices == []


def test_add_identity_to_idle_with_multiple_registers():
    """Tests idle qubits are correctly detected even with many registers."""
    circuit_multi_reg, circuit_single_reg = _multi_reg_circuits()
    _add_identity_to_idle(circuit_multi_reg)
    _add_identity_to_idle(circuit_single_reg)

    # The result should be the same for both types of registers
    assert _equal(
        convert_to_mitiq(circuit_multi_reg)[0],
        convert_to_mitiq(circuit_single_reg)[0],
        require_qubit_equality=False,  # Qubit names can be different
    )


def test_remove_identity_from_idle_with_multiple_registers():
    """Tests identities are correctly removed even with many registers."""
    circuit_multi_reg, circuit_single_reg = _multi_reg_circuits()

    idle_qubits_multi = _add_identity_to_idle(circuit_multi_reg.copy())
    idle_qubits_single = _add_identity_to_idle(circuit_single_reg.copy())

    _remove_identity_from_idle(circuit_multi_reg, idle_qubits_multi)
    _remove_identity_from_idle(circuit_single_reg, idle_qubits_single)

    # Adding and removing identities should preserve the input
    input_multi, input_single = _multi_reg_circuits()
    assert circuit_multi_reg == input_multi
    assert circuit_single_reg == input_single


def test_convert_to_mitiq_with_rx_and_rzz():
    """Tests that convert_to_mitiq works with RX and RZZ gates."""
    test_qc = qiskit.QuantumCircuit(2)
    test_qc.rx(0.1, 0)
    test_qc.rzz(0.1, 0, 1)
    assert convert_to_mitiq(test_qc)


def test_convert_to_mitiq_with_rx_and_ryy():
    """
    Tests that convert_to_mitiq works with RX and RYY gates.
    """
    test_qc = qiskit.QuantumCircuit(2)
    test_qc.rx(0.1, 0)
    test_qc.ryy(0.1, 0, 1)
    assert convert_to_mitiq(test_qc)


def test_convert_to_mitiq_with_sx():
    """
    Tests that convert_to_mitiq works with SX gates.
    """
    test_qc = qiskit.QuantumCircuit(1)
    test_qc.sx(0)
    assert convert_to_mitiq(test_qc)


def test_convert_to_mitiq_with_u():
    """
    Tests that convert_to_mitiq works with U gates.
    """

    test_qc = qiskit.QuantumCircuit(1)
    test_qc.u(0.1, 0.2, 0.3, 0)
    assert convert_to_mitiq(test_qc)


def test_convert_to_mitiq_with_p():
    """
    Tests that convert_to_mitiq works with P gates.
    """
    circuit = qiskit.QuantumCircuit(1)
    circuit.p(np.pi / 4, 0)

    assert convert_to_mitiq(circuit)


def test_convert_to_mitiq_with_cu1():
    """
    Tests that convert_to_mitiq works with CU1 gates.
    """
    test_qc = qiskit.QuantumCircuit(2)
    test_qc.h(0)
    test_qc.h(1)
    cu1_gate = qiskit.circuit.library.CU1Gate(np.pi / 4)
    test_qc.append(cu1_gate, [0, 1])
    assert convert_to_mitiq(test_qc)


def test_convert_to_mitiq_with_ecrgate():
    """
    Tests that convert_to_mitiq works with ECR gates.
    """
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.h(1)
    circuit.append(qiskit.circuit.library.ECRGate(), [0, 1])
    assert convert_to_mitiq(circuit)


def test_convert_to_mitiq_with_rxx_rzz_ecr():
    """
    Tests that convert_to_mitiq works with RXX, RZZ, and ECR gates.
    """
    circuit = qiskit.QuantumCircuit(2)
    circuit.sx(0)
    circuit.append(qiskit.circuit.library.RXXGate(np.pi / 3), [0, 1])
    circuit.append(qiskit.circuit.library.RZZGate(np.pi / 4), [0, 1])
    circuit.append(qiskit.circuit.library.ECRGate(), [0, 1])
    assert convert_to_mitiq(circuit)


def test_convert_to_mitiq_with_rzx_ryy_p():
    """
    Tests that convert_to_mitiq works with RZX, RYY, and P gates.
    """
    rotation_circuit = qiskit.QuantumCircuit(2)
    rotation_circuit.p(np.pi / 8, 0)
    rotation_circuit.append(qiskit.circuit.library.RZXGate(np.pi / 6), [0, 1])
    rotation_circuit.append(qiskit.circuit.library.RYYGate(np.pi / 5), [0, 1])
    assert convert_to_mitiq(rotation_circuit)


def test_convert_to_mitiq_with_qft_cu1_rzx():
    """
    Tests that convert_to_mitiq works with QFT, CU1, and RZX gates.
    """
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.h(1)
    circuit.append(qiskit.circuit.library.QFT(2), [0, 1])
    circuit.append(qiskit.circuit.library.CU1Gate(np.pi / 3), [0, 1])
    circuit.append(qiskit.circuit.library.RZXGate(np.pi / 4), [0, 1])
    assert convert_to_mitiq(circuit)


def test_convert_to_mitiq_with_rzz_u_p_ecr():
    """
    Tests that convert_to_mitiq works with RZZ, U, P, and ECR gates.
    """
    circuit = qiskit.QuantumCircuit(2)
    circuit.append(qiskit.circuit.library.RZZGate(np.pi / 4), [0, 1])
    circuit.u(0.1, 0.2, 0.3, 0)
    circuit.p(np.pi / 4, 0)
    circuit.append(qiskit.circuit.library.ECRGate(), [0, 1])
    circuit.append(qiskit.circuit.library.RZZGate(np.pi / 2), [0, 1])
    assert convert_to_mitiq(circuit)


def test_convert_to_mitiq_with_rxx_ryy_sx_cu1():
    """
    Tests that convert_to_mitiq works with RXX, RYY, SX, and CU1 gates.
    """
    circuit = qiskit.QuantumCircuit(2)
    circuit.sx(0)
    circuit.append(qiskit.circuit.library.RXXGate(np.pi / 4), [0, 1])
    circuit.append(qiskit.circuit.library.RYYGate(np.pi / 6), [0, 1])
    circuit.append(qiskit.circuit.library.CU1Gate(np.pi / 8), [0, 1])
    circuit.u(0.5, 0.7, 0.2, 0)
    assert convert_to_mitiq(circuit)


def test_convert_to_mitiq_with_custom_operator():
    """
    Tests that convert_to_mitiq works with a custom operator.
    """
    gate = qiskit.quantum_info.Operator([[0.0, 1.0], [-1.0, 0.0]])
    qreg = qiskit.QuantumRegister(1)
    circ = qiskit.QuantumCircuit(qreg)
    circ.unitary(gate, [0])
    assert convert_to_mitiq(circ)
