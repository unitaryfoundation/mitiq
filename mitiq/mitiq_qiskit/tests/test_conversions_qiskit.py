# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for conversions between Mitiq circuits and Qiskit circuits."""
import numpy as np
import pytest

import cirq
import qiskit

from mitiq.utils import _equal
from mitiq.mitiq_qiskit.conversions import (
    to_qasm,
    to_qiskit,
    from_qasm,
    from_qiskit,
    _map_bit_index,
    _transform_registers,
    _measurement_order,
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


@pytest.mark.parametrize("as_qasm", (True, False))
def test_convert_with_barrier(as_qasm):
    """Tests converting a Qiskit circuit with a barrier to a Cirq circuit."""
    n = 5
    qiskit_circuit = qiskit.QuantumCircuit(qiskit.QuantumRegister(n))
    qiskit_circuit.barrier()

    if as_qasm:
        cirq_circuit = from_qasm(qiskit_circuit.qasm())
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
        cirq_circuit = from_qasm(qiskit_circuit.qasm())
    else:
        cirq_circuit = from_qiskit(qiskit_circuit)

    qbit = cirq.LineQubit(0)
    correct = cirq.Circuit(cirq.ops.H.on(qbit) for _ in range(num_ops))
    assert _equal(cirq_circuit, correct)


@pytest.mark.parametrize("qreg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
def test_to_qiskit_assign_qregs(qreg_sizes):
    nbits = sum(qreg_sizes)
    cirq_circuit = cirq.testing.random_circuit(
        nbits, n_moments=5, op_density=1, random_state=10
    )

    qregs = [qiskit.QuantumRegister(s) for s in qreg_sizes]
    qiskit_circuit = to_qiskit(cirq_circuit, qregs=qregs)

    assert qiskit_circuit.qregs == qregs
    assert qiskit_circuit.cregs == []
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(from_qiskit(qiskit_circuit)),
        cirq.unitary(cirq_circuit),
        atol=1e-5,
    )


@pytest.mark.parametrize("qreg_sizes", [[2], [1, 3, 2], [2, 1], [1, 1, 1]])
@pytest.mark.parametrize("measure", [True, False])
@pytest.mark.parametrize("flip_creg", [True, False])
def test_to_qiskit_assign_qregs_and_cregs(qreg_sizes, measure, flip_creg):
    nbits = sum(qreg_sizes)
    cirq_circuit = cirq.testing.random_circuit(
        nbits, n_moments=5, op_density=1, random_state=10
    )
    if measure:
        cirq_circuit.append(cirq.measure_each(*cirq_circuit.all_qubits()))

    qregs = [qiskit.QuantumRegister(s) for s in qreg_sizes]
    cregs = [qiskit.ClassicalRegister(s) for s in qreg_sizes]
    if flip_creg:
        cregs = cregs[::-1]

    qiskit_circuit = to_qiskit(cirq_circuit, qregs=qregs, cregs=cregs)

    assert qiskit_circuit.qregs == qregs
    assert qiskit_circuit.cregs == cregs

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(from_qiskit(qiskit_circuit)),
        cirq.unitary(cirq_circuit),
        atol=1e-5,
    )


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
    _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
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
    _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert circ.cregs == orig.cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("nbits", [1, 5])
@pytest.mark.parametrize("with_ops", [True, False])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_cregs(nbits, with_ops, measure):
    qreg = qiskit.QuantumRegister(nbits)
    creg = qiskit.ClassicalRegister(nbits)
    circ = qiskit.QuantumCircuit(qreg, creg)
    if with_ops:
        circ.h(qreg)
    if measure:
        circ.measure(qreg, creg)

    orig = circ.copy()

    new_cregs = [qiskit.ClassicalRegister(1) for _ in range(nbits)]
    _transform_registers(circ, new_cregs=new_cregs)

    assert circ.cregs == new_cregs
    assert circ.qregs == orig.qregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("new_reg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_qregs_and_cregs_random_circuit(new_reg_sizes, measure):
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
    new_cregs = [qiskit.ClassicalRegister(s) for s in new_reg_sizes]
    _transform_registers(circ, new_qregs=new_qregs, new_cregs=new_cregs)

    assert circ.qregs == new_qregs
    assert circ.cregs == new_cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


def test_transform_registers_wrong_bit_number():
    nqubits = 2
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(nqubits))
    new_qregs = [qiskit.QuantumRegister(1) for _ in range(2 * nqubits)]

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs=new_qregs)

    new_cregs = [qiskit.ClassicalRegister(1)]
    with pytest.raises(ValueError):
        _transform_registers(circ, new_cregs=new_cregs)

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs, new_cregs)


@pytest.mark.parametrize("size", [5])
def test_measurement_order(size):
    q, c = qiskit.QuantumRegister(size), qiskit.ClassicalRegister(size)
    circuit = qiskit.QuantumCircuit(q, c)

    index_order = [int(i) for i in np.random.RandomState(1).permutation(size)]
    for i in index_order:
        circuit.measure(q[i], c[i])

    order = _measurement_order(circuit)
    assert order == [(q[i], c[i]) for i in index_order]
