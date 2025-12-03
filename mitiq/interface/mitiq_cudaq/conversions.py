# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
Cuda-Quantum's circuit representation.
"""

import cudaq
import numpy as np
from cirq import Circuit
from cudaq import PyKernel
from qbraid.transpiler.conversions.qasm2 import qasm2_to_cirq, qasm2_to_qiskit
from qbraid.transpiler.conversions.qiskit import qiskit_to_qasm2
from qiskit import QuantumCircuit, transpile

from mitiq.interface.mitiq_qiskit import to_qiskit


def from_cudaq(kernel: PyKernel) -> Circuit:
    """Convert a cudaq PyKernel to a cirq Circuit via OpenQASM 3.

    Args:
        kernel (PyKernel): The cudaq PyKernel to convert.

    Returns:
        Circuit: The converted cirq Circuit.
    """
    cudaq_qasm = cudaq.translate(kernel, format="openqasm2")

    # The register naming convention used by `cudaq.translate` for qasm
    # causes certain equality checks like `mitiq.utils._equal` to fail
    # despite identical circuits
    # We resolve this by adding the circuit to an empty qiskit circuit
    qiskit_qc = qasm2_to_qiskit(cudaq_qasm)
    qiskit_corrected = QuantumCircuit(
        qiskit_qc.num_qubits, qiskit_qc.num_clbits
    )
    qiskit_corrected.compose(qiskit_qc, inplace=True)

    cirq_qc = qasm2_to_cirq(qiskit_to_qasm2(qiskit_corrected))

    # To avoid failing `cirq.has_stabilizer_effects` due to
    # floating point changes introduced by `cudaq.translate`
    # we round the exponent of each gate
    for op in cirq_qc.all_operations():
        if hasattr(op.gate, "exponent"):
            op.gate._exponent = np.round(op.gate.exponent, 6)  # type: ignore

    return cirq_qc


def to_cudaq(circuit: Circuit) -> PyKernel:
    """Convert a cirq Circuit to a cudaq PyKernel via OpenQASM 3.

    Args:
        circuit (Circuit): The cirq Circuit to convert.

    Returns:
        PyKernel: The converted cudaq PyKernel.
    """
    qiskit_qc = to_qiskit(circuit)

    qiskit_qc = transpile(
        qiskit_qc,
        basis_gates=["x", "y", "z", "h", "rx", "rz", "cx", "ccx"],
        optimization_level=0,
    )

    cudaq_qc = cudaq.make_kernel()
    qr = cudaq_qc.qalloc(qiskit_qc.num_qubits)

    for gate in qiskit_qc:
        if gate.name == "x":
            qubit = gate.qubits[0]._index
            cudaq_qc.x(qr[qubit])
        elif gate.name == "y":
            qubit = gate.qubits[0]._index
            cudaq_qc.y(qr[qubit])
        elif gate.name == "z":
            qubit = gate.qubits[0]._index
            cudaq_qc.z(qr[qubit])
        elif gate.name == "h":
            qubit = gate.qubits[0]._index
            cudaq_qc.h(qr[qubit])
        elif gate.name == "rx":
            qubit = gate.qubits[0]._index
            param = gate.params[0]
            cudaq_qc.rx(param, qr[qubit])
        elif gate.name == "rz":
            qubit = gate.qubits[0]._index
            param = gate.params[0]
            cudaq_qc.rz(param, qr[qubit])
        elif gate.name == "cx":
            qubits = [qubit._index for qubit in gate.qubits]
            cudaq_qc.cx(qr[qubits[0]], qr[qubits[1]])
        elif gate.name in ["ccx", "toffoli"]:
            qubits = [qubit._index for qubit in gate.qubits]
            cudaq_qc.cx([qr[qubits[0]], qr[qubits[1]]], qr[qubits[2]])
        elif gate.name == "measure":
            qubits = [qubit._index for qubit in gate.qubits]
            for qubit in qubits:
                cudaq_qc.mz(qr[qubit])

    return cudaq_qc
