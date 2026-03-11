# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for qiskit TransformationPass (transpiler.py)."""

import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import Layout
from qiskit.transpiler.exceptions import TranspilerError

from mitiq.interface.mitiq_qiskit import ApplyMitiqLayout, ClearLayout


def test_apply_mitiq_layout_success():
    """Tests ApplyMitiqLayout successful run."""
    # Setup: Create a simple quantum circuit and convert it to a DAG
    qr = QuantumRegister(3, name="qr")
    circuit = QuantumCircuit(qr)
    circuit.h(qr[0])
    circuit.cx(qr[0], qr[2])
    dag = circuit_to_dag(circuit)

    new_qregs = [QuantumRegister(3, name="new_qr")]
    layout_pass = ApplyMitiqLayout(new_qregs=new_qregs)

    # Manually set a layout in the property_set (usually done by previous pass)
    layout_pass.property_set = {
        "layout": Layout.from_qubit_list(circuit.qubits)
    }

    # Run the transformation
    new_dag = layout_pass.run(dag)

    # Assertions: Verify the new DAG has the expected structure
    assert len(new_dag.qubits) == len(qr)
    for qubit in new_dag.qubits:
        assert qubit._register.name == "new_qr"


def test_apply_mitiq_layout_fail_small_register():
    """Tests ApplyMitiqLayout run fails with too small a QuantumRegister."""
    # Setup: Create a simple quantum circuit and convert it to a DAG
    qr = QuantumRegister(3, name="qr")
    circuit = QuantumCircuit(qr)
    circuit.h(qr[0])
    circuit.cx(qr[0], qr[2])
    dag = circuit_to_dag(circuit)

    new_qregs = [QuantumRegister(2, name="new_qr")]
    layout_pass = ApplyMitiqLayout(new_qregs=new_qregs)

    # Manually set a layout in the property_set (usually done by previous pass)
    layout_pass.property_set = {
        "layout": Layout.from_qubit_list(circuit.qubits)
    }

    # Expect TranspilerError due to small QuantumRegister
    with pytest.raises(TranspilerError):
        layout_pass.run(dag)


def test_apply_mitiq_layout_fail_no_layout():
    """Tests ApplyMitiqLayout run fails with no layout."""
    # Setup a basic circuit and DAG
    qr = QuantumRegister(2, name="qr")
    circuit = QuantumCircuit(qr)
    dag = circuit_to_dag(circuit)

    layout_pass = ApplyMitiqLayout(new_qregs=[QuantumRegister(2, "new_qr")])

    # Make sure there is no layout
    if layout_pass.property_set["layout"]:
        del layout_pass.property_set["layout"]

    # Expect TranspilerError due to missing layout
    with pytest.raises(TranspilerError):
        layout_pass.run(dag)


def test_clear_layout_clears_the_layout():
    """Tests ClearLayout clears the layout."""
    # Setup: Create a simple quantum circuit and convert it to a DAG
    qr = QuantumRegister(2, name="qr")
    circuit = QuantumCircuit(qr)
    dag = circuit_to_dag(circuit)

    layout_pass = ClearLayout()

    # Manually set a layout in the property_set (usually done by previous pass)
    layout_pass.property_set = {
        "layout": Layout.from_qubit_list(circuit.qubits)
    }

    layout_pass.run(dag)
    assert "layout" not in layout_pass.property_set
