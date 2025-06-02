# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
#
# This module includes a modified version of the ApplyLayout class originally
# licensed under the Apache License, Version 2.0.
#
# Original Author: IBM
# Modification made by: André Alves on 07 Apr 2024
#
# For the original code and license, see
# https://github.com/Qiskit/qiskit/blob/stable/1.0/qiskit/transpiler/passes/layout/apply_layout.py
#
# Modifications:
# - Renamed class to ApplyMitiqLayout
# - Removed layout length check
# - Added QuantumRegisters as parameter for class
# - Removed check for no post_layout
# - Improved typing
# - Modified docstring format
# - Modified length check

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class ApplyMitiqLayout(TransformationPass):  # type: ignore
    """Transform a circuit with virtual qubits into a circuit with physical
    qubits specified in the given list of QuantumRegisters.

    Transforms a DAGCircuit with virtual qubits into a DAGCircuit with physical
    qubits by applying the Layout given in `property_set`.
    Requires either of passes to set/select Layout, e.g. `SetLayout`,
    `TrivialLayout`. Assumes the Layout has full physical qubits.

    Args:
        new_qregs: The new quantum registers for the circuit.
    """

    def __init__(self, new_qregs: list[QuantumRegister]) -> None:
        super().__init__()
        self._new_qregs = new_qregs

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ApplyLayout pass on ``dag``.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG (with physical qubits).

        Raises:
            TranspilerError: if no layout is found in ``property_set`` or
                             no full physical qubits.
        """
        layout = self.property_set["layout"]
        if not layout:
            raise TranspilerError(
                "No 'layout' is found in property_set. "
                "Please run a Layout pass in advance."
            )

        q = [qbit for qreg in self._new_qregs for qbit in qreg]
        if len(q) < len(dag.qubits):
            raise TranspilerError(
                "New layout has less qubits than DAGCircuit."
            )

        new_dag = DAGCircuit()

        for qreg in self._new_qregs:
            new_dag.add_qreg(qreg)

        new_dag.metadata = dag.metadata
        new_dag.add_clbits(dag.clbits)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        self.property_set["original_qubit_indices"] = {
            bit: index for index, bit in enumerate(dag.qubits)
        }
        for qreg in dag.qregs.values():
            self.property_set["layout"].add_register(qreg)
        virtual_physical_map = layout.get_virtual_bits()
        for node in dag.topological_op_nodes():
            qargs = [q[virtual_physical_map[qarg]] for qarg in node.qargs]
            new_dag.apply_operation_back(
                node.op, qargs, node.cargs, check=False
            )

        new_dag.global_phase = dag.global_phase

        return new_dag


class ClearLayout(TransformationPass):  # type: ignore
    """Clears the layout of the DAGCircuit"""

    def __init__(self) -> None:
        super().__init__()

    def run(self, dag: DAGCircuit) -> None:
        """Clears the layout from the DAGCircuit's `property_set`."""
        self.property_set.pop("layout", None)
