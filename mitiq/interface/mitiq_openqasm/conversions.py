# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
OpenQASM's circuit representation.
"""

import re

from cirq import Circuit, NamedQubit

from mitiq.typing import QasmStringType


class _OpenQasmCircuit(Circuit):
    """A parsed circuit with its original quantum register declarations."""

    def __init__(
        self, circuit: Circuit, registers: tuple[tuple[str, int], ...]
    ) -> None:
        super().__init__(circuit.moments, tags=circuit.tags)
        self.registers = registers

    def copy(self) -> "_OpenQasmCircuit":
        return _OpenQasmCircuit(self, self.registers)


def from_openqasm(circuit: str) -> Circuit:
    """Parse an OpenQASM program and retain its quantum register layout.

    Args:
        circuit: The OpenQASM program to convert.

    Returns:
        A Cirq circuit containing the program's operations.
    """
    from cirq.contrib.qasm_import._parser import QasmParser

    from mitiq.interface.mitiq_qiskit.conversions import _remove_qasm_barriers

    # Cirq's circuit_from_qasm discards the parser's register declarations.
    parsed = QasmParser().parse(_remove_qasm_barriers(circuit))
    return _OpenQasmCircuit(parsed.circuit, tuple(parsed.qregs.items()))


def _restore_registers(qasm: str, circuit: _OpenQasmCircuit) -> str:
    """Restore quantum addresses in Cirq-generated OpenQASM 2 output."""
    registers = dict(circuit.registers)
    active = sorted(circuit.all_qubits())
    if circuit.registers == (("q", len(active)),) and all(
        q == NamedQubit(f"q_{i}") for i, q in enumerate(active)
    ):
        return qasm

    classical = set(re.findall(r"^creg (\w+)\[", qasm, re.MULTILINE))
    used_names = registers.keys() | classical

    def available_name(name: str) -> str:
        while name in used_names:
            name += "_"
        used_names.add(name)
        return name

    # A generated measurement register can share an original qreg's name.
    renamed = {
        name: available_name(name)
        for name in sorted(classical & registers.keys())
    }
    extra = available_name("ancilla")
    extra_size = 0
    operands = []
    for qubit in active:
        if isinstance(qubit, NamedQubit):
            address = re.fullmatch(r"(.+)_(0|[1-9][0-9]*)", qubit.name)
            if address and int(address[2]) < registers.get(address[1], 0):
                operands.append(f"{address[1]}[{address[2]}]")
                continue
        # A circuit modifier may introduce qubits of its own.
        operands.append(f"{extra}[{extra_size}]")
        extra_size += 1
    if extra_size:
        registers[extra] = extra_size

    # Rewrite only the fixed syntax emitted by Cirq. Leave comments and
    # include paths intact. A single pass avoids remapping a restored name.
    def replace(match: re.Match[str]) -> str:
        if match[1] is not None:
            return operands[int(match[1])]
        return renamed.get(match[0], match[0])

    qasm = re.sub(r"^qreg q\[\d+\];\n", "", qasm, flags=re.MULTILINE)
    qasm = re.sub(
        r'//[^\n]*|"[^"\n]*"|\bq\[(\d+)\]|\b[a-zA-Z_]\w*\b', replace, qasm
    )
    declarations = "".join(f"qreg {n}[{s}];\n" for n, s in registers.items())
    return qasm.replace(
        'include "qelib1.inc";\n', 'include "qelib1.inc";\n' + declarations, 1
    )


def to_openqasm(circuit: Circuit) -> str:
    """Serialize a Cirq circuit, restoring imported quantum registers.

    Args:
        circuit: The Cirq circuit to convert.

    Returns:
        An OpenQASM 2 program.
    """
    from mitiq.interface.mitiq_qiskit import to_qasm

    qasm = to_qasm(circuit.copy())
    if isinstance(circuit, _OpenQasmCircuit):
        qasm = _restore_registers(qasm, circuit)
    return QasmStringType(qasm)
