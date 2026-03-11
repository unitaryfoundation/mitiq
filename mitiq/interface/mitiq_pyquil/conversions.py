# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions to convert between Mitiq's internal circuit representation and
pyQuil's circuit representation (Quil programs).
"""

from cirq import Circuit
from pyquil import Program
from qbraid.transpiler.conversions.cirq import cirq_to_pyquil
from qbraid.transpiler.conversions.pyquil import pyquil_to_cirq

QuilType = str


def to_quil(circuit: Circuit) -> QuilType:
    """Returns a Quil string representing the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a Quil string.

    Returns:
        QuilType: Quil string equivalent to the input Mitiq circuit.
    """
    return cirq_to_pyquil(circuit).out()


def to_pyquil(circuit: Circuit) -> Program:
    """Returns a pyQuil Program equivalent to the input Mitiq circuit.

    Args:
        circuit: Mitiq circuit to convert to a pyQuil Program.

    Returns:
        pyquil.Program object equivalent to the input Mitiq circuit.
    """
    return cirq_to_pyquil(circuit)


def from_pyquil(program: Program) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input pyQuil Program.

    Args:
        program: PyQuil Program to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input pyQuil Program.
    """
    return pyquil_to_cirq(program)


def from_quil(quil: QuilType) -> Circuit:
    """Returns a Mitiq circuit equivalent to the input Quil string.

    Args:
        quil: Quil string to convert to a Mitiq circuit.

    Returns:
        Mitiq circuit representation equivalent to the input Quil string.
    """
    return pyquil_to_cirq(Program(quil))
