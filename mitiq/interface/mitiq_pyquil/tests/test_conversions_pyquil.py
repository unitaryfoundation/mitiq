# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests conversions to/from pyQuil circuits."""

import numpy as np
from pyquil import Program
from pyquil.gates import CNOT, CZ, RZ, H, X, Y, Z

from mitiq.interface.mitiq_pyquil.conversions import from_pyquil, to_pyquil


def test_to_pyquil_from_pyquil_simple():
    p = Program()
    p += X(0)
    p += Y(1)
    p += Z(2)
    p += CNOT(0, 1)
    p += CZ(1, 2)
    assert p.out() == to_pyquil(from_pyquil(p)).out()


def maxcut_qaoa_program(gamma: float) -> Program:
    """
    Generates a 2Q MAXCUT QAOA circuit with beta = pi/8 and with the provided
    gamma.

    Args:
        gamma: One of the two variational parameters (the other is fixed).
    Returns:
        A 2Q MAXCUT QAOA circuit with fixed beta and gamma.
    """
    q0, q1 = (0, 1)
    p = Program()
    p += H(q0)
    p += H(q1)
    p += CNOT(q0, q1)
    p += RZ(2 * gamma, q1)
    p += CNOT(q0, q1)
    p += H(q0)
    p += H(q1)
    p += RZ(np.pi / 4, q0)
    p += RZ(np.pi / 4, q1)
    p += H(q0)
    p += H(q1)
    return p


def test_to_pyquil_from_pyquil_parameterized():
    p = maxcut_qaoa_program(np.pi)
    assert p.out() == to_pyquil(from_pyquil(p)).out()


MEASURELESS_QUIL_PROGRAM = """
I 0
I 1
I 2
X 0
Y 1
Z 2
RX(1.5707963267948966) 0
RY(1.5707963267948966) 1
RZ(1.5707963267948966) 2
H 0
CZ 0 1
CNOT 1 2
CPHASE(1.5707963267948966) 0 1
ISWAP 0 1
SWAP 1 2
XY(1.5707963267948966) 0 1
CCNOT 0 1 2
CSWAP 0 1 2
"""


def test_to_pyquil_from_pyquil_almost_all_gates():
    """PHASE, PSWAP, CHPASEij, S, T, declaration, and measurement don't convert
    back and forth perfectly (in terms of labels -- the program unitaries and
    number of measurements are equivalent)."""
    p = Program(MEASURELESS_QUIL_PROGRAM)
    assert p.out() == to_pyquil(from_pyquil(p)).out()


def test_to_pyquil_from_pyquil_not_starting_at_zero():
    p = Program()
    p += X(10)
    p += Y(11)
    p += Z(12)
    p += CNOT(10, 11)
    p += CZ(11, 12)
    assert p.out() == to_pyquil(from_pyquil(p)).out()
