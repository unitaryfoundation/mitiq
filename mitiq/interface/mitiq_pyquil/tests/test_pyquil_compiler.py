# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""NB: Copied in large part from rigetti/forest-benchmarking (Apache-2.0)
and modified to test a larger gateset.
"""

import inspect
import itertools
import random
from math import pi

import numpy as np
import pytest
from cirq import equal_up_to_global_phase
from pyquil.gates import (
    CCNOT,
    CNOT,
    CPHASE,
    CSWAP,
    CZ,
    ISWAP,
    PHASE,
    RX,
    RY,
    RZ,
    SWAP,
    XY,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
)
from pyquil.quil import Program
from pyquil.simulation.tools import program_unitary

from mitiq.interface.mitiq_pyquil.compiler import (
    _CCNOT,
    _CNOT,
    _CPHASE,
    _H,
    _ISWAP,
    _RY,
    _S,
    _SWAP,
    _T,
    _X,
    _Y,
    _Z,
    basic_compile,
)


def test_basic_compile_defgate():
    p = Program()
    p.inst(RX(pi, 0))
    p.defgate("test", [[0, 1], [1, 0]])
    p.inst(("test", 2))
    p.inst(RZ(pi / 2, 0))

    assert p == basic_compile(p)


def test_CCNOT():
    for perm in itertools.permutations([0, 1, 2]):
        u1 = program_unitary(Program(CCNOT(*perm)), n_qubits=3)
        u2 = program_unitary(_CCNOT(*perm), n_qubits=3)
        assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_CNOT():
    u1 = program_unitary(Program(CNOT(0, 1)), n_qubits=2)
    u2 = program_unitary(_CNOT(0, 1), n_qubits=2)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)

    u1 = program_unitary(Program(CNOT(1, 0)), n_qubits=2)
    u2 = program_unitary(_CNOT(1, 0), n_qubits=2)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_CPHASE():
    for theta in np.linspace(-2 * np.pi, 2 * np.pi):
        u1 = program_unitary(Program(CPHASE(theta, 0, 1)), n_qubits=2)
        u2 = program_unitary(_CPHASE(theta, 0, 1), n_qubits=2)
        assert equal_up_to_global_phase(u1, u2, atol=1e-12)

        u1 = program_unitary(Program(CPHASE(theta, 1, 0)), n_qubits=2)
        u2 = program_unitary(_CPHASE(theta, 1, 0), n_qubits=2)
        assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_H():
    u1 = program_unitary(Program(H(0)), n_qubits=1)
    u2 = program_unitary(_H(0), n_qubits=1)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_ISWAP():
    u1 = program_unitary(Program(ISWAP(0, 1)), n_qubits=2)
    u2 = program_unitary(_ISWAP(0, 1), n_qubits=2)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)

    u1 = program_unitary(Program(ISWAP(1, 0)), n_qubits=2)
    u2 = program_unitary(_ISWAP(1, 0), n_qubits=2)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_RX():
    for theta in np.linspace(-2 * np.pi, 2 * np.pi):
        p = Program(RX(theta, 0))
        u1 = program_unitary(p, n_qubits=1)
        u2 = program_unitary(basic_compile(p), n_qubits=1)
        assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_RY():
    for theta in np.linspace(-2 * np.pi, 2 * np.pi):
        u1 = program_unitary(Program(RY(theta, 0)), n_qubits=1)
        u2 = program_unitary(_RY(theta, 0), n_qubits=1)
        assert equal_up_to_global_phase(u1, u2)


def test_S():
    u1 = program_unitary(Program(S(0)), n_qubits=1)
    u2 = program_unitary(_S(0), n_qubits=1)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_SWAP():
    u1 = program_unitary(Program(SWAP(0, 1)), n_qubits=2)
    u2 = program_unitary(_SWAP(0, 1), n_qubits=2)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)

    u1 = program_unitary(Program(SWAP(1, 0)), n_qubits=2)
    u2 = program_unitary(_SWAP(1, 0), n_qubits=2)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_T():
    u1 = program_unitary(Program(T(0)), n_qubits=1)
    u2 = program_unitary(_T(0), n_qubits=1)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_X():
    u1 = program_unitary(Program(X(0)), n_qubits=1)
    u2 = program_unitary(_X(0), n_qubits=1)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_XY():
    for theta in np.linspace(-2 * np.pi, 2 * np.pi):
        p = Program(XY(theta, 0, 1))
        u1 = program_unitary(p, n_qubits=2)
        u2 = program_unitary(basic_compile(p), n_qubits=2)
        assert equal_up_to_global_phase(u1, u2, atol=1e-12)

        p = Program(XY(theta, 1, 0))
        u1 = program_unitary(p, n_qubits=2)
        u2 = program_unitary(basic_compile(p), n_qubits=2)
        assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_Y():
    u1 = program_unitary(Program(Y(0)), n_qubits=1)
    u2 = program_unitary(_Y(0), n_qubits=1)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_Z():
    u1 = program_unitary(Program(Z(0)), n_qubits=1)
    u2 = program_unitary(_Z(0), n_qubits=1)
    assert equal_up_to_global_phase(u1, u2, atol=1e-12)


# Note to developers: unsupported gates are commented out.
QUANTUM_GATES = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
    "PHASE": PHASE,
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "CZ": CZ,
    "CNOT": CNOT,
    "CCNOT": CCNOT,
    # 'CPHASE00': CPHASE00,
    # 'CPHASE01': CPHASE01,
    # 'CPHASE10': CPHASE10,
    "CPHASE": CPHASE,
    "SWAP": SWAP,
    # 'CSWAP': CSWAP,
    "ISWAP": ISWAP,
    # 'PSWAP': PSWAP
}


def _generate_random_program(n_qubits, length):
    """Randomly sample gates and arguments (qubits, angles)"""
    if n_qubits < 3:
        raise ValueError(
            "Please request n_qubits >= 3 so we can use 3-qubit gates."
        )

    gates = list(QUANTUM_GATES.values())
    prog = Program()
    for _ in range(length):
        gate = random.choice(gates)
        possible_qubits = set(range(n_qubits))
        sig = inspect.signature(gate)

        param_vals = []
        for param in sig.parameters:
            if param in [
                "qubit",
                "q1",
                "q2",
                "control",
                "control1",
                "control2",
                "target",
                "target_1",
                "target_2",
                "classical_reg",
            ]:
                param_val = random.choice(list(possible_qubits))
                possible_qubits.remove(param_val)
            elif param == "angle":
                # TODO: support rx(theta)
                if gate == RX:
                    param_val = random.choice([-1, -0.5, 0, 0.5, 1]) * pi
                else:
                    param_val = random.uniform(-2 * pi, 2 * pi)
            else:
                raise ValueError("Unknown gate parameter {}".format(param))

            param_vals.append(param_val)

        prog += gate(*param_vals)

    return prog


@pytest.fixture(params=list(range(3, 5)))
def n_qubits(request):
    return request.param


@pytest.fixture(params=[2, 50, 67])
def prog_length(request):
    return request.param


def test_random_progs(n_qubits, prog_length):
    for repeat_i in range(10):
        prog = _generate_random_program(n_qubits=n_qubits, length=prog_length)
        u1 = program_unitary(prog, n_qubits=n_qubits)
        u2 = program_unitary(basic_compile(prog), n_qubits=n_qubits)
        assert equal_up_to_global_phase(u1, u2, atol=1e-12)


def test_unsupported_gate():
    with pytest.raises(ValueError):
        basic_compile(Program(CSWAP(0, 1, 2)))


def test_other_instructions():
    p = Program("DECLARE ro BIT[2]")
    assert p == basic_compile(p)
