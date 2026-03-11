# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for parameter scaling."""

from copy import deepcopy

import numpy as np
import pytest
from cirq import CSWAP, Circuit, LineQubit, ZPowGate, ops

from mitiq.utils import _equal
from mitiq.zne.scaling.parameter import (
    CircuitMismatchException,
    GateTypeException,
    _generate_parameter_calibration_circuit,
    _get_base_gate,
    compute_parameter_variance,
    scale_parameters,
)


def test_identity_scale_1q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(3)
    circ = Circuit([ops.X.on_each(qreg)], [ops.Y.on(qreg[0])])
    scaled = scale_parameters(circ, scale_factor=1, base_variance=0.001)
    assert _equal(circ, scaled)


def test_non_identity_scale_1q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.rx(np.pi * 1.0).on_each(qreg)], [ops.ry(np.pi * 1.0).on(qreg[0])]
    )
    np.random.seed(42)
    stretch = 2
    base_noise = 0.001
    noises = np.random.normal(
        loc=0.0, scale=np.sqrt((stretch - 1) * base_noise), size=(4,)
    )
    np.random.seed(42)

    scaled = scale_parameters(
        circ, scale_factor=stretch, base_variance=base_noise, seed=42
    )
    result = []
    for moment in scaled:
        for op in moment.operations:
            gate = deepcopy(op.gate)
            param = gate.exponent
            result.append(param * np.pi - np.pi)
    assert np.all(np.isclose(result - noises, 0))


def test_identity_scale_2q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(2)
    circ = Circuit([ops.CNOT.on(qreg[0], qreg[1])])
    scaled = scale_parameters(circ, scale_factor=1, base_variance=0.001)
    assert _equal(circ, scaled)


def test_non_identity_scale_2q():
    """Tests that when scale factor = 1, the circuit is the
    same.
    """
    qreg = LineQubit.range(2)
    circ = Circuit([ops.CNOT.on(qreg[0], qreg[1])])
    np.random.seed(42)
    stretch = 2
    base_noise = 0.001
    noises = np.random.normal(
        loc=0.0, scale=np.sqrt((stretch - 1) * base_noise), size=(1,)
    )
    np.random.seed(42)
    scaled = scale_parameters(
        circ, scale_factor=stretch, base_variance=base_noise, seed=42
    )
    result = []
    for moment in scaled:
        for op in moment.operations:
            gate = deepcopy(op.gate)
            param = gate.exponent
            result.append(param * np.pi - np.pi)
    assert np.all(np.isclose(result - noises, 0))


def test_scale_with_measurement():
    """Tests that we ignore measurement gates.

    Test circuit:
    0: ───H───T───@───M───
                  │   │
    1: ───H───M───┼───┼───
                  │   │
    2: ───H───────X───M───

    """
    qreg = LineQubit.range(3)
    circ = Circuit(
        [ops.H.on_each(qreg)],
        [ops.T.on(qreg[0])],
        [ops.measure(qreg[1])],
        [ops.CNOT.on(qreg[0], qreg[2])],
        [ops.measure(qreg[0], qreg[2])],
    )
    scaled = scale_parameters(circ, scale_factor=1, base_variance=0.001)
    assert _equal(circ, scaled)


def test_gate_type():
    qreg = LineQubit.range(3)
    allowed_op = ops.H.on(qreg[0])
    _get_base_gate(allowed_op.gate)
    with pytest.raises(GateTypeException):
        forbidden_op = CSWAP(qreg[0], qreg[1], qreg[2])
        _get_base_gate(forbidden_op)


def test_compute_parameter_variance():
    def noiseless_executor_mock(circuit):
        return 1

    # Perfect executor should have sigma = 0
    qubit = LineQubit(0)
    gate = ops.H.on(qubit).gate
    sigma = compute_parameter_variance(
        noiseless_executor_mock, gate, qubit, depth=10
    )
    assert sigma == 0

    # Perfectly imperfect executor should have sigma = inf
    def noisy_executor_mock(circuit):
        return 0.5

    qubit = LineQubit(0)
    gate = ops.H.on(qubit).gate
    with pytest.warns(RuntimeWarning):
        # Runtime warning for divide by zero
        sigma = compute_parameter_variance(
            noisy_executor_mock, gate, qubit, depth=10
        )
    assert sigma == np.inf


def test_generate_parameter_calibration_circuit():
    """Tests generating a simple Parameter Calibration circuit"""
    n_qubits = 1
    qubits = LineQubit.range(n_qubits)
    depth = 10
    circuit = _generate_parameter_calibration_circuit(qubits, depth, ZPowGate)
    assert len(circuit) == depth
    # Make sure the exponents in the
    for i in range(len(circuit)):
        assert circuit[i].operations[0].gate.exponent == 2 * np.pi / depth


def test_generate_parameter_calibration_circuit_failure():
    """Tests that parameter calibration circuit generation fails because there
    are too many qubits"""
    n_qubits = 3
    qubits = LineQubit.range(n_qubits)
    depth = 10
    # Should raise exception because too many qubits
    with pytest.raises(CircuitMismatchException):
        _generate_parameter_calibration_circuit(qubits, depth, ZPowGate)
