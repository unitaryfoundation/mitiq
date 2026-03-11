# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for generating (near) Clifford circuits."""

import cirq
import numpy as np
import pytest
from cirq.circuits import Circuit

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.cdr.clifford_utils import (
    _CLIFFORD_ANGLES,
    angle_to_proximities,
    angles_to_proximities,
    closest_clifford,
    count_non_cliffords,
    is_clifford,
    is_clifford_angle,
    probabilistic_angle_to_clifford,
    random_clifford,
)
from mitiq.interface import convert_from_mitiq


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_is_clifford_with_clifford(circuit_type):
    circuit = convert_from_mitiq(
        cirq.Circuit(cirq.Z.on(cirq.LineQubit(0))), circuit_type
    )
    assert is_clifford(circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_is_clifford_with_nonclifford(circuit_type):
    circuit = convert_from_mitiq(
        cirq.Circuit(cirq.T.on(cirq.LineQubit(0))), circuit_type
    )
    assert not is_clifford(circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_count_non_cliffords(circuit_type):
    a, b = cirq.LineQubit.range(2)
    circuit = Circuit(
        cirq.rz(0.0).on(a),  # Clifford.
        cirq.rx(0.1 * np.pi).on(b),  # Non-Clifford.
        cirq.rx(0.5 * np.pi).on(b),  # Clifford
        cirq.rz(0.4 * np.pi).on(b),  # Non-Clifford.
        cirq.rz(0.5 * np.pi).on(b),  # Clifford.
        cirq.CNOT.on(a, b),  # Clifford.
    )
    circuit = convert_from_mitiq(circuit, circuit_type)

    assert count_non_cliffords(circuit) == 2


def test_count_non_cliffords_empty_circuit():
    assert count_non_cliffords(Circuit()) == 0


def test_is_clifford_angle():
    for p in range(4):
        assert is_clifford_angle(p * np.array(_CLIFFORD_ANGLES)).all()

    assert not is_clifford_angle(-0.17)


def test_closest_clifford():
    for ang in _CLIFFORD_ANGLES:
        angs = np.linspace(ang - np.pi / 4 + 0.01, ang + np.pi / 4 - 0.01)
        for a in angs:
            assert closest_clifford(a) == ang


def test_random_clifford():
    assert set(random_clifford(20, np.random.RandomState(1))).issubset(
        _CLIFFORD_ANGLES
    )


@pytest.mark.parametrize("sigma", np.linspace(0.1, 2, 5))
def test_angle_to_proximities(sigma):
    for ang in _CLIFFORD_ANGLES:
        probabilities = angle_to_proximities(ang, sigma)
        assert (isinstance(p, float) for p in probabilities)


@pytest.mark.parametrize("sigma", np.linspace(0.1, 2, 5))
def test_angles_to_proximities(sigma):
    probabilities = angles_to_proximities(_CLIFFORD_ANGLES, sigma)
    assert all(isinstance(p, float) for p in probabilities)


@pytest.mark.parametrize("sigma", np.linspace(0.1, 2, 5))
def test_probabilistic_angles_to_clifford(sigma):
    angles = probabilistic_angle_to_clifford(
        _CLIFFORD_ANGLES, sigma, np.random.RandomState(1)
    )
    assert all(a in _CLIFFORD_ANGLES for a in angles)
