# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for rotated randomized benchmarking circuits."""

import cirq
import numpy as np
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.rotated_randomized_benchmarking import (
    generate_rotated_rb_circuits,
)


@pytest.mark.parametrize("n_qubits", (1, 2))
@pytest.mark.parametrize("theta", [0, np.pi / 2, np.pi])
def test_rotated_rb_circuits(n_qubits, theta):
    depth = 10
    for trials in [2, 3]:
        circuits = generate_rotated_rb_circuits(
            n_qubits=n_qubits, num_cliffords=depth, theta=theta, trials=trials
        )
        for circ in circuits:
            zero_prob = (
                cirq.DensityMatrixSimulator()
                .simulate(circ)
                .final_density_matrix[0, 0]
                .real
            )
            assert -1.0001 <= zero_prob <= 1.0001


@pytest.mark.parametrize("n_qubits", (1, 2))
@pytest.mark.parametrize("theta", [0, np.pi / 2, np.pi])
@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_rotated_rb_conversion(n_qubits, theta, return_type):
    depth = 10
    for trials in [2, 3]:
        circuits = generate_rotated_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=depth,
            theta=theta,
            trials=trials,
            return_type=return_type,
        )
        for circ in circuits:
            assert return_type in circ.__module__


def test_rotated_rb_circuit_no_theta():
    circuit = generate_rotated_rb_circuits(n_qubits=1, num_cliffords=5)[0]
    assert (
        len(list(circuit.findall_operations_with_gate_type(cirq.ops.Rz))) > 0
    )
