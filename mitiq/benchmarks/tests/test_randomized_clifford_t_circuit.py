# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for randomized Clifford+T benchmarking circuits."""

import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.randomized_clifford_t_circuit import (
    generate_random_clifford_t_circuit,
)


def test_seed_circuit_equality():
    """Test that a model circuit is determined by its seed."""
    circuit_1 = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        seed=1,
    )
    circuit_2 = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        seed=1,
    )
    circuit_3 = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        seed=3,
    )

    assert circuit_1 == circuit_2
    assert circuit_2 != circuit_3


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_conversion(return_type):
    circuit = generate_random_clifford_t_circuit(
        num_qubits=2,
        num_oneq_cliffords=2,
        num_twoq_cliffords=2,
        num_t_gates=2,
        return_type=return_type,
    )
    assert return_type in circuit.__module__


@pytest.mark.parametrize(
    "test_num_qubits, test_num_twoq_cliffords, error_msg",
    [
        (0, 2, "Cannot prepare a circuit with"),
        (1, 2, "Need more than 2 qubits for two-qubit Clifford gates."),
    ],
)
def test_invalid_input(test_num_qubits, test_num_twoq_cliffords, error_msg):
    """Ensures error raised as expected."""
    with pytest.raises(ValueError, match=error_msg):
        generate_random_clifford_t_circuit(
            num_qubits=test_num_qubits,
            num_oneq_cliffords=2,
            num_twoq_cliffords=test_num_twoq_cliffords,
            num_t_gates=2,
        )
