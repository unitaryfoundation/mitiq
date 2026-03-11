# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for randomized benchmarking circuits."""

import numpy as np
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits


@pytest.mark.parametrize("n_qubits", (1, 2))
def test_rb_circuits(n_qubits):
    depth = 10

    # test single qubit RB
    for trials in [2, 3]:
        circuits = generate_rb_circuits(
            n_qubits=n_qubits, num_cliffords=depth, trials=trials
        )
        for qc in circuits:
            # we check the ground state population to ignore any global phase
            wvf = qc.final_state_vector()
            zero_prob = abs(wvf[0] ** 2)
            assert np.isclose(zero_prob, 1)


@pytest.mark.parametrize("n_qubits", (1, 2))
@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_rb_conversion(n_qubits, return_type):
    depth = 10
    for trials in [2, 3]:
        circuits = generate_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=depth,
            trials=trials,
            return_type=return_type,
        )
        for qc in circuits:
            assert return_type in qc.__module__


def test_bad_n_qubits():
    with pytest.raises(
        ValueError, match="Only generates RB circuits on one or two "
    ):
        for trials in [2, 3]:
            generate_rb_circuits(n_qubits=4, num_cliffords=10, trials=trials)
