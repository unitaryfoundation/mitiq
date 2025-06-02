# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for GHZ circuits."""

import numpy as np
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks import ghz_circuits


@pytest.mark.parametrize("nqubits", [1, 5])
def test_ghz_circuits(nqubits):
    # test GHZ creation
    circuit = ghz_circuits.generate_ghz_circuit(nqubits)
    # check that the state |0...0> and the state |1...1>
    # both have probability close to 0.5
    sv = circuit.final_state_vector()
    zero_prob = abs(sv[0]) ** 2
    one_prob = abs(sv[-1]) ** 2
    assert np.isclose(zero_prob, 0.5)
    assert np.isclose(one_prob, 0.5)


def test_ghz_value_error():
    # test GHZ error raising
    with pytest.raises(ValueError):
        circuit = ghz_circuits.generate_ghz_circuit(0)  # noqa: F841


@pytest.mark.parametrize("return_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_ghz_conversion(return_type):
    circuit = ghz_circuits.generate_ghz_circuit(3, return_type)
    assert return_type in circuit.__module__
