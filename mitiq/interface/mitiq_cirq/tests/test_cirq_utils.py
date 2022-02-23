# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
""" Tests for Cirq executors defined in cirq_utils.py"""

import numpy as np
import cirq

from mitiq.interface.mitiq_cirq import execute_with_depolarizing_noise


def test_execute_with_depolarizing_noise():
    """Tests if executor function for Cirq returns a proper expectation
    value when used for noisy depoalrizing simulation."""

    qc = cirq.Circuit()
    for _ in range(100):
        qc += cirq.X(cirq.LineQubit(0))

    observable = np.diag([0, 1])
    # Test 1
    noise1 = 0.0
    observable_exp_value = execute_with_depolarizing_noise(
        qc, observable, noise1
    )
    assert 0.0 == observable_exp_value

    # Test 2
    noise2 = 0.5
    observable_exp_value = execute_with_depolarizing_noise(
        qc, observable, noise2
    )
    assert np.isclose(observable_exp_value, 0.5)

    # Test 3
    noise3 = 0.001
    observable_exp_value = execute_with_depolarizing_noise(
        qc, observable, noise3
    )
    assert np.isclose(observable_exp_value, 0.062452)
