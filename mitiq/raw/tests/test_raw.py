# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for RAW (no error mitigation)."""

import functools

import cirq
import numpy as np

from mitiq import Executor, Observable, PauliString, raw
from mitiq.interface import mitiq_cirq


def test_raw():
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0)))
    observable = Observable(PauliString("X"))
    executor = Executor(mitiq_cirq.compute_density_matrix)
    raw_value = raw.execute(circuit, executor, observable)

    assert isinstance(raw_value, float)
    assert executor.executed_circuits == [circuit]

    compute_density_matrix_noiseless = functools.partial(
        mitiq_cirq.compute_density_matrix, noise_level=(0.0,)
    )
    executor_noiseless = Executor(compute_density_matrix_noiseless)
    true_value = raw.execute(circuit, executor_noiseless, observable)
    assert np.isclose(true_value, 1.0)
    assert executor_noiseless.executed_circuits == [circuit]
