# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for executors defined under PEP 563.

When ``from __future__ import annotations`` is active in the module where
an executor is defined, every annotation is stored as a string.
``Executor`` must still resolve the return-type annotation
(e.g. ``"float"`` -> ``float``) so that float / measurement /
density-matrix executors are detected correctly.

See https://github.com/unitaryfoundation/mitiq for the originating issue.
"""

from __future__ import annotations

import cirq
import numpy as np

from mitiq.executor.executor import Executor
from mitiq.observable import Observable, PauliString


# Defined in a module using PEP 563, so this annotation is the string "float".
def float_executor(circuit) -> float:
    return 1.0


def test_return_annotation_resolved_to_type():
    """The stringized "float" annotation should resolve to the float type."""
    executor = Executor(float_executor)
    assert executor._executor_return_type is float


def test_evaluate_float_executor_under_pep563():
    """A float executor defined under PEP 563 must evaluate without error."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q))

    results = Executor(float_executor).evaluate(circuit)

    assert np.allclose(results, [1.0])
    assert not Executor(float_executor).can_batch


def test_evaluate_with_observable_under_pep563():
    """An observable-based evaluation should also work under PEP 563."""

    def density_matrix_executor(circuit) -> np.ndarray:
        return cirq.final_density_matrix(circuit)

    obs = Observable(PauliString("Z"))
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q))

    results = Executor(density_matrix_executor).evaluate(circuit, obs)

    assert np.allclose(results, [1.0])
