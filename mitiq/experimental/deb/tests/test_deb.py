# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for high-level debiasing functions."""

import cirq

from mitiq.experimental.deb.deb import (
    execute_with_debiasing,
    execute_with_debiasing_and_sharpening,
)


def test_execute_with_debiasing():
    """Test basic debiasing execution."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key="result"))

    def executor(circ):
        simulator = cirq.Simulator()
        result = simulator.run(circ, repetitions=100)
        return dict(result.histogram(key="result"))

    debiased = execute_with_debiasing(
        circuit, executor, num_variants=3, random_state=42
    )
    assert isinstance(debiased, dict)
    assert len(debiased) > 0
    # Check normalization
    total = sum(debiased.values())
    assert abs(total - 1.0) < 0.01


def test_execute_with_debiasing_and_sharpening():
    """Test debiasing with sharpening."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key="result"))

    def executor(circ):
        simulator = cirq.Simulator()
        result = simulator.run(circ, repetitions=100)
        return dict(result.histogram(key="result"))

    sharpened = execute_with_debiasing_and_sharpening(
        circuit, executor, num_variants=3, random_state=42, sharpen_threshold=2
    )
    assert isinstance(sharpened, dict)
    assert len(sharpened) > 0
    # Check normalization
    total = sum(sharpened.values())
    assert abs(total - 1.0) < 0.01
