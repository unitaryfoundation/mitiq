# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import numpy as np
import pytest

from mitiq import MeasurementResult
from mitiq.experimental.deb import (
    construct_circuits,
    execute_with_debiasing,
    execute_with_debiasing_and_sharpening,
    sharpen,
)


def _sample_counts(
    circuit: cirq.Circuit, *, noise: float = 0.0, shots: int = 2000
) -> MeasurementResult:
    simulator = cirq.DensityMatrixSimulator(
        noise=cirq.depolarize(noise) if noise else None, seed=1
    )
    result = simulator.run(circuit, repetitions=shots)
    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())),
        qubit_indices=tuple(
            int(q[2:-1]) for key in result.measurements for q in key.split(",")
        ),
    )


def test_construct_circuits_count_and_reproducibility():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))

    variants = construct_circuits(circuit, 5, random_state=123)
    repeated = construct_circuits(circuit, 5, random_state=123)

    assert len(variants) == 5
    assert [str(c) for c in variants] == [str(c) for c in repeated]


def test_construct_circuits_rejects_non_positive_variants():
    with pytest.raises(ValueError, match="positive"):
        construct_circuits(cirq.Circuit(), 0)


def test_pauli_layers_cancel_on_noiseless_identity_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, q1))
    base_counts = _sample_counts(circuit).get_counts()

    for variant in construct_circuits(circuit, 8, random_state=5):
        variant_counts = _sample_counts(variant).get_counts()
        assert variant_counts == base_counts


def test_execute_with_debiasing_averages_variant_distributions():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0))

    distribution, data = execute_with_debiasing(
        circuit,
        lambda c: _sample_counts(c, shots=500),
        12,
        random_state=7,
        full_output=True,
    )

    assert distribution["1"] == pytest.approx(1.0)
    assert len(data["circuits"]) == 12
    assert len(data["variant_distributions"]) == 12


def test_execute_with_debiasing_works_with_depolarizing_noise():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0))

    noisy_distribution = execute_with_debiasing(
        circuit,
        lambda c: _sample_counts(c, noise=0.05, shots=2000),
        16,
        random_state=11,
    )

    assert noisy_distribution["1"] > noisy_distribution.get("0", 0.0)
    assert sum(noisy_distribution.values()) == pytest.approx(1.0)


def test_sharpen_returns_plurality_bitstring():
    sharpened = sharpen(
        [
            {"00": 8, "11": 2},
            {"00": 3, "11": 5},
            {"00": 9, "11": 1},
        ]
    )
    assert sharpened == {
        "00": pytest.approx(2 / 3),
        "11": pytest.approx(1 / 3),
    }


def test_execute_with_debiasing_and_sharpening():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0))

    sharpened = execute_with_debiasing_and_sharpening(
        circuit,
        lambda c: _sample_counts(c, shots=500),
        10,
        random_state=13,
    )

    assert sharpened == {"1": pytest.approx(1.0)}
