# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import pytest

from mitiq import MeasurementResult
from mitiq.experimental.deb.deb import (
    combine_results,
    execute_with_debiasing,
    execute_with_debiasing_and_sharpening,
)
from mitiq.experimental.deb.symmetrization import _construct_variants


def sampling_executor(shots: int = 2000, seed: int = 7):
    """Build a noiseless sampling executor backed by a seeded simulator."""
    simulator = cirq.Simulator(seed=seed)

    def executor(circuit: cirq.Circuit) -> MeasurementResult:
        qubits = sorted(circuit.all_qubits())
        measured = circuit + cirq.measure(*qubits, key="m")
        samples = simulator.run(measured, repetitions=shots)
        return MeasurementResult(samples.measurements["m"].tolist())

    return executor


def noisy_executor(noise: float = 0.05, shots: int = 2000, seed: int = 9):
    """Build a sampling executor with a depolarizing noise model."""
    simulator = cirq.DensityMatrixSimulator(
        seed=seed, noise=cirq.depolarize(noise)
    )

    def executor(circuit: cirq.Circuit) -> MeasurementResult:
        qubits = sorted(circuit.all_qubits())
        measured = circuit + cirq.measure(*qubits, key="m")
        samples = simulator.run(measured, repetitions=shots)
        return MeasurementResult(samples.measurements["m"].tolist())

    return executor


def test_permutations_unscramble_to_the_original_result():
    # An asymmetric, deterministic circuit: q0 -> 1, q1 -> 0 (the two X gates
    # on q1 cancel but keep it in the circuit). Each permuted variant measures
    # a different bitstring, but unscrambling must recover "10" every time, so
    # the debiased distribution is a delta on "10".
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.X(q0), cirq.X(q1), cirq.X(q1)])
    distribution = execute_with_debiasing(
        circuit, sampling_executor(), num_variants=8, random_state=2
    )
    assert distribution == {"10": 1.0}


def test_debiasing_returns_a_normalized_distribution():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])
    distribution = execute_with_debiasing(
        circuit, noisy_executor(), num_variants=6, random_state=5
    )
    assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_debiasing_and_sharpening_returns_a_normalized_distribution():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])
    distribution = execute_with_debiasing_and_sharpening(
        circuit, noisy_executor(), num_variants=6, random_state=5
    )
    assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_combine_results_supports_averaging_and_sharpening():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])
    variants = _construct_variants(circuit, 6, 5)
    executor = noisy_executor()
    results = [executor(variant) for variant, _ in variants]
    permutations = [permutation for _, permutation in variants]

    averaged = combine_results(results, permutations, method="averaging")
    sharpened = combine_results(results, permutations, method="sharpening")
    assert abs(sum(averaged.values()) - 1.0) < 1e-9
    assert abs(sum(sharpened.values()) - 1.0) < 1e-9


def test_combine_results_rejects_unknown_method():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])
    variants = _construct_variants(circuit, 2, 0)
    executor = noisy_executor()
    results = [executor(variant) for variant, _ in variants]
    permutations = [permutation for _, permutation in variants]
    with pytest.raises(ValueError):
        combine_results(results, permutations, method="nope")
