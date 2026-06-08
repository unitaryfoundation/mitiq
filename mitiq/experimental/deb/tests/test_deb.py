# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import numpy as np

from mitiq import MeasurementResult
from mitiq.experimental.deb.deb import (
    execute_with_debiasing,
    execute_with_debiasing_and_sharpening,
)
from mitiq.experimental.deb.symmetrization import construct_circuits


def sampling_executor(shots: int = 4000, seed: int = 7):
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


def exact_distribution(circuit: cirq.Circuit) -> np.ndarray:
    """Return the exact computational-basis distribution of a circuit."""
    qubits = sorted(circuit.all_qubits())
    state = (
        cirq.Simulator()
        .simulate(circuit, qubit_order=qubits)
        .final_state_vector
    )
    return np.abs(state) ** 2


def test_variants_cancel_on_a_noiseless_simulator():
    # On a noiseless simulator the pre/post Pauli layers cancel, so every
    # variant reproduces the distribution of the original circuit.
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit([cirq.rx(0.7).on(qubit), cirq.T(qubit)])
    reference = exact_distribution(circuit)
    variants = construct_circuits(circuit, num_variants=8, random_state=11)
    for variant in variants:
        np.testing.assert_allclose(
            exact_distribution(variant), reference, atol=1e-9
        )


def test_debiasing_preserves_a_symmetric_distribution():
    # Every Pauli conjugation of H|0> still measures uniformly, so the
    # debiased distribution should stay close to (0.5, 0.5).
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    distribution = execute_with_debiasing(
        circuit, sampling_executor(), num_variants=8, random_state=3
    )
    np.testing.assert_allclose(distribution, [0.5, 0.5], atol=0.05)


def test_debiasing_returns_a_normalized_distribution():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])
    distribution = execute_with_debiasing(
        circuit, sampling_executor(), num_variants=6, random_state=5
    )
    assert distribution.shape == (4,)
    assert np.isclose(distribution.sum(), 1.0)


def test_debiasing_and_sharpening_returns_a_normalized_distribution():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])
    distribution = execute_with_debiasing_and_sharpening(
        circuit, sampling_executor(), num_variants=6, random_state=5
    )
    assert distribution.shape == (4,)
    assert np.isclose(distribution.sum(), 1.0)


def test_debiasing_runs_end_to_end_under_depolarizing_noise():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(qubits[0]), cirq.CNOT(*qubits)])
    distribution = execute_with_debiasing(
        circuit, noisy_executor(), num_variants=6, random_state=2
    )
    assert distribution.shape == (4,)
    assert np.isclose(distribution.sum(), 1.0)
