# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for readout confusion inversion."""

from functools import partial

import cirq
import numpy as np
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)

from mitiq import Executor, MeasurementResult, Observable, PauliString
from mitiq.interface.mitiq_cirq import sample_bitstrings
from mitiq.raw import execute as raw_execute
from mitiq.rem import execute_with_rem, mitigate_executor, rem_decorator
from mitiq.rem.inverse_confusion_matrix import (
    generate_inverse_confusion_matrix,
)

# Default qubit register and circuit for unit tests
qreg = [cirq.LineQubit(i) for i in range(2)]
circ_with_measurements = cirq.Circuit(
    cirq.ops.X.on_each(*qreg), cirq.measure_each(*qreg)
)
circ = cirq.Circuit(cirq.ops.X.on_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))


def noisy_readout_executor(
    circuit, p0: float = 0.01, p1: float = 0.01, shots: int = 8192
) -> MeasurementResult:
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)

    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())),
        qubit_indices=tuple(
            # q[2:-1] is necessary to convert "q(number)" into "number"
            int(q[2:-1])
            for k in result.measurements.keys()
            for q in k.split(",")
        ),
    )


npX = np.array([[0, 1], [1, 0]])
"""Defines the sigma_x Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

npZ = np.array([[1, 0], [0, -1]])
"""Defines the sigma_z Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""


# An invalid executor for unit tests since it returns an expectation
def invalid_executor(circuit) -> float:
    wavefunction = circuit.final_state_vector(
        ignore_terminal_measurements=True
    )
    return np.real(wavefunction.conj().T @ np.kron(npX, npZ) @ wavefunction)


def test_rem_invalid_executor():
    identity = np.identity(4)
    with pytest.raises(TypeError, match="not of type MeasurementResult"):
        execute_with_rem(
            circ,
            invalid_executor,
            observable=None,
            inverse_confusion_matrix=identity,
        )


def test_rem_identity():
    executor = partial(sample_bitstrings, noise_level=(0,))
    identity = np.identity(4)
    result = execute_with_rem(
        circ, executor, observable, inverse_confusion_matrix=identity
    )
    assert np.isclose(result, -2.0)


def test_rem_with_matrix():
    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    unmitigated = raw_execute(circ, noisy_executor, observable)
    assert np.isclose(unmitigated, 2.0)

    inverse_confusion_matrix = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    mitigated = execute_with_rem(
        circ,
        Executor(noisy_executor),
        observable,
        inverse_confusion_matrix=inverse_confusion_matrix,
    )
    assert np.isclose(mitigated, -2.0)


def test_rem_with_invalid_matrix():
    executor = partial(sample_bitstrings, noise_level=(0,))
    identity = np.identity(2)
    with pytest.raises(ValueError):
        execute_with_rem(
            circ, executor, observable, inverse_confusion_matrix=identity
        )


def test_doc_is_preserved():
    """Tests that the doc of the original executor is preserved."""

    def first_executor(circuit):
        """Doc of the original executor."""
        return MeasurementResult()

    identity = np.identity(4)

    mit_executor = mitigate_executor(
        first_executor, inverse_confusion_matrix=identity
    )
    assert mit_executor.__doc__ == first_executor.__doc__

    @rem_decorator(inverse_confusion_matrix=identity)
    def second_executor(circuit):
        """Doc of the original executor."""
        return 0

    assert second_executor.__doc__ == first_executor.__doc__


@pytest.mark.parametrize(
    "p0, p1, atol",
    [
        (0, 0, 0),
        (1, 1, 0),
        (
            1,
            0,
            2.1,
        ),  # catastrophic failure from all 0s state, which is why atol=2.1
        (
            0,
            1,
            0,
        ),  # undetectable: results in all 1s state, which is the desired state
        (0.3, 0.1, 0),
        (0.1, 0.3, 0),
        (0.001, 0.001, 1e-3),
        (0.02, 0.04, 0),
        (0.3, 0.7, 0),
    ],
)
def test_mitigate_executor(p0, p1, atol):
    true_rem_value = -2.0

    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)

    num_qubits = len(qreg)
    inverse_confusion_matrix = generate_inverse_confusion_matrix(
        num_qubits, p0=p0, p1=p1
    )

    base = raw_execute(circ, noisy_executor, observable)

    mitigated_executor = mitigate_executor(
        noisy_executor,
        inverse_confusion_matrix=inverse_confusion_matrix,
    )
    rem_value = raw_execute(circ, mitigated_executor, observable)
    assert abs(true_rem_value - rem_value) <= abs(
        true_rem_value - base
    ) or np.isclose(
        abs(true_rem_value - rem_value),
        abs(true_rem_value - base),
        atol=atol,
    )


def test_rem_decorator():
    true_rem_value = -2.0

    num_qubits = len(qreg)

    # test with an executor that completely flips results
    p0 = 1
    p1 = 1
    inverse_confusion_matrix = generate_inverse_confusion_matrix(
        num_qubits, p0=p0, p1=p1
    )

    @rem_decorator(inverse_confusion_matrix=inverse_confusion_matrix)
    def noisy_readout_decorated_executor(qp) -> MeasurementResult:
        return noisy_readout_executor(qp, p0=p0, p1=p1)

    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)

    base = raw_execute(circ, noisy_executor, observable)

    rem_value = raw_execute(circ, noisy_readout_decorated_executor, observable)
    assert abs(true_rem_value - rem_value) < abs(true_rem_value - base)


def test_rem_decorator_batched():
    circuits = [circ, 3 * circ]
    true_values = [-2.0, -2.0]
    num_qubits = len(qreg)
    p0 = 0.2
    p1 = 0.1
    inverse_confusion_matrix = generate_inverse_confusion_matrix(
        num_qubits, p0=p0, p1=p1
    )

    @rem_decorator(inverse_confusion_matrix=inverse_confusion_matrix)
    def noisy_readout_batched(circuits) -> list[MeasurementResult]:
        return [noisy_readout_executor(c, p0=p0, p1=p1) for c in circuits]

    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    base_values = [
        raw_execute(c, noisy_executor, observable) for c in circuits
    ]
    rem_values = Executor(noisy_readout_batched).evaluate(circuits, observable)
    for true_val, base, rem_val in zip(true_values, base_values, rem_values):
        assert abs(true_val - rem_val) < abs(true_val - base)
        assert np.isclose(true_val, rem_val, atol=0.05)
