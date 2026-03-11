# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Collector."""

from random import choices

import cirq
import numpy as np
import pyquil
import pytest
from qiskit import QuantumCircuit

from mitiq import MeasurementResult
from mitiq.executor.executor import Executor
from mitiq.interface.mitiq_cirq import (
    compute_density_matrix,
    sample_bitstrings,
)
from mitiq.observable import Observable, PauliString


# Serial / batched executors which return floats.
def executor_batched(circuits, **kwargs) -> list[float]:
    return [
        float(v)
        for v in np.full(
            shape=(len(circuits),),
            fill_value=kwargs.setdefault("return_value", 0.0),
        )
    ]


def executor_batched_unique(circuits) -> list[float]:
    return [executor_serial_unique(circuit) for circuit in circuits]


def executor_serial_unique(circuit) -> float:
    return float(len(circuit))


def executor_serial_typed(*args, **kwargs) -> float:
    return executor_serial(*args, **kwargs)


def executor_serial(*args, **kwargs):
    return kwargs.setdefault("return_value", 0.0)


def executor_pyquil_batched(programs) -> list[float]:
    for p in programs:
        if not isinstance(p, pyquil.Program):
            raise TypeError

    return [0.0] * len(programs)


# Serial / batched executors which return measurements.
def executor_measurements(circuit):
    return sample_bitstrings(circuit, noise_level=(0,))


def executor_measurements_typed(circuit) -> MeasurementResult:
    return sample_bitstrings(circuit, noise_level=(0,))


def executor_measurements_batched(circuits) -> list[MeasurementResult]:
    return [executor_measurements_typed(circuit) for circuit in circuits]


# Serial / batched executors which return density matrices.
def executor_density_matrix(circuit):
    return compute_density_matrix(circuit, noise_level=(0,))


def executor_density_matrix_typed(circuit) -> np.ndarray:
    return compute_density_matrix(circuit, noise_level=(0,))


def executor_density_matrix_batched(circuits) -> list[np.ndarray]:
    return [executor_density_matrix_typed(circuit) for circuit in circuits]


def executor_typed_npfloat32(circuit) -> np.float32:
    return np.float32(3.14)


def test_executor_simple():
    collector = Executor(executor=executor_batched, max_batch_size=10)
    assert collector.can_batch
    assert collector._max_batch_size == 10
    assert collector.calls_to_executor == 0


def test_executors_can_batch():
    assert Executor(executor_batched).can_batch
    assert not Executor(executor_serial_typed).can_batch
    assert not Executor(executor_serial).can_batch
    assert not Executor(executor_measurements_typed).can_batch
    assert not Executor(executor_density_matrix_typed).can_batch
    assert Executor(executor_measurements_batched).can_batch


def test_executor_non_hermitian_observable():
    obs = Observable(PauliString("Z", coeff=1j))

    q = cirq.LineQubit(0)
    circuits = [cirq.Circuit(cirq.I.on(q)), cirq.Circuit(cirq.X.on(q))]

    executor = Executor(executor_measurements_typed)

    with pytest.warns(UserWarning, match="hermitian"):
        executor.evaluate(circuits, obs)


def test_run_executor_single_circuit():
    collector = Executor(executor=executor_serial)
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    results_no_sequence = collector.run(circuit)
    results_sequence = collector.run([circuit])
    assert np.allclose(results_no_sequence, np.zeros(1))
    assert np.allclose(results_no_sequence, results_sequence)


@pytest.mark.parametrize("ncircuits", (5, 10, 25))
@pytest.mark.parametrize("executor", (executor_batched, executor_serial))
def test_run_executor_identical_circuits_batched(ncircuits, executor):
    collector = Executor(executor=executor, max_batch_size=10)
    circuits = [cirq.Circuit(cirq.H(cirq.LineQubit(0)))] * ncircuits
    results = collector.run(circuits, force_run_all=False)

    assert np.allclose(results, np.zeros(ncircuits))
    assert collector.calls_to_executor == 1


@pytest.mark.parametrize("batch_size", (1, 2, 10))
def test_run_executor_nonidentical_pyquil_programs(batch_size):
    collector = Executor(
        executor=executor_pyquil_batched, max_batch_size=batch_size
    )
    assert collector.can_batch

    circuits = [
        pyquil.Program(pyquil.gates.X(0)),
        pyquil.Program(pyquil.gates.H(0)),
    ] * 10
    results = collector.run(circuits, force_run_all=False)

    assert np.allclose(results, np.zeros(len(circuits)))
    if batch_size == 1:
        assert collector.calls_to_executor == 2
    else:
        assert collector.calls_to_executor == 1


@pytest.mark.parametrize("ncircuits", (10, 11, 23))
@pytest.mark.parametrize("batch_size", (1, 2, 5, 50))
def test_run_executor_all_unique(ncircuits, batch_size):
    collector = Executor(executor=executor_batched, max_batch_size=batch_size)
    assert collector.can_batch

    random_state = np.random.RandomState(seed=1)
    circuits = [
        cirq.testing.random_circuit(
            qubits=4, n_moments=10, op_density=1, random_state=random_state
        )
        for _ in range(ncircuits)
    ]
    results = collector.run(circuits)

    assert np.allclose(results, np.zeros(ncircuits))
    assert collector.calls_to_executor == np.ceil(ncircuits / batch_size)


@pytest.mark.parametrize("ncircuits", (5, 21))
@pytest.mark.parametrize("force_run_all", (True, False))
def test_run_executor_force_run_all_serial_executor_identical_circuits(
    ncircuits, force_run_all
):
    collector = Executor(executor=executor_serial)
    assert not collector.can_batch

    circuits = [cirq.Circuit(cirq.H(cirq.LineQubit(0)))] * ncircuits
    results = collector.run(circuits, force_run_all=force_run_all)

    assert np.allclose(results, np.zeros(ncircuits))
    if force_run_all:
        assert collector.calls_to_executor == ncircuits
    else:
        assert collector.calls_to_executor == 1


@pytest.mark.parametrize("s", (50, 100, 150))
@pytest.mark.parametrize("b", (1, 2, 100))
def test_run_executor_preserves_order(s, b):
    collector = Executor(executor=executor_batched_unique, max_batch_size=b)
    assert collector.can_batch

    circuits = [
        cirq.Circuit(cirq.H(cirq.LineQubit(0))),
        cirq.Circuit([cirq.H(cirq.LineQubit(0))] * 2),
    ]
    batch = choices(circuits, k=s)

    assert np.allclose(collector.run(batch), executor_batched_unique(batch))


@pytest.mark.parametrize(
    "execute", [executor_serial_unique, executor_batched_unique]
)
def test_executor_evaluate_float(execute):
    q = cirq.LineQubit(0)
    circuits = [
        cirq.Circuit(cirq.X(q), cirq.M(q)),
        cirq.Circuit(cirq.H(q), cirq.Z(q), cirq.M(q)),
    ]

    executor = Executor(execute)

    executor.evaluate(circuits)

    if execute is executor_serial_unique:
        assert executor.calls_to_executor == 2
    else:
        assert executor.calls_to_executor == 1

    assert executor.executed_circuits == circuits
    assert executor.quantum_results == [2, 3]


@pytest.mark.parametrize(
    "execute", [executor_measurements_typed, executor_measurements_batched]
)
def test_executor_evaluate_measurements(execute):
    obs = Observable(PauliString("Z"))

    q = cirq.LineQubit(0)
    circuits = [cirq.Circuit(cirq.I.on(q)), cirq.Circuit(cirq.X.on(q))]

    executor = Executor(execute)

    results = executor.evaluate(circuits, obs)
    assert np.allclose(results, [1, -1])

    if execute is executor_measurements_typed:
        assert executor.calls_to_executor == 2
    else:
        assert executor.calls_to_executor == 1

    assert executor.executed_circuits[0] == circuits[0] + cirq.measure(q)
    assert executor.executed_circuits[1] == circuits[1] + cirq.measure(q)
    assert executor.quantum_results[0] == executor_measurements_typed(
        circuits[0] + cirq.measure(q)
    )
    assert executor.quantum_results[1] == executor_measurements_typed(
        circuits[1] + cirq.measure(q)
    )
    assert len(executor.quantum_results) == len(circuits)


@pytest.mark.parametrize(
    "execute", [executor_density_matrix_typed, executor_density_matrix_batched]
)
def test_executor_evaluate_density_matrix(execute):
    obs = Observable(PauliString("Z"))

    q = cirq.LineQubit(0)
    circuits = [cirq.Circuit(cirq.I.on(q)), cirq.Circuit(cirq.X.on(q))]

    executor = Executor(execute)

    results = executor.evaluate(circuits, obs)
    assert np.allclose(results, [1, -1])

    if execute is executor_density_matrix_typed:
        assert executor.calls_to_executor == 2
    else:
        assert executor.calls_to_executor == 1

    assert executor.executed_circuits == circuits
    assert np.allclose(
        executor.quantum_results[0], executor_density_matrix_typed(circuits[0])
    )
    assert np.allclose(
        executor.quantum_results[1], executor_density_matrix_typed(circuits[1])
    )
    assert len(executor.quantum_results) == len(circuits)


def test_executor_float_with_observable_typed():
    obs = Observable(PauliString("Z"))
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X.on(q))
    executor = Executor(executor_serial_typed)
    with pytest.raises(
        ValueError,
        match="When using an executor which returns a float-like ",
    ):
        executor.evaluate(circuit, obs)


def test_executor_measurements_without_observable_typed():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X.on(q))
    executor = Executor(executor_measurements_typed)
    with pytest.raises(
        ValueError,
        match="When using a measurement, or bitstring, like result",
    ):
        executor.evaluate(circuit)


def test_executor_density_matrix_without_observable_typed():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X.on(q))
    executor = Executor(executor_density_matrix_typed)
    with pytest.raises(
        ValueError,
        match="When using a density matrix result",
    ):
        executor.evaluate(circuit)


def test_executor_float_not_typed():
    executor = Executor(executor_serial)
    executor_typed = Executor(executor_serial_typed)
    qcirc = QuantumCircuit(1)
    qcirc.h(0)
    assert executor.evaluate(qcirc) == executor_typed.evaluate(qcirc)


def test_executor_density_matrix_not_typed():
    obs = Observable(PauliString("Z"))
    executor = Executor(executor_density_matrix)
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X.on(q))
    with pytest.raises(
        ValueError,
        match="When using an observable",
    ):
        executor.evaluate(circuit, obs)


def test_executor_measurements_not_typed():
    obs = Observable(PauliString("Z"))
    executor = Executor(executor_measurements)
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X.on(q))
    with pytest.raises(
        ValueError,
        match="When using an observable",
    ):
        executor.evaluate(circuit, obs)


def test_executor_unknown_type():
    obs = Observable(PauliString("Z"))
    executor = Executor(executor_typed_npfloat32)
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X.on(q))
    with pytest.raises(
        ValueError,
        match="Could not parse executed results",
    ):
        executor.evaluate(circuit, obs)
