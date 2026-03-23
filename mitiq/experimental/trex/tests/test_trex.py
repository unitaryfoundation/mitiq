# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for TREX (Twirled Readout Error eXtinction)."""

from functools import partial

import cirq
import numpy as np
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import (
    NoisySingleQubitReadoutSampler,
)

from mitiq import MeasurementResult, Observable, PauliString
from mitiq.experimental.trex import (
    combine_results,
    construct_circuits,
    execute_with_trex,
    mitigate_executor,
    trex_decorator,
)
from mitiq.raw import execute as raw_execute

# Default qubit register and circuit for unit tests
qreg = [cirq.LineQubit(i) for i in range(2)]
circ = cirq.Circuit(cirq.ops.X.on_each(*qreg))
observable = Observable(PauliString("ZI"), PauliString("IZ"))


def noisy_readout_executor(
    circuit, p0: float = 0.01, p1: float = 0.01, shots: int = 8192
) -> MeasurementResult:
    """Executor with noisy readout for testing."""
    simulator = NoisySingleQubitReadoutSampler(p0, p1)
    result = simulator.run(circuit, repetitions=shots)

    return MeasurementResult(
        result=np.column_stack(list(result.measurements.values())),
        qubit_indices=tuple(
            int(q[2:-1])
            for k in result.measurements.keys()
            for q in k.split(",")
        ),
    )


def noiseless_executor(circuit, shots: int = 8192) -> MeasurementResult:
    """Noiseless executor for testing."""
    return noisy_readout_executor(circuit, p0=0, p1=0, shots=shots)


# --- Tests for execute_with_trex ---


def test_trex_noiseless():
    """TREX with noiseless executor should return ideal value."""
    result = execute_with_trex(
        circ,
        noiseless_executor,
        observable,
        num_randomizations=16,
        random_state=42,
    )
    assert np.isclose(result, -2.0, atol=0.1)


def test_trex_reduces_error():
    """TREX should reduce readout error compared to unmitigated result."""
    p0, p1 = 0.1, 0.1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    true_value = -2.0

    unmitigated = raw_execute(circ, noisy_executor, observable)
    mitigated = execute_with_trex(
        circ,
        noisy_executor,
        observable,
        num_randomizations=32,
        random_state=42,
    )

    assert abs(true_value - mitigated) <= abs(true_value - unmitigated)


@pytest.mark.parametrize(
    "p0, p1",
    [
        (0.05, 0.05),
        (0.1, 0.1),
        (0.2, 0.2),
        (0.1, 0.2),
        (0.2, 0.1),
    ],
)
def test_trex_various_noise_levels(p0, p1):
    """TREX should reduce error for various readout noise levels."""
    true_value = -2.0
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1, shots=8192)

    unmitigated = raw_execute(circ, noisy_executor, observable)
    mitigated = execute_with_trex(
        circ,
        noisy_executor,
        observable,
        num_randomizations=32,
        random_state=42,
    )

    # TREX should not make things significantly worse.
    assert abs(true_value - mitigated) <= abs(true_value - unmitigated) + 0.1


def test_trex_single_qubit():
    """TREX with a single-qubit observable."""
    q = cirq.LineQubit(0)
    single_circuit = cirq.Circuit(cirq.X(q))
    single_obs = Observable(PauliString("Z"))

    result = execute_with_trex(
        single_circuit,
        noiseless_executor,
        single_obs,
        num_randomizations=16,
        random_state=42,
    )
    assert np.isclose(result, -1.0, atol=0.1)


def test_trex_identity_circuit():
    """TREX on an identity circuit should give expectation +1 for Z."""
    q = cirq.LineQubit(0)
    identity_circuit = cirq.Circuit(cirq.I(q))
    z_obs = Observable(PauliString("Z"))

    result = execute_with_trex(
        identity_circuit,
        noiseless_executor,
        z_obs,
        num_randomizations=16,
        random_state=42,
    )
    assert np.isclose(result, 1.0, atol=0.1)


def test_trex_full_output():
    """Test full_output mode returns dict with expected keys."""
    result, data = execute_with_trex(
        circ,
        noiseless_executor,
        observable,
        num_randomizations=8,
        random_state=42,
        full_output=True,
    )

    assert isinstance(result, float)
    assert isinstance(data, dict)
    assert "trex_value" in data
    assert "twirled_circuits" in data
    assert "calibration_circuits" in data
    assert "twirled_results" in data
    assert "calibration_results" in data
    assert "randomization_strings" in data
    assert data["trex_value"] == result


def test_trex_reproducibility():
    """Results should be reproducible with same random_state."""
    result1 = execute_with_trex(
        circ,
        noiseless_executor,
        observable,
        num_randomizations=8,
        random_state=123,
    )
    result2 = execute_with_trex(
        circ,
        noiseless_executor,
        observable,
        num_randomizations=8,
        random_state=123,
    )
    assert result1 == result2


# --- Tests for construct_circuits ---


def test_construct_circuits_count():
    """Verify correct number of circuits generated."""
    num_rand = 10
    twirled, calib, strings = construct_circuits(
        circ, observable, num_randomizations=num_rand, random_state=42
    )

    num_groups = observable.ngroups
    assert len(twirled) == num_groups * num_rand
    assert len(calib) == num_rand
    assert len(strings) == num_rand


def test_construct_circuits_have_measurements():
    """All generated circuits should have measurement gates."""
    twirled, calib, _ = construct_circuits(
        circ, observable, num_randomizations=4, random_state=42
    )

    for circuit in twirled + calib:
        has_measurement = any(
            cirq.is_measurement(op) for op in circuit.all_operations()
        )
        assert has_measurement


def test_construct_circuits_randomization_strings_are_binary():
    """Randomization strings should be binary arrays."""
    _, _, strings = construct_circuits(
        circ, observable, num_randomizations=10, random_state=42
    )

    for s in strings:
        assert set(s).issubset({0, 1})


# --- Tests for combine_results ---


def test_combine_results_noiseless():
    """combine_results should recover ideal value for noiseless results."""
    num_rand = 16
    twirled, calib, strings = construct_circuits(
        circ, observable, num_randomizations=num_rand, random_state=42
    )

    all_circuits = twirled + calib
    results = [noiseless_executor(c) for c in all_circuits]

    n_twirled = len(twirled)
    twirled_results = results[:n_twirled]
    calibration_results = results[n_twirled:]

    value = combine_results(
        twirled_results, calibration_results, strings, observable
    )
    assert np.isclose(value, -2.0, atol=0.15)


# --- Tests for mitigate_executor ---


def test_mitigate_executor():
    """mitigate_executor should return a callable that reduces error."""
    true_value = -2.0
    p0, p1 = 0.1, 0.1
    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)

    base = raw_execute(circ, noisy_executor, observable)

    mitigated = mitigate_executor(
        noisy_executor,
        observable,
        num_randomizations=32,
        random_state=42,
    )
    trex_value = mitigated(circ)

    assert abs(true_value - trex_value) <= abs(true_value - base) + 0.1


def test_mitigate_executor_preserves_doc():
    """The docstring of the original executor should be preserved."""

    def my_executor(circuit) -> MeasurementResult:
        """My custom executor docstring."""
        return noiseless_executor(circuit)

    mitigated = mitigate_executor(
        my_executor,
        observable,
        num_randomizations=4,
        random_state=42,
    )
    assert mitigated.__doc__ == my_executor.__doc__


# --- Tests for trex_decorator ---


def test_trex_decorator():
    """trex_decorator should produce a callable that reduces error."""
    true_value = -2.0
    p0, p1 = 0.1, 0.1

    @trex_decorator(
        observable,
        num_randomizations=32,
        random_state=42,
    )
    def decorated_executor(circuit) -> MeasurementResult:
        return noisy_readout_executor(circuit, p0=p0, p1=p1)

    noisy_executor = partial(noisy_readout_executor, p0=p0, p1=p1)
    base = raw_execute(circ, noisy_executor, observable)
    trex_value = decorated_executor(circ)

    assert abs(true_value - trex_value) <= abs(true_value - base) + 0.1


def test_trex_decorator_preserves_doc():
    """The docstring of the original executor should be preserved."""

    @trex_decorator(
        observable,
        num_randomizations=4,
        random_state=42,
    )
    def my_executor(circuit) -> MeasurementResult:
        """My custom executor docstring."""
        return noiseless_executor(circuit)

    assert my_executor.__doc__ == "My custom executor docstring."


def test_construct_circuits_with_random_state_object():
    """construct_circuits should accept a RandomState object directly."""
    rng = np.random.RandomState(42)
    twirled, calib, strings = construct_circuits(
        circ, observable, num_randomizations=4, random_state=rng
    )
    assert len(twirled) == observable.ngroups * 4
    assert len(calib) == 4


def test_mitigate_executor_with_executor_object():
    """mitigate_executor should accept an Executor object."""
    from mitiq.executor.executor import Executor

    executor_obj = Executor(noiseless_executor)
    mitigated = mitigate_executor(
        executor_obj,
        observable,
        num_randomizations=4,
        random_state=42,
    )
    result = mitigated(circ)
    assert np.isclose(result, -2.0, atol=0.2)


def test_mitigate_executor_batch():
    """mitigate_executor should work with a batching executor."""

    def batch_executor(circuits) -> list[MeasurementResult]:
        return [noiseless_executor(c) for c in circuits]

    mitigated = mitigate_executor(
        batch_executor,
        observable,
        num_randomizations=4,
        random_state=42,
    )
    results = mitigated([circ, circ])
    assert len(results) == 2
    for r in results:
        assert np.isclose(r, -2.0, atol=0.2)


def test_combine_results_empty_randomizations():
    """combine_results with no randomizations raises ValueError."""
    with pytest.raises(ValueError, match="At least one randomization"):
        combine_results([], [], [], observable)


def test_combine_results_small_calibration_factor():
    """combine_results falls back when calibration factor is near zero."""
    num_rand = 4
    twirled, calib, strings = construct_circuits(
        circ, observable, num_randomizations=num_rand, random_state=42
    )

    all_circuits = twirled + calib
    results = [noiseless_executor(c) for c in all_circuits]

    n_twirled = len(twirled)
    twirled_results = results[:n_twirled]

    # Craft calibration results where parity on support qubits averages to ~0.
    # Half the shots are 0, half are 1 on the support qubit.
    n_qubits = 2
    half = 4096
    mixed_bits = np.vstack(
        [
            np.zeros((half, n_qubits), dtype=int),
            np.ones((half, n_qubits), dtype=int),
        ]
    )
    fake_calib = [
        MeasurementResult(result=mixed_bits.tolist(), qubit_indices=(0, 1))
        for _ in range(num_rand)
    ]

    # Should not raise; falls back to uncorrected value.
    value = combine_results(twirled_results, fake_calib, strings, observable)
    assert isinstance(value, float)
