# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for high-level DDD tools."""

import inspect
import logging

import cirq
import numpy as np
from pytest import mark

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES, Executor
from mitiq.ddd import (
    DDDInfo,
    construct_circuits,
    ddd_decorator,
    execute_with_ddd,
    insert_ddd_sequences,
    mitigate_executor,
)
from mitiq.ddd.rules import xx, xyxy, yy
from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.pec.tests.test_pec import (
    batched_executor,
    noiseless_serial_executor,
    serial_executor,
)

# A layer of X gates is useful otherwise amplitude damping is not effective
x_layer = cirq.Circuit(cirq.X.on_each(cirq.LineQubit.range(7)))
circuit_cirq_a = x_layer + cirq.Circuit(
    cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)
)
# Manually append inverse to avoid conversions of SWAP^-1.
circuit_cirq_a += (
    cirq.Circuit(cirq.SWAP(q, q + 1) for q in cirq.LineQubit.range(7)[::-1])
    + x_layer
)

circuit_cirq_b = x_layer[:4] + cirq.Circuit(
    cirq.CNOT(q, q + 1) for q in cirq.LineQubit.range(4)
)
circuit_cirq_b += cirq.inverse(circuit_cirq_b)


def amp_damp_executor(circuit: QPROGRAM, noise: float = 0.005) -> float:
    circuit, _ = convert_to_mitiq(circuit)
    return compute_density_matrix(
        circuit, noise_model_function=cirq.amplitude_damp, noise_level=(noise,)
    )[0, 0].real


@mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@mark.parametrize("circuit", [circuit_cirq_a, circuit_cirq_b])
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_without_noise(circuit_type, circuit, rule):
    """Tests that execute_with_ddd preserves expected results
    in the absence of noise.
    """
    circuit = convert_from_mitiq(circuit, circuit_type)
    true_noiseless_value = 1.0
    unmitigated = noiseless_serial_executor(circuit)
    mitigated = execute_with_ddd(
        circuit,
        executor=noiseless_serial_executor,
        rule=rule,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)
    assert np.isclose(error_unmitigated, error_mitigated)


@mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@mark.parametrize("circuit", [circuit_cirq_a, circuit_cirq_b])
@mark.parametrize("executor", [serial_executor, batched_executor])
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_and_depolarizing_noise(
    circuit_type, circuit, executor, rule
):
    """Tests that with execute_with_ddd the error of a noisy
    expectation value is unchanged with depolarizing noise.
    """
    circuit = convert_from_mitiq(circuit, circuit_type)
    true_noiseless_value = 1.0
    unmitigated = serial_executor(circuit)
    mitigated = execute_with_ddd(
        circuit,
        executor,
        rule=rule,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    # For moment-based depolarizing noise DDD should
    # have no effect (since noise commutes with DDD gates).
    assert np.isclose(error_mitigated, error_unmitigated)


@mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
@mark.parametrize("rule", [xx, yy, xyxy])
def test_execute_with_ddd_and_damping_noise(circuit_type, rule):
    """Tests that with execute_with_ddd the error of a noisy
    expectation value is unchanged with depolarizing noise.
    """
    circuit = convert_from_mitiq(circuit_cirq_a, circuit_type)
    true_noiseless_value = 1.0
    unmitigated = amp_damp_executor(circuit)
    mitigated = execute_with_ddd(
        circuit,
        amp_damp_executor,
        rule=rule,
    )
    error_unmitigated = abs(unmitigated - true_noiseless_value)
    error_mitigated = abs(mitigated - true_noiseless_value)

    assert error_mitigated < error_unmitigated


@mark.parametrize("executor", [serial_executor, batched_executor])
def test_execute_with_ddd_with_num_trials(executor):
    """Tests the option num_trials of execute_with_ddd."""
    executor = Executor(executor)
    mitigated_1 = execute_with_ddd(
        circuit_cirq_a,
        executor,
        rule=xx,
        num_trials=1,
    )
    assert executor.calls_to_executor == 1
    assert len(executor.executed_circuits) == 1

    mitigated_2 = execute_with_ddd(
        circuit_cirq_a,
        executor,
        rule=xx,
        num_trials=2,
    )
    # Note executor contains the history of both experiments
    if executor.can_batch:
        assert executor.calls_to_executor == 2
    else:
        assert executor.calls_to_executor == 3
    assert len(executor.executed_circuits) == 3

    # For deterministic DDD sequences num_trials is irrelevant
    assert np.isclose(mitigated_1, mitigated_2)


def test_execute_with_ddd_with_full_output():
    """Tests the option full_output of execute_with_ddd."""
    executor = Executor(noiseless_serial_executor)

    ddd_value, ddd_data = execute_with_ddd(
        circuit_cirq_a,
        executor,
        rule=xx,
        num_trials=2,
        full_output=True,
    )
    assert len(executor.executed_circuits) == 2
    assert len(ddd_data["circuits_with_ddd"]) == 2
    assert len(ddd_data["ddd_trials"]) == 2
    assert ddd_data["ddd_value"] == ddd_value
    # For a deterministic rule
    assert ddd_data["ddd_trials"][0] == ddd_data["ddd_trials"][1]


def test_mitigate_executor_ddd():
    ddd_value = execute_with_ddd(
        circuit_cirq_a,
        serial_executor,
        rule=xx,
    )
    mitigated_executor = mitigate_executor(serial_executor, rule=xx)
    assert np.isclose(mitigated_executor(circuit_cirq_a), ddd_value)

    batched_mitigated_executor = mitigate_executor(batched_executor, rule=xx)
    assert np.isclose(
        *batched_mitigated_executor([circuit_cirq_a] * 3), ddd_value
    )


def test_ddd_decorator():
    ddd_value = execute_with_ddd(
        circuit_cirq_a,
        serial_executor,
        rule=xx,
    )

    @ddd_decorator(rule=xx)
    def my_serial_executor(circuit):
        return serial_executor(circuit)

    assert np.isclose(my_serial_executor(circuit_cirq_a), ddd_value)

    # Test batched executors too
    @ddd_decorator(rule=xx)
    def my_batched_executor(circuits) -> list[float]:
        return batched_executor(circuits)

    assert np.isclose(*my_batched_executor([circuit_cirq_a]), ddd_value)


def test_ddd_decorator_with_rule_args():
    """Tests that rule_args option is working."""
    unmitigated = amp_damp_executor(circuit_cirq_a)

    @ddd_decorator(rule=xx)
    def exec_xx(circuit):
        return amp_damp_executor(circuit)

    mitigated = exec_xx(circuit_cirq_a)
    assert unmitigated < mitigated

    @ddd_decorator(rule=xx, rule_args={"spacing": 100})
    def exec_xx_large_spacing(circuit):
        return amp_damp_executor(circuit)

    mitigated_large_spacing = exec_xx_large_spacing(circuit_cirq_a)
    # With very large spacing DDD sequences should not fit in the circuit.
    # So we should get the same result as without mitigation.
    assert np.isclose(unmitigated, mitigated_large_spacing)

    @ddd_decorator(rule=xx, rule_args={"spacing": 1})
    def exec_xx_small_spacing(circuit):
        return amp_damp_executor(circuit)

    mitigated_small_spacing = exec_xx_small_spacing(circuit_cirq_a)
    # With small spacing results can be better or worst than default spacing.
    # What is important to test is getting different results.
    assert not np.isclose(unmitigated, mitigated_small_spacing)
    assert not np.isclose(mitigated_large_spacing, mitigated_small_spacing)


@mark.parametrize("num_trials", [1, 10, 20, 30])
def test_num_trials_generates_circuits(num_trials: int):
    """Test that the number of generated circuits follows num_trials."""

    circuits = construct_circuits(
        circuit_cirq_a, rule=xx, num_trials=num_trials
    )

    assert num_trials == len(circuits)


def test_construct_circuits_return_info_default_is_list_only():
    """Default API remains a plain list of circuits (no tuple)."""
    circuits = construct_circuits(circuit_cirq_a, rule=xx, num_trials=3)
    assert isinstance(circuits, list)
    assert all(isinstance(c, QPROGRAM) for c in circuits)


def test_construct_circuits_return_info_true():
    """Optional return_info reports one DDDInfo per trial."""
    circuits, infos = construct_circuits(
        circuit_cirq_a, rule=xx, num_trials=3, return_info=True
    )

    assert len(circuits) == 3
    assert len(infos) == 3
    assert isinstance(infos, tuple)
    assert all(isinstance(info, DDDInfo) for info in infos)
    assert circuits == construct_circuits(
        circuit_cirq_a, rule=xx, num_trials=3
    )


def test_construct_circuits_and_insert_have_no_mutable_defaults():
    """Public kwargs use None/immutable defaults, not a shared list/dict."""
    for func in (construct_circuits, insert_ddd_sequences, execute_with_ddd):
        for param in inspect.signature(func).parameters.values():
            default = param.default
            if default is inspect.Parameter.empty:
                continue
            assert not isinstance(default, (list, dict, set)), param.name


def test_construct_circuits_return_info_not_shared_across_calls():
    """Tuple info from one call must not leak into the next."""
    q = cirq.LineQubit(0)
    empty = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.H(q))
    idle = cirq.Circuit(cirq.H(q), cirq.I(q), cirq.I(q), cirq.H(q))

    _, infos_empty = construct_circuits(empty, rule=xx, return_info=True)
    _, infos_idle = construct_circuits(idle, rule=xx, return_info=True)
    _, infos_empty_again = construct_circuits(empty, rule=xx, return_info=True)

    assert infos_empty[0].num_idle_windows == 0
    assert infos_idle[0].num_idle_windows == 1
    assert infos_empty_again[0].num_idle_windows == 0
    assert infos_idle[0].idle_window_lengths == (2,)


def test_construct_circuits_logs_insertion(caplog):
    """INFO logging on mitiq.ddd reports idle windows and insertions."""
    caplog.set_level(logging.INFO, logger="mitiq.ddd")
    construct_circuits(circuit_cirq_a, rule=xx, num_trials=1)

    assert "idle windows" in caplog.text
    assert "inserted" in caplog.text
    assert "sequences" in caplog.text


def test_construct_circuits_logs_each_trial(caplog):
    """Multiple trials log one line each, not a single combined message."""
    caplog.set_level(logging.INFO, logger="mitiq.ddd")
    construct_circuits(circuit_cirq_a, rule=xx, num_trials=3)

    trial_records = [
        rec
        for rec in caplog.records
        if rec.name == "mitiq.ddd" and "DDD trial" in rec.getMessage()
    ]
    assert len(trial_records) == 3
    assert "1/3" in trial_records[0].getMessage()
    assert "3/3" in trial_records[2].getMessage()


def test_construct_circuits_logs_when_nothing_inserted(caplog):
    """Logging still reports zeros when DDD finds no idle windows."""
    caplog.set_level(logging.INFO, logger="mitiq.ddd")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.H(q))
    construct_circuits(circuit, rule=xx)

    assert "found 0 idle windows; inserted 0 sequences" in caplog.text


def test_insert_ddd_sequences_does_not_log(caplog):
    """Logging lives on construct_circuits only (no duplicate noise)."""
    caplog.set_level(logging.INFO, logger="mitiq.ddd")
    insert_ddd_sequences(circuit_cirq_a, rule=xx)
    insert_ddd_sequences(circuit_cirq_a, rule=xx, return_info=True)

    assert caplog.records == []
