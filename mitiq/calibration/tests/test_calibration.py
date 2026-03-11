# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Clifford data regression top-level API."""

import re
from functools import partial
from unittest.mock import MagicMock, call

import cirq
import numpy as np
import pytest

from mitiq import SUPPORTED_PROGRAM_TYPES, Executor, MeasurementResult
from mitiq.benchmarks import generate_rb_circuits
from mitiq.calibration import Calibrator, Settings, execute_with_mitigation
from mitiq.calibration.calibrator import (
    ExperimentResults,
    MissingResultsError,
    convert_to_expval_executor,
)
from mitiq.calibration.settings import Strategy
from mitiq.interface import convert_to_mitiq
from mitiq.pec.representations import (
    represent_operation_with_local_biased_noise,
    represent_operation_with_local_depolarizing_noise,
)
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_global

light_zne_settings = Settings(
    [
        {
            "circuit_type": "mirror",
            "num_qubits": 1,
            "circuit_depth": 1,
        },
        {
            "circuit_type": "rotated_rb",
            "num_qubits": 1,
            "circuit_depth": 10,
            "theta": np.pi / 3,
        },
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0]),
        },
    ],
)

light_pec_settings = Settings(
    [
        {
            "circuit_type": "mirror",
            "num_qubits": 1,
            "circuit_depth": 1,
        },
        {
            "circuit_type": "mirror",
            "num_qubits": 2,
            "circuit_depth": 1,
        },
    ],
    strategies=[
        {
            "technique": "pec",
            "representation_function": (
                represent_operation_with_local_biased_noise
            ),
            "is_qubit_dependent": False,
            "noise_level": 0.01,
            "noise_bias": 0.1,
            "num_samples": 200,
            "force_run_all": False,
        },
    ],
)


settings = Settings(
    [
        {
            "circuit_type": "ghz",
            "num_qubits": 2,
            "circuit_depth": 10,
        },
        {"circuit_type": "rb", "num_qubits": 2, "circuit_depth": 10},
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
    ],
)

strategies = settings.make_strategies()
problems = settings.make_problems()


def damping_execute(circuit, noise_level=0.001):
    circuit = circuit.with_noise(cirq.amplitude_damp(noise_level))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=100)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


def depolarizing_execute(circuit, noise_level=0.01):
    circuit = circuit.with_noise(cirq.depolarize(noise_level))

    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=100)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


def non_cirq_damping_execute(circuit):
    # Ensure test circuits are converted to user's frontend by the Calibrator
    assert not isinstance(circuit, cirq.Circuit)
    circuit, circuit_type = convert_to_mitiq(circuit)
    # Pennylane and Braket conversions discard measurements so we re-append
    if circuit_type in ["braket", "pennylane"]:
        circuit.append(cirq.measure(q) for q in circuit.all_qubits())
    return damping_execute(circuit)


def non_cirq_depolarizing_execute(circuit):
    # Ensure test circuits are converted to user's frontend by the Calibrator
    assert not isinstance(circuit, cirq.Circuit)
    circuit, circuit_type = convert_to_mitiq(circuit)
    # Pennylane and Braket conversions discard measurements so we re-append
    if circuit_type in ["braket", "pennylane"]:
        circuit.append(cirq.measure(q) for q in circuit.all_qubits())
    return depolarizing_execute(circuit)


def test_ZNE_workflow():
    cal = Calibrator(damping_execute, frontend="cirq")
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 8 * 3 * 4 + 4, "ideal_executions": 0}

    cal.run()
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


def test_PEC_workflow():
    executor = MagicMock(return_value=MeasurementResult(["111", "110"]))
    cal = Calibrator(executor, frontend="cirq", settings=light_pec_settings)
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 2 * 200 + 2, "ideal_executions": 0}

    cal.run()
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_ZNE_workflow_multi_platform(circuit_type):
    """Test the ZNE workflow runs with all possible frontends."""
    # Only test frontends different from cirq
    if circuit_type == "cirq":
        return

    cal = Calibrator(
        non_cirq_damping_execute,
        frontend=circuit_type,
        settings=light_zne_settings,
    )
    cost = cal.get_cost()
    assert cost == {"noisy_executions": 2 * 2 + 2, "ideal_executions": 0}
    cal.run()
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_PEC_workflow_multi_platform(circuit_type):
    """Test the PEC workflow runs with all possible frontends."""
    # Only test frontends different from cirq
    if circuit_type == "cirq":
        return

    executor = MagicMock(return_value=MeasurementResult(["000", "001"]))
    cal = Calibrator(
        executor, frontend=circuit_type, settings=light_pec_settings
    )
    cost = cal.get_cost()
    num_executions = 2 * 200 + 2
    assert cost == {"noisy_executions": num_executions, "ideal_executions": 0}
    cal.run()
    assert executor.call_count == num_executions
    assert isinstance(cal.results, ExperimentResults)
    assert isinstance(cal.best_strategy(), Strategy)


def test_get_cost():
    cal = Calibrator(damping_execute, frontend="cirq", settings=settings)
    cost = cal.get_cost()
    expected_cost = 2 * (12 + 1)  # circuits * (num_experiments + 1)
    assert cost["noisy_executions"] == expected_cost
    assert cost["ideal_executions"] == 0


def test_best_strategy():
    test_strategy_settings = Settings(
        benchmarks=[
            {"circuit_type": "ghz", "num_qubits": 2, "circuit_depth": 10},
            {
                "circuit_type": "mirror",
                "num_qubits": 2,
                "circuit_depth": 10,
                "circuit_seed": 1,
            },
        ],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 2.0, 3.0]),
            },
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 3.0, 5.0]),
            },
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": LinearFactory([1.0, 2.0, 3.0]),
            },
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": LinearFactory([1.0, 3.0, 5.0]),
            },
        ],
    )

    cal = Calibrator(
        damping_execute, frontend="cirq", settings=test_strategy_settings
    )
    cal.run()
    assert not np.isnan(cal.results.mitigated).all()

    strategy = cal.best_strategy()
    assert strategy.technique.name == "ZNE"


def test_convert_to_expval_executor():
    noiseless_bitstring_executor = Executor(
        partial(damping_execute, noise_level=0)
    )
    noiseless_expval_executor = convert_to_expval_executor(
        noiseless_bitstring_executor, bitstring="00"
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    rb_circuit_expval = noiseless_expval_executor.evaluate(rb_circuit)
    assert np.isclose(rb_circuit_expval, 1.0)


def test_execute_with_mitigation(monkeypatch):
    cal = Calibrator(damping_execute, frontend="cirq")

    expval_executor = convert_to_expval_executor(
        Executor(damping_execute), bitstring="00"
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    # override the def of `input` so that it returns "yes"
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    expval = execute_with_mitigation(
        rb_circuit, expval_executor, calibrator=cal
    )
    assert isinstance(expval, float)
    assert 0 <= expval <= 1.5


def test_cal_execute_w_mitigation():
    cal = Calibrator(damping_execute, frontend="cirq")
    cal.run()

    expval_executor = convert_to_expval_executor(
        Executor(damping_execute), bitstring="00"
    )
    rb_circuit = generate_rb_circuits(2, 10)[0]
    rb_circuit.append(cirq.measure(rb_circuit.all_qubits()))

    expval = cal.execute_with_mitigation(rb_circuit, expval_executor)
    assert isinstance(expval, float)
    assert 0 <= expval <= 1.5


def test_double_run():
    cal = Calibrator(damping_execute, frontend="cirq")
    cal.run()
    cal.run()


def test_ExtrapolationResults_add_result():
    er = ExperimentResults(strategies, problems)
    assert er.is_missing_data()
    er.add_result(
        strategies[0],
        problems[0],
        ideal_val=1.0,
        noisy_val=0.8,
        mitigated_val=0.9,
    )
    assert er.is_missing_data()
    er.mitigated = np.ones((2, 4))
    er.ideal = np.ones((2, 4))
    er.noisy = np.ones((2, 4))
    assert not er.is_missing_data()


def test_ExtrapolationResults_errors():
    num_strategies, num_problems = 4, 2
    er = ExperimentResults(strategies, problems)
    er.mitigated = np.random.random((num_strategies, num_problems))
    er.ideal = np.random.random((num_strategies, num_problems))

    assert (er.squared_errors() > 0).all()


def test_ExtrapolationResults_best_strategy():
    num_strategies, num_problems = 4, 2
    er = ExperimentResults(strategies, problems)
    er.mitigated = np.zeros((num_strategies, num_problems))
    er.mitigated[3, 1] = 0.8
    er.ideal = np.ones((num_strategies, num_problems))
    assert er.best_strategy_id() == 3


light_combined_settings = Settings(
    benchmarks=light_zne_settings.benchmarks,
    strategies=[
        {
            "technique": "pec",
            "representation_function": (
                represent_operation_with_local_depolarizing_noise
            ),
            "is_qubit_dependent": False,
            "noise_level": 0.001,
            "num_samples": 200,
        },
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0]),
        },
    ],
)


def test_ExtrapolationResults_performance_str():
    nerr = 0.9876543
    merr = 0.1234567
    perf_str = ExperimentResults._performance_str(nerr, merr)
    mandatory_lines_count = 0
    for line in perf_str.split("\n"):
        if "Noisy error: " in line:
            assert line.split("Noisy error: ")[1] == str(round(nerr, 4))
            mandatory_lines_count += 1
        if "Mitigated error: " in line:
            assert line.split("Mitigated error: ")[1] == str(round(merr, 4))
            mandatory_lines_count += 1
        if "Improvement factor: " in line:
            assert line.split("Improvement factor: ")[1] == str(
                round(nerr / merr, 4)
            )
            mandatory_lines_count += 1
    assert mandatory_lines_count == 3


def test_Calibrator_log_argument():
    cal = Calibrator(
        damping_execute, frontend="cirq", settings=Settings([], [])
    )
    cal.results = MagicMock()
    cal.run()
    cal.results.log_results_flat.assert_not_called()

    cal.run(log="flat")
    cal.results.log_results_flat.assert_called_once()

    cal.run(log="cartesian")
    cal.results.log_results_cartesian.assert_called_once()

    with pytest.raises(ValueError):
        cal.run(log="foo")


def test_ExtrapolationResults_logging_results_flat(capfd):
    settings = light_zne_settings
    problems = settings.make_problems()
    strategies = settings.make_strategies()
    results = ExperimentResults(strategies, problems)
    perf_str = "performance result"
    results._performance_str = MagicMock(return_value=perf_str)
    for s in strategies:
        for p in problems:
            results.add_result(
                s, p, ideal_val=1.0, noisy_val=0.5, mitigated_val=0.6
            )
    results.log_results_flat()
    captured = capfd.readouterr()
    results._performance_str.assert_has_calls(
        [call(0.5, 0.4)] * len(strategies) * len(problems)
    )
    for s in strategies:
        for line in str(s).split():
            assert line in captured.out
    for p in problems:
        for line in str(p).split():
            assert line in captured.out
    # Here we make sure that we have len(strategies) * len(problems)
    # performance results lines in the output. One performance result
    # per strategy/problem combination.
    perf_count = 0
    for line in captured.out.split("\n"):
        if perf_str in line:
            perf_count += 1
    assert perf_count == (len(strategies) * len(problems))


def test_ExtrapolationResults_logging_results_cartesian(capfd):
    settings = light_zne_settings
    problems = settings.make_problems()
    strategies = settings.make_strategies()
    results = ExperimentResults(strategies, problems)
    perf_str = "performance result"
    results._performance_str = MagicMock(return_value=perf_str)
    for s in strategies:
        for p in problems:
            results.add_result(
                s, p, ideal_val=1.0, noisy_val=0.5, mitigated_val=0.6
            )
    results.log_results_cartesian()
    captured = capfd.readouterr()
    results._performance_str.assert_has_calls(
        [call(0.5, 0.4)] * len(strategies) * len(problems)
    )
    for s in strategies:
        for line in str(s).split():
            assert line in captured.out
    for p in problems:
        for line in str(p).split():
            assert line in captured.out
    # Here we make sure the output contains multiple columns.
    # One column per problem
    for line in captured.out.split("\n"):
        if perf_str in line:
            assert len(re.findall(perf_str, line)) == len(problems)


def test_ExperimentResults_reset_data():
    er = ExperimentResults(strategies, problems)
    er.add_result(
        strategies[1],
        problems[1],
        ideal_val=1.0,
        noisy_val=0.8,
        mitigated_val=0.9,
    )
    assert not np.isnan(er.mitigated).all()
    er.reset_data()
    assert np.isnan(er.mitigated).all()


def test_ExperimentResults_ensure_full():
    er = ExperimentResults(strategies, problems)
    with pytest.raises(MissingResultsError):
        er.ensure_full()


def test_calibrator_custom_ideal_distribution():
    q = cirq.LineQubit(0)
    circ = cirq.Circuit(cirq.X(q))

    settings = Settings(
        benchmarks=[{"circuit_type": "custom", "circuit": circ}],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": LinearFactory([1.0, 2.0]),
            }
        ],
    )

    ideal_exec = Executor(partial(damping_execute, noise_level=0))
    cal = Calibrator(
        damping_execute,
        frontend="cirq",
        settings=settings,
        ideal_executor=ideal_exec,
    )
    problem = cal.problems[0]
    assert problem.ideal_distribution == {"1": 1.0}
