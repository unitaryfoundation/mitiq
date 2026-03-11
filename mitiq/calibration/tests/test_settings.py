# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
import json

import cirq
import numpy as np
import pytest
import qiskit

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES
from mitiq.calibration import PEC_SETTINGS, ZNE_SETTINGS, Settings
from mitiq.calibration.settings import (
    BenchmarkProblem,
    MitigationTechnique,
    Strategy,
)
from mitiq.pec import (
    execute_with_pec,
    represent_operation_with_local_depolarizing_noise,
)
from mitiq.raw import execute
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_global

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
                represent_operation_with_local_depolarizing_noise
            ),
            "is_qubit_dependent": False,
            "noise_level": 0.001,
            "num_samples": 200,
        },
    ],
)


light_zne_settings = Settings(
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
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": LinearFactory([1.0, 2.0]),
        },
    ],
)


basic_settings = Settings(
    benchmarks=[
        {
            "circuit_type": "ghz",
            "num_qubits": 2,
            "circuit_depth": 999,
        }
    ],
    strategies=[
        {
            "technique": "zne",
            "scale_noise": fold_global,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "pec",
            "representation_function": (
                represent_operation_with_local_depolarizing_noise
            ),
            "is_qubit_dependent": False,
            "noise_level": 0.001,
            "num_samples": 200,
        },
    ],
)


def test_MitigationTechnique():
    pec_enum = MitigationTechnique.PEC
    assert pec_enum.mitigation_function == execute_with_pec
    assert pec_enum.name == "PEC"

    raw_enum = MitigationTechnique.RAW
    assert raw_enum.mitigation_function == execute
    assert raw_enum.name == "RAW"


def test_BenchmarkProblem_make_problems():
    settings = basic_settings
    problems = settings.make_problems()
    assert len(problems) == 1
    ghz_problem = problems[0]
    assert len(ghz_problem.circuit) == 2
    assert ghz_problem.two_qubit_gate_count == 1
    assert ghz_problem.ideal_distribution == {"00": 0.5, "11": 0.5}


def test_BenchmarkProblem_repr():
    settings = basic_settings
    problems = settings.make_problems()
    problem_repr = repr(problems[0]).replace("'", '"')
    assert isinstance(json.loads(problem_repr), dict)


def test_BenchmarkProblem_str():
    settings = basic_settings
    circuits = settings.make_problems()
    problem = circuits[0]
    lines = str(problem).split("\n")
    problem_dict = problem.to_dict()
    for line in lines:
        [title, value] = line.split(":", 1)
        key = title.lower().replace(" ", "_")
        value = value.strip()
        assert key in problem_dict
        assert value == str(problem_dict[key])
    assert "Ideal distribution: " not in str(problem)


def test_Strategy_repr():
    settings = basic_settings
    strategies = settings.make_strategies()
    strategy_repr = repr(strategies[0]).replace("'", '"')
    assert isinstance(json.loads(strategy_repr), dict)


def test_Strategy_str():
    settings = basic_settings
    strategies = settings.make_strategies()

    strategy_str = str(strategies[0])
    strategy_pretty_dict = strategies[0].to_pretty_dict()
    for line in strategy_str.split("\n"):
        [title, value] = line.split(":")
        key = title.lower().replace(" ", "_")
        value = value.strip()
        assert key in strategy_pretty_dict
        assert value == str(strategy_pretty_dict[key])


def test_Strategy_pretty_dict():
    settings = basic_settings
    strategies = settings.make_strategies()

    strategy_dict = strategies[0].to_dict()
    strategy_pretty_dict = strategies[0].to_pretty_dict()
    if strategy_pretty_dict["technique"] == "ZNE":
        assert strategy_pretty_dict["factory"] == strategy_dict["factory"][:-7]
        assert (
            strategy_pretty_dict["scale_factors"]
            == str(strategy_dict["scale_factors"])[1:-1]
        )
    elif strategy_pretty_dict["technique"] == "PEC":
        assert strategy_pretty_dict["noise_bias"] == strategy_dict.get(
            "noise_bias", "N/A"
        )
        assert (
            strategy_pretty_dict["representation_function"]
            == strategy_dict["representation_function"][25:]
        )


def test_make_circuits_rotated_rb_circuits():
    settings = Settings(
        benchmarks=[
            {
                "circuit_type": "rotated_rb",
                "num_qubits": 1,
                "circuit_depth": 10,
                "theta": np.pi / 3,
            }
        ],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 2.0, 3.0]),
            },
        ],
    )
    problems = settings.make_problems()
    assert len(problems) == 1
    assert problems[0].type == "rotated_rb"


def test_make_circuits_rotated_rb_circuits_invalid_qubits():
    settings = Settings(
        benchmarks=[
            {
                "circuit_type": "rotated_rb",
                "num_qubits": 2,
                "circuit_depth": 10,
                "theta": np.pi / 3,
            }
        ],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 2.0, 3.0]),
            },
        ],
    )
    with pytest.raises(NotImplementedError, match="rotated rb circuits"):
        settings.make_problems()


def test_make_circuits_qv_circuits():
    settings = Settings(
        [
            {
                "circuit_type": "qv",
                "num_qubits": 2,
                "circuit_depth": 999,
            }
        ],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 2.0, 3.0]),
            }
        ],
    )
    with pytest.raises(NotImplementedError, match="quantum volume circuits"):
        settings.make_problems()


def test_make_circuits_invalid_circuit_type():
    settings = Settings(
        [{"circuit_type": "foobar", "num_qubits": 2, "circuit_depth": 999}],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 2.0, 3.0]),
            }
        ],
    )
    with pytest.raises(
        ValueError, match="invalid value passed for `circuit_types`"
    ):
        settings.make_problems()


def test_make_strategies_invalid_technique():
    with pytest.raises(KeyError, match="DESTROY"):
        Settings(
            [{"circuit_types": "shor", "num_qubits": 2, "circuit_depth": 999}],
            strategies=[
                {
                    "technique": "destroy_my_errors",
                    "scale_noise": fold_global,
                    "factory": RichardsonFactory([1.0, 2.0, 3.0]),
                }
            ],
        )


def test_unsupported_technique_error():
    strategy = Strategy(1, MitigationTechnique.RAW, {})
    with pytest.raises(
        ValueError,
        match="Specified technique is not supported by calibration.",
    ):
        strategy.mitigation_function()


def test_ZNE_SETTINGS():
    circuits = ZNE_SETTINGS.make_problems()
    strategies = ZNE_SETTINGS.make_strategies()
    repr_string = repr(circuits[0])
    assert all(
        s in repr_string
        for s in ("type", "ideal_distribution", "num_qubits", "circuit_depth")
    )
    assert len(circuits) == 4
    assert len(strategies) == 2 * 2 * 2


def test_PEC_SETTINGS():
    circuits = PEC_SETTINGS.make_problems()
    strategies = PEC_SETTINGS.make_strategies()
    repr_string = repr(circuits[0])
    assert all(
        s in repr_string
        for s in ("type", "ideal_distribution", "num_qubits", "circuit_depth")
    )
    assert len(circuits) == 4
    assert len(strategies) == 2


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES)
def test_benchmark_problem_class(circuit_type):
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(qubit))
    circuit_with_measurements = circuit.copy()
    circuit_with_measurements.append(cirq.measure(qubit))
    problem = BenchmarkProblem(
        id=7,
        circuit=circuit,
        type="",
        ideal_distribution={},
    )
    assert problem.circuit == circuit
    conv_circ = problem.converted_circuit(circuit_type)
    assert any([isinstance(conv_circ, q) for q in QPROGRAM.__args__])
    # For at least one case, test the circuit is correct and has measurements
    if circuit_type == "qiskit":
        qreg = qiskit.QuantumRegister(1, name="q")
        creg = qiskit.ClassicalRegister(1, name="m0")
        expected = qiskit.QuantumCircuit(qreg, creg)
        expected.x(0)
        expected.measure(0, 0)
        assert conv_circ == expected


def test_settings_make_problems():
    """Test the `make_problems` method of `Settings`"""
    settings = Settings(
        [
            {
                "circuit_type": "w",
                "num_qubits": 2,
            }
        ],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": RichardsonFactory([1.0, 2.0, 3.0]),
            }
        ],
    )

    problems = settings.make_problems()
    assert len(problems) == 1

    ideal_distribution = {"01": 0.5, "10": 0.5}

    problem = problems[0]

    assert problem.ideal_distribution == ideal_distribution
    assert problem.two_qubit_gate_count == 2
    assert problem.num_qubits == 2
    assert problem.circuit_depth == 2


def test_to_dict():
    pec_strategy = light_pec_settings.make_strategies()[0]
    assert pec_strategy.to_dict() == {
        "technique": "PEC",
        "representation_function": (
            "represent_operation_with_local_depolarizing_noise"
        ),
        "is_qubit_dependent": False,
        "noise_level": 0.001,
        "noise_bias": 0,
        "num_samples": 200,
    }

    zne_strategy = light_zne_settings.make_strategies()[0]
    assert zne_strategy.to_dict() == {
        "technique": "ZNE",
        "scale_method": "fold_global",
        "factory": "LinearFactory",
        "scale_factors": [1.0, 2.0],
    }


def test_num_circuits_required_raw_execution():
    undefine_strategy = Strategy(
        id=1,
        technique=MitigationTechnique.RAW,
        technique_params={},
    )
    assert undefine_strategy.num_circuits_required() == 1


def test_custom_circuit_problem():
    q0, q1 = cirq.LineQubit.range(2)
    circ = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    ideal = {"00": 0.5, "11": 0.5}

    settings = Settings(
        benchmarks=[
            {
                "circuit_type": "custom",
                "circuit": circ,
                "ideal_distribution": ideal,
            }
        ],
        strategies=[
            {
                "technique": "zne",
                "scale_noise": fold_global,
                "factory": LinearFactory([1.0, 2.0]),
            }
        ],
    )

    problems = settings.make_problems()
    assert len(problems) == 1
    problem = problems[0]
    assert problem.type == "custom"
    assert problem.circuit == circ
    assert problem.ideal_distribution == ideal
