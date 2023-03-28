# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import pytest
import json
import cirq
import qiskit

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES
from mitiq.calibration import ZNESettings, Settings
from mitiq.calibration.settings import MitigationTechnique, BenchmarkProblem
from mitiq.raw import execute
from mitiq.pec import execute_with_pec
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory, LinearFactory


def test_MitigationTechnique():
    pec_enum = MitigationTechnique.PEC
    assert pec_enum.mitigation_function == execute_with_pec
    assert pec_enum.name == "PEC"

    raw_enum = MitigationTechnique.RAW
    assert raw_enum.mitigation_function == execute
    assert raw_enum.name == "RAW"


def test_basic_settings():
    settings = Settings(
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
    circuits = settings.make_problems()
    assert len(circuits) == 1
    ghz_problem = circuits[0]
    assert len(ghz_problem.circuit) == 2
    assert ghz_problem.two_qubit_gate_count == 1
    assert ghz_problem.ideal_distribution == {"00": 0.5, "11": 0.5}

    strategies = settings.make_strategies()
    num_strategies = 4
    assert len(strategies) == num_strategies

    strategy_summary = str(strategies[0]).replace("'", '"')
    assert isinstance(json.loads(strategy_summary), dict)


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


def test_ZNESettings():
    circuits = ZNESettings.make_problems()
    strategies = ZNESettings.make_strategies()

    repr_string = repr(circuits[0])
    assert all(
        s in repr_string
        for s in ("type", "ideal_distribution", "num_qubits", "circuit_depth")
    )

    assert len(circuits) == 3
    assert len(strategies) == 2 * 2 * 2


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
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
