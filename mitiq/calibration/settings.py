# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, cast

import cirq
import networkx as nx
import numpy as np

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES, Executor
from mitiq.benchmarks import (
    generate_ghz_circuit,
    generate_mirror_circuit,
    generate_rb_circuits,
    generate_rotated_rb_circuits,
    generate_w_circuit,
)
from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.pec import execute_with_pec
from mitiq.pec.representations import (
    represent_operation_with_local_biased_noise,
    represent_operation_with_local_depolarizing_noise,
)
from mitiq.raw import execute
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random, fold_global


class MitigationTechnique(Enum):
    """Simple enum type for handling validation, and providing helper functions
    when accessing mitigation techniques."""

    ZNE = auto()
    PEC = auto()
    RAW = auto()

    @property
    def mitigation_function(self) -> Callable[..., float]:
        if self is MitigationTechnique.ZNE:
            return execute_with_zne
        elif self is MitigationTechnique.PEC:
            return cast(Callable[..., float], execute_with_pec)
        elif self is MitigationTechnique.RAW:
            return execute


calibration_supported_techniques = {
    "ZNE": MitigationTechnique.ZNE,
    "PEC": MitigationTechnique.PEC,
}


@dataclass
class BenchmarkProblem:
    """A dataclass containing information for instances of problems that will
    be run during the calibrations process.

    Args:
        id: A unique numerical id.
        circuit: The circuit to be run.
        type: The type of the circuit (often the name of the algorithm)
        ideal_distribution: The ideal probability distribution after applying
            ``circuit``.
    """

    id: int
    circuit: cirq.Circuit
    type: str
    ideal_distribution: dict[str, float]

    def most_likely_bitstring(self) -> str:
        distribution = self.ideal_distribution
        return max(distribution, key=distribution.__getitem__)

    def largest_probability(self) -> float:
        return max(self.ideal_distribution.values())

    def converted_circuit(
        self, circuit_type: SUPPORTED_PROGRAM_TYPES
    ) -> QPROGRAM:
        """Adds measurements to all qubits and convert
        to the input frontend type.

        Args:
            circuit_type: The circuit type as a string.
                For supported circuit types see mitiq.SUPPORTED_PROGRAM_TYPES.
        Returns:
            The converted circuit with final measurements.
        """
        circuit = self.circuit.copy()
        circuit.append(cirq.measure(circuit.all_qubits()))
        return convert_from_mitiq(circuit, circuit_type.name)

    @property
    def num_qubits(self) -> int:
        return len(self.circuit.all_qubits())

    @property
    def circuit_depth(self) -> int:
        return len(self.circuit)

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(len(op.qubits) > 1 for op in self.circuit.all_operations())

    def to_dict(self) -> dict[str, Any]:
        """Produces a summary of the ``BenchmarkProblem``, to be used in
        recording the results when running calibration experiments.

        Returns:
            Dictionary summarizing important attributes of the problem's
            circuit.
        """
        base = asdict(self)
        # remove circuit; it can be regenerated if needed
        del base["circuit"]
        del base["id"]
        base["num_qubits"] = self.num_qubits
        base["circuit_depth"] = self.circuit_depth
        base["two_qubit_gate_count"] = self.two_qubit_gate_count
        return base

    def __repr__(self) -> str:
        return str(self.to_dict())

    def __str__(self) -> str:
        result = ""
        for key, value in self.to_dict().items():
            if key == "ideal_distribution":
                continue
            title: str = key.replace("_", " ").capitalize()
            result += f"{title}: {value}\n"
        return result.rstrip()


@dataclass
class Strategy:
    """A dataclass which describes precisely an error mitigation approach by
    specifying a technique and the associated options.

    Args:
        id: A unique numerical id.
        technique: One of Mitiq's support error mitigation strategies,
            specified as a :class:`MitigationTechnique`.
        technique_params: A dictionary of options to pass to the mitigation
            method specified in `technique`.
    """

    id: int
    technique: MitigationTechnique
    technique_params: dict[str, Any]

    @property
    def mitigation_function(self) -> Callable[..., float]:
        if self.technique is MitigationTechnique.PEC:
            self.technique_params.setdefault("noise_bias", 0)

            def partial_pec(circuit: cirq.Circuit, execute: Executor) -> float:
                rep_function = self.technique_params["representation_function"]
                operations = []
                for op in circuit.all_operations():
                    if len(op.qubits) >= 2 and op not in operations:
                        operations.append(cirq.Circuit(op))

                num_samples = self.technique_params["num_samples"]
                if (
                    self.technique_params["representation_function"]
                    == represent_operation_with_local_biased_noise
                ):
                    reps = [
                        rep_function(
                            op,
                            self.technique_params["noise_level"],
                            self.technique_params["noise_bias"],
                        )
                        for op in operations
                    ]
                else:
                    reps = [
                        rep_function(
                            op,
                            self.technique_params["noise_level"],
                        )
                        for op in operations
                    ]
                return self.technique.mitigation_function(
                    circuit,
                    execute,
                    representations=reps,
                    num_samples=num_samples,
                )

            return partial_pec
        elif self.technique is MitigationTechnique.ZNE:
            return partial(
                self.technique.mitigation_function, **self.technique_params
            )
        else:
            raise ValueError(
                """Specified technique is not supported by calibration.
                    See {} for supported techniques.""",
                calibration_supported_techniques,
            )

    def to_dict(self) -> dict[str, Any]:
        """A summary of the strategies parameters, without the technique added.

        Returns:
            A dictionary describing the strategies parameters."""
        summary = {"technique": self.technique.name}
        if self.technique is MitigationTechnique.ZNE:
            inference_func = self.technique_params["factory"]
            summary["factory"] = inference_func.__class__.__name__
            summary["scale_factors"] = inference_func._scale_factors
            summary["scale_method"] = self.technique_params[
                "scale_noise"
            ].__name__

        elif self.technique is MitigationTechnique.PEC:
            summary["representation_function"] = self.technique_params[
                "representation_function"
            ].__name__
            summary["noise_level"] = self.technique_params["noise_level"]
            summary["noise_bias"] = self.technique_params.setdefault(
                "noise_bias", 0
            )
            summary["is_qubit_dependent"] = self.technique_params[
                "is_qubit_dependent"
            ]
            summary["num_samples"] = self.technique_params["num_samples"]
        return summary

    def to_pretty_dict(self) -> dict[str, str]:
        summary = self.to_dict()
        if self.technique is MitigationTechnique.ZNE:
            summary["scale_factors"] = str(summary["scale_factors"])[1:-1]
            summary["factory"] = summary["factory"][:-7]
        elif self.technique is MitigationTechnique.PEC:
            summary["noise_bias"] = summary.get("noise_bias", "N/A")
            summary["representation_function"] = summary[
                "representation_function"
            ][25:]
        return summary

    def __repr__(self) -> str:
        return str(self.to_dict())

    def __str__(self) -> str:
        result = ""
        for key, value in self.to_pretty_dict().items():
            title: str = key.replace("_", " ").capitalize()
            result += f"{title}: {value}\n"
        return result.rstrip()

    def num_circuits_required(self) -> int | None:
        summary = self.to_dict()
        if self.technique is MitigationTechnique.ZNE:
            return len(summary["scale_factors"])
        elif self.technique is MitigationTechnique.PEC:
            return summary["num_samples"]
        elif self.technique is MitigationTechnique.RAW:
            return 1
        return None


class Settings:
    """A class to store the configuration settings of a :class:`.Calibrator`.

    Args:
        benchmarks: A list where each element is a dictionary of parameters for
            generating circuits to be used in calibration experiments. The
            dictionary keys include ``circuit_type``, ``num_qubits``,
            ``circuit_depth``, and in the case of mirror circuits, a random
            seed ``circuit_seed``. An example of input to ``benchmarks`` is::

                [
                    {
                        "circuit_type": "rb",
                        "num_qubits": 2,
                        "circuit_depth": 7,
                    },
                    {
                        "circuit_type": "mirror",
                        "num_qubits": 2,
                        "circuit_depth": 7,
                        "circuit_seed": 1,
                    }
                ]

        strategies: A specification of the methods/parameters to be used in
            calibration experiments.
    """

    def __init__(
        self,
        benchmarks: list[dict[str, Any]],
        strategies: list[dict[str, Any]],
    ):
        self.techniques = [
            MitigationTechnique[technique["technique"].upper()]
            for technique in strategies
        ]
        self.technique_params = strategies
        self.benchmarks = benchmarks
        self.strategy_dict: dict[int, Strategy] = {}
        self.problem_dict: dict[int, BenchmarkProblem] = {}

    def get_strategy(self, strategy_id: int) -> Strategy:
        return self.strategy_dict[strategy_id]

    def get_problem(self, problem_id: int) -> BenchmarkProblem:
        return self.problem_dict[problem_id]

    def make_problems(self) -> list[BenchmarkProblem]:
        """Generate the benchmark problems for the calibration experiment.
        Returns:
            A list of :class:`BenchmarkProblem` objects"""
        circuits = []
        for i, benchmark in enumerate(self.benchmarks):
            circuit_type = benchmark["circuit_type"]
            circuit: Any

            if circuit_type == "custom":
                user_circuit = benchmark["circuit"]
                circuit = convert_to_mitiq(user_circuit)[0]
                num_qubits = len(circuit.all_qubits())
                depth = len(circuit)
                ideal = benchmark.get("ideal_distribution", {})
            else:
                num_qubits = benchmark["num_qubits"]
                depth = benchmark.get("circuit_depth", -1)

            if circuit_type == "ghz":
                circuit = generate_ghz_circuit(num_qubits)
                ideal = {"0" * num_qubits: 0.5, "1" * num_qubits: 0.5}
            elif circuit_type == "w":
                circuit = generate_w_circuit(num_qubits)
                ideal = {}
                for i in range(num_qubits):
                    bitstring = "0" * i + "1" + "0" * (num_qubits - i - 1)
                    ideal[bitstring] = 1 / num_qubits
            elif circuit_type == "rb":
                circuit = generate_rb_circuits(num_qubits, depth)[0]
                ideal = {"0" * num_qubits: 1.0}
            elif circuit_type == "rotated_rb":
                theta = benchmark["theta"]
                if num_qubits == 1:
                    circuit = generate_rotated_rb_circuits(num_qubits, depth)[
                        0
                    ]
                    p = (2 / 3) * np.sin(theta / 2) ** 2
                    ideal = {"0": p, "1": 1 - p}
                else:
                    raise NotImplementedError(
                        """rotated rb circuits with >1 qubits
                        not yet supported in calibration"""
                    )

            elif circuit_type == "mirror":
                seed = benchmark.get("circuit_seed", None)
                circuit, bitstring_list = generate_mirror_circuit(
                    nlayers=depth,
                    two_qubit_gate_prob=1.0,
                    connectivity_graph=nx.complete_graph(num_qubits),
                    seed=seed,
                )
                ideal_bitstring = "".join(map(str, bitstring_list))
                ideal = {ideal_bitstring: 1.0}
            elif circuit_type == "qv":
                raise NotImplementedError(
                    "quantum volume circuits not yet supported in calibration"
                )
            elif circuit_type == "custom":
                # ideal distribution already set above (may be empty)
                pass
            elif circuit_type == "custom":
                # ideal distribution already set above (may be empty)
                pass
            else:
                raise ValueError(
                    "invalid value passed for `circuit_types`. Must be "
                    "one of `ghz`, `rb`, `mirror`, `w`, `qv`, or `custom`, "
                    "one of `ghz`, `rb`, `mirror`, `w`, `qv`, or `custom`, "
                    f"but got {circuit_type}."
                )

            circuit = cast(cirq.Circuit, circuit)
            problem = BenchmarkProblem(
                id=i,
                circuit=circuit,
                type=circuit_type,
                ideal_distribution=ideal,
            )
            circuits.append(problem)
            self.problem_dict[problem.id] = problem

        return circuits

    def make_strategies(self) -> list[Strategy]:
        """Generates a list of :class:`Strategy` objects using the specified
        configurations.

        Returns:
            A list of :class:`Strategy` objects."""
        funcs = []
        for i, (technique, params) in enumerate(
            zip(self.techniques, self.technique_params)
        ):
            params_copy = params.copy()
            del params_copy["technique"]

            strategy = Strategy(
                id=i, technique=technique, technique_params=params_copy
            )
            funcs.append(strategy)
            self.strategy_dict[strategy.id] = strategy
        return funcs


ZNE_SETTINGS = Settings(
    benchmarks=[
        {
            "circuit_type": "ghz",
            "num_qubits": 2,
        },
        {
            "circuit_type": "w",
            "num_qubits": 2,
        },
        {
            "circuit_type": "rb",
            "num_qubits": 2,
            "circuit_depth": 7,
        },
        {
            "circuit_type": "mirror",
            "num_qubits": 2,
            "circuit_depth": 7,
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
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": RichardsonFactory([1.0, 3.0, 5.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 2.0, 3.0]),
        },
        {
            "technique": "zne",
            "scale_noise": fold_gates_at_random,
            "factory": LinearFactory([1.0, 3.0, 5.0]),
        },
    ],
)

PEC_SETTINGS = Settings(
    benchmarks=[
        {
            "circuit_type": "ghz",
            "num_qubits": 2,
        },
        {
            "circuit_type": "w",
            "num_qubits": 2,
        },
        {
            "circuit_type": "rb",
            "num_qubits": 2,
            "circuit_depth": 7,
        },
        {
            "circuit_type": "mirror",
            "num_qubits": 2,
            "circuit_depth": 7,
            "circuit_seed": 1,
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
            "force_run_all": False,
        },
        {
            "technique": "pec",
            "representation_function": (
                represent_operation_with_local_depolarizing_noise
            ),
            "is_qubit_dependent": False,
            "noise_level": 0.01,
            "num_samples": 200,
            "force_run_all": False,
        },
    ],
)

DefaultStrategy = Strategy(0, MitigationTechnique.RAW, {})
