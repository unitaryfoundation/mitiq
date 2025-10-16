# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Callable, Sequence
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
    PIPELINE = auto()

    @property
    def mitigation_function(self) -> Callable[..., float]:
        if self is MitigationTechnique.ZNE:
            return execute_with_zne
        elif self is MitigationTechnique.PEC:
            return cast(Callable[..., float], execute_with_pec)
        elif self is MitigationTechnique.RAW:
            return execute
        elif self is MitigationTechnique.PIPELINE:

            def _pipeline_not_supported(*_args: Any, **_kwargs: Any) -> float:
                raise NotImplementedError(
                    "Pipeline mitigation functions are orchestrated by the "
                    "Calibrator and cannot be invoked directly."
                )

            return _pipeline_not_supported
        else:
            raise ValueError(f"Unsupported mitigation technique: {self!r}.")


calibration_supported_techniques = {
    "ZNE": MitigationTechnique.ZNE,
    "PEC": MitigationTechnique.PEC,
    "PIPELINE": MitigationTechnique.PIPELINE,
}


DEFAULT_CALIBRATION_BENCHMARKS: list[dict[str, Any]] = [
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
]


def default_calibration_benchmarks() -> list[dict[str, Any]]:
    """Returns a deep copy of the default benchmark specifications."""

    return copy.deepcopy(DEFAULT_CALIBRATION_BENCHMARKS)


DEFAULT_ZNE_FACTORY_SPECS: list[dict[str, Any]] = [
    {
        "factory_ctor": RichardsonFactory,
        "factory_args": ([1.0, 2.0, 3.0],),
        "factory_kwargs": {},
        "scale_noise": fold_global,
    },
    {
        "factory_ctor": RichardsonFactory,
        "factory_args": ([1.0, 3.0, 5.0],),
        "factory_kwargs": {},
        "scale_noise": fold_global,
    },
    {
        "factory_ctor": LinearFactory,
        "factory_args": ([1.0, 2.0, 3.0],),
        "factory_kwargs": {},
        "scale_noise": fold_global,
    },
    {
        "factory_ctor": LinearFactory,
        "factory_args": ([1.0, 3.0, 5.0],),
        "factory_kwargs": {},
        "scale_noise": fold_global,
    },
    {
        "factory_ctor": RichardsonFactory,
        "factory_args": ([1.0, 2.0, 3.0],),
        "factory_kwargs": {},
        "scale_noise": fold_gates_at_random,
    },
    {
        "factory_ctor": RichardsonFactory,
        "factory_args": ([1.0, 3.0, 5.0],),
        "factory_kwargs": {},
        "scale_noise": fold_gates_at_random,
    },
    {
        "factory_ctor": LinearFactory,
        "factory_args": ([1.0, 2.0, 3.0],),
        "factory_kwargs": {},
        "scale_noise": fold_gates_at_random,
    },
    {
        "factory_ctor": LinearFactory,
        "factory_args": ([1.0, 3.0, 5.0],),
        "factory_kwargs": {},
        "scale_noise": fold_gates_at_random,
    },
]


def default_zne_strategy_dicts() -> list[dict[str, Any]]:
    """Generates the default ZNE strategy specifications."""

    strategies: list[dict[str, Any]] = []
    for spec in DEFAULT_ZNE_FACTORY_SPECS:
        factory = spec["factory_ctor"](
            *spec["factory_args"], **spec["factory_kwargs"]
        )
        strategies.append(
            {
                "technique": "zne",
                "scale_noise": spec["scale_noise"],
                "factory": factory,
            }
        )
    return strategies


def default_pec_strategy_dicts() -> list[dict[str, Any]]:
    """Generates the default PEC strategy specifications."""

    return [
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
    ]


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
        elif self.technique is MitigationTechnique.PIPELINE:
            stages = self.technique_params.get("stages", [])
            summary["pipeline"] = self.technique_params.get(
                "pipeline_label",
                " -> ".join(stage["name"].upper() for stage in stages),
            )
            summary["stages"] = " -> ".join(
                stage["name"].upper() for stage in stages
            )
            for stage in stages:
                if stage["name"] == "pt":
                    summary.setdefault(
                        "pt_num_circuits", stage["params"].get("num_circuits")
                    )
                    continue
                if stage["name"] == "zne":
                    factory = _factory_from_stage_params(stage["params"])
                    summary["factory"] = factory.__class__.__name__
                    summary["scale_factors"] = getattr(
                        factory, "_scale_factors", None
                    )
                    summary["scale_method"] = stage["params"][
                        "scale_noise"
                    ].__name__
                    summary["num_to_average"] = stage["params"].get(
                        "num_to_average", 1
                    )
                    break
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
        elif self.technique is MitigationTechnique.PIPELINE:
            if "scale_factors" in summary and summary["scale_factors"]:
                summary["scale_factors"] = str(summary["scale_factors"])[1:-1]
            if "factory" in summary and summary["factory"]:
                summary["factory"] = summary["factory"][:-7]
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
        elif self.technique is MitigationTechnique.PIPELINE:
            circuits = 1
            for stage in self.technique_params.get("stages", []):
                if stage["name"] == "pt":
                    circuits *= max(
                        1, int(stage["params"].get("num_circuits", 1))
                    )
                    continue
                if stage["name"] == "zne":
                    factory = _factory_from_stage_params(
                        stage["params"], fresh=True
                    )
                    scale_factors = getattr(factory, "_scale_factors", [])
                    num_to_average = stage["params"].get("num_to_average", 1)
                    circuits *= len(scale_factors) * num_to_average
            return circuits
        return None


PIPELINE_STAGE_SPECS: dict[str, dict[str, Any]] = {
    "rem": {
        "input": {"measurement"},
        "output": "measurement",
    },
    "zne": {
        "input": {"measurement", "expectation"},
        "output": "expectation",
    },
    "pt": {
        "input": {"measurement", "expectation"},
        "output": "same",
    },
}


def _normalize_pipeline_string(pipeline: str) -> tuple[str, ...]:
    tokens = [segment.strip().lower() for segment in pipeline.split("|")]
    normalized = tuple(token for token in tokens if token)
    if not normalized:
        raise ValueError(
            "Pipeline specification must contain at least one stage."
        )
    return normalized


def _validate_pipeline_tokens(tokens: tuple[str, ...], original: str) -> None:
    current_type = "measurement"
    for stage_name in tokens:
        if stage_name not in PIPELINE_STAGE_SPECS:
            supported = ", ".join(sorted(PIPELINE_STAGE_SPECS))
            raise ValueError(
                f"Unsupported pipeline stage '{stage_name}' in '{original}'. "
                f"Supported stages: {supported}."
            )
        spec = PIPELINE_STAGE_SPECS[stage_name]
        if current_type not in spec["input"]:
            expected = ", ".join(sorted(spec["input"]))
            raise ValueError(
                f"Pipeline '{original}' is invalid. Stage '{stage_name}' "
                f"expects input of type {expected}, received {current_type}."
            )
        output_type = spec["output"]
        if output_type != "same":
            current_type = output_type


def _stage_variants(stage_name: str) -> list[dict[str, Any]]:
    if stage_name == "rem":
        return [
            {
                "name": "rem",
                "params": {"inverse_confusion_matrix": "identity"},
            }
        ]
    if stage_name == "zne":
        variants: list[dict[str, Any]] = []
        for spec in DEFAULT_ZNE_FACTORY_SPECS:
            variants.append(
                {
                    "name": "zne",
                    "params": {
                        "factory_ctor": spec["factory_ctor"],
                        "factory_args": spec["factory_args"],
                        "factory_kwargs": spec["factory_kwargs"],
                        "scale_noise": spec["scale_noise"],
                        "num_to_average": spec.get("num_to_average", 1),
                    },
                }
            )
        return variants
    if stage_name == "pt":
        num_circuit_options = [4, 8]
        return [
            {
                "name": "pt",
                "params": {
                    "num_circuits": option,
                },
            }
            for option in num_circuit_options
        ]
    raise ValueError(f"Unsupported pipeline stage '{stage_name}'.")


def _build_pipeline_strategy_dicts(
    tokens: tuple[str, ...], pipeline_label: str
) -> list[dict[str, Any]]:
    stage_options: list[list[dict[str, Any]]] = [[]]
    for stage_name in tokens:
        variants = _stage_variants(stage_name)
        stage_options = [
            option + [copy.deepcopy(variant)]
            for option in stage_options
            for variant in variants
        ]

    normalized_label = " | ".join(stage.upper() for stage in tokens)
    strategies: list[dict[str, Any]] = []
    for stage_list in stage_options:
        stages = [
            {
                "name": stage["name"],
                "params": stage["params"],
            }
            for stage in stage_list
        ]
        strategies.append(
            {
                "technique": "pipeline",
                "pipeline_label": pipeline_label or normalized_label,
                "stages": stages,
            }
        )
    return strategies


def _factory_from_stage_params(params: dict[str, Any], *, fresh: bool = False):
    factory = params.get("factory")
    if factory is not None:
        return factory.reset() if fresh else factory

    factory_ctor = params.get("factory_ctor")
    if factory_ctor is None:
        raise ValueError(
            "Pipeline stage parameters must include either 'factory' or "
            "'factory_ctor'."
        )
    args = params.get("factory_args", ())
    kwargs = params.get("factory_kwargs", {})
    return factory_ctor(*args, **kwargs)


def build_settings_from_pipelines(pipelines: Sequence[str]) -> "Settings":
    """Constructs calibration settings from user-provided pipeline strings."""

    if not pipelines:
        raise ValueError("At least one pipeline must be provided.")

    strategy_dicts: list[dict[str, Any]] = []
    for pipeline in pipelines:
        tokens = _normalize_pipeline_string(pipeline)
        _validate_pipeline_tokens(tokens, pipeline)
        strategy_dicts.extend(_build_pipeline_strategy_dicts(tokens, pipeline))

    return Settings(
        default_calibration_benchmarks(),
        strategy_dicts,
    )


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
            else:
                raise ValueError(
                    "invalid value passed for `circuit_types`. Must be "
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
    benchmarks=default_calibration_benchmarks(),
    strategies=default_zne_strategy_dicts(),
)

PEC_SETTINGS = Settings(
    benchmarks=default_calibration_benchmarks(),
    strategies=default_pec_strategy_dicts(),
)

DefaultStrategy = Strategy(0, MitigationTechnique.RAW, {})
