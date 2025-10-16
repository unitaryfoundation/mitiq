# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections.abc import Callable, Sequence
from enum import Enum
from operator import itemgetter
from typing import Any, cast

import cirq
import numpy as np
import numpy.typing as npt
from tabulate import tabulate

from mitiq import (
    QPROGRAM,
    Executor,
    MeasurementResult,
    Observable,
    QuantumResult,
)
from mitiq.calibration.settings import (
    ZNE_SETTINGS,
    BenchmarkProblem,
    MitigationTechnique,
    Settings,
    Strategy,
    build_settings_from_pipelines,
)
from mitiq.interface import convert_from_mitiq
from mitiq.rem import mitigate_executor as rem_mitigate_executor
from mitiq.zne import mitigate_executor as zne_mitigate_executor


class MissingResultsError(Exception):
    pass


class OutputForm(str, Enum):
    flat = "flat"
    cartesian = "cartesian"


class ExperimentResults:
    """Class to store calibration experiment data, and provide helper methods
    for computing results based on it."""

    def __init__(
        self, strategies: list[Strategy], problems: list[BenchmarkProblem]
    ) -> None:
        self.strategies = strategies
        self.problems = problems
        self.num_strategies = len(strategies)
        self.num_problems = len(problems)
        self.reset_data()

    def add_result(
        self,
        strategy: Strategy,
        problem: BenchmarkProblem,
        *,
        ideal_val: float,
        noisy_val: float,
        mitigated_val: float,
    ) -> None:
        """Add a single result from a (Strategy, BenchmarkProblem) pair and
        store the results."""
        self.mitigated[strategy.id, problem.id] = mitigated_val
        self.noisy[strategy.id, problem.id] = noisy_val
        self.ideal[strategy.id, problem.id] = ideal_val

    @staticmethod
    def _performance_str(noisy_error: float, mitigated_error: float) -> str:
        """Get human readable performance representation."""
        return (
            f"{'✔' if mitigated_error < noisy_error else '✘'}\n"
            f"Noisy error: {round(noisy_error, 4)}\n"
            f"Mitigated error: {round(mitigated_error, 4)}\n"
            f"Improvement factor: {round(noisy_error / mitigated_error, 4)}"
        )

    def _get_errors(
        self, strategy_id: int, problem_id: int
    ) -> tuple[float, float]:
        """Get errors for a given strategy/problem combination.

        Returns:
            A tuple comprising:
            - absolute value of the noisy error
            - absolute value of the mitigated error
        """
        mitigated = self.mitigated[strategy_id, problem_id]
        noisy = self.noisy[strategy_id, problem_id]
        ideal = self.ideal[strategy_id, problem_id]
        mitigated_error = abs(ideal - mitigated)
        noisy_error = abs(ideal - noisy)
        return noisy_error, mitigated_error

    def log_results_flat(self) -> None:
        """
        Prints the results of the calibrator in flat form::

            ┌──────────────────────────┬──────────────────────────────┬────────────────────────────┐
            │ benchmark                │ strategy                     │ performance                │
            ├──────────────────────────┼──────────────────────────────┼────────────────────────────┤
            │ Type: rb                 │ Technique: ZNE               │ ✔                          │
            │ Num qubits: 2            │ Factory: Richardson          │ Noisy error: 0.101         │
            │ Circuit depth: 323       │ Scale factors: 1.0, 3.0, 5.0 │ Mitigated error: 0.0294    │
            │ Two qubit gate count: 77 │ Scale method: fold_global    │ Improvement factor: 3.4398 │
            ├──────────────────────────┼──────────────────────────────┼────────────────────────────┤
            │ Type: rb                 │ Technique: ZNE               │ ✔                          │
            │ Num qubits: 2            │ Factory: Richardson          │ Noisy error: 0.101         │
            │ Circuit depth: 323       │ Scale factors: 1.0, 2.0, 3.0 │ Mitigated error: 0.0501    │
            │ Two qubit gate count: 77 │ Scale method: fold_global    │ Improvement factor: 2.016  │
            ├──────────────────────────┼──────────────────────────────┼────────────────────────────┤
            │ Type: ghz                │ Technique: ZNE               │ ✔                          │
            │ Num qubits: 2            │ Factory: Richardson          │ Noisy error: 0.0128        │
            │ Circuit depth: 2         │ Scale factors: 1.0, 2.0, 3.0 │ Mitigated error: 0.0082    │
            │ Two qubit gate count: 1  │ Scale method: fold_global    │ Improvement factor: 1.561  │
            ├──────────────────────────┼──────────────────────────────┼────────────────────────────┤
            │ Type: ghz                │ Technique: ZNE               │ ✘                          │
            │ Num qubits: 2            │ Factory: Richardson          │ Noisy error: 0.0128        │
            │ Circuit depth: 2         │ Scale factors: 1.0, 3.0, 5.0 │ Mitigated error: 0.0137    │
            │ Two qubit gate count: 1  │ Scale method: fold_global    │ Improvement factor: 0.9369 │
            └──────────────────────────┴──────────────────────────────┴────────────────────────────┘
        """  # noqa: E501
        table: list[list[str | float]] = []
        headers: list[str] = ["benchmark", "strategy", "performance"]
        for problem in self.problems:
            row_group: list[list[str | float]] = []
            for strategy in self.strategies:
                nerr, merr = self._get_errors(strategy.id, problem.id)
                row_group.append(
                    [
                        str(problem),
                        str(strategy),
                        self._performance_str(nerr, merr),
                        # this is only for sorting
                        # removed after sorting
                        merr - nerr,
                    ]
                )
            row_group.sort(key=itemgetter(-1))
            table.extend([r[:-1] for r in row_group])
        return print(tabulate(table, headers, tablefmt="simple_grid"))

    def log_results_cartesian(self) -> None:
        """
        Prints the results of the calibrator in cartesian form::

            ┌──────────────────────────────┬────────────────────────────┬────────────────────────────┐
            │ strategy\\benchmark           │ Type: rb                   │ Type: ghz                  │
            │                              │ Num qubits: 2              │ Num qubits: 2              │
            │                              │ Circuit depth: 337         │ Circuit depth: 2           │
            │                              │ Two qubit gate count: 80   │ Two qubit gate count: 1    │
            ├──────────────────────────────┼────────────────────────────┼────────────────────────────┤
            │ Technique: ZNE               │ ✔                          │ ✘                          │
            │ Factory: Richardson          │ Noisy error: 0.1128        │ Noisy error: 0.0117        │
            │ Scale factors: 1.0, 2.0, 3.0 │ Mitigated error: 0.0501    │ Mitigated error: 0.0439    │
            │ Scale method: fold_global    │ Improvement factor: 2.2515 │ Improvement factor: 0.2665 │
            ├──────────────────────────────┼────────────────────────────┼────────────────────────────┤
            │ Technique: ZNE               │ ✔                          │ ✘                          │
            │ Factory: Richardson          │ Noisy error: 0.1128        │ Noisy error: 0.0117        │
            │ Scale factors: 1.0, 3.0, 5.0 │ Mitigated error: 0.0408    │ Mitigated error: 0.0171    │
            │ Scale method: fold_global    │ Improvement factor: 2.7672 │ Improvement factor: 0.6852 │
            └──────────────────────────────┴────────────────────────────┴────────────────────────────┘
        """  # noqa: E501
        table: list[list[str]] = []
        headers: list[str] = ["strategy\\benchmark"]
        for problem in self.problems:
            headers.append(str(problem))
        for strategy in self.strategies:
            row: list[str] = [str(strategy)]
            for problem in self.problems:
                nerr, merr = self._get_errors(strategy.id, problem.id)
                row.append(self._performance_str(nerr, merr))
            table.append(row)
        return print(tabulate(table, headers, tablefmt="simple_grid"))

    def is_missing_data(self) -> bool:
        """Method to check if there is any missing data that was expected from
        the calibration experiments."""
        return np.isnan(self.mitigated + self.noisy + self.ideal).any()

    def ensure_full(self) -> None:
        """Check to ensure all expected data is collected. All mitigated, noisy
        and ideal values must be nonempty for this to pass and return True."""
        if self.is_missing_data():
            raise MissingResultsError(
                "There are missing results from the expected calibration "
                "experiments. Please try running the experiments again with "
                "the `run` function."
            )

    def squared_errors(self) -> npt.NDArray[np.float32]:
        """Returns an array of squared errors, one for each (strategy, problem)
        pair."""
        return (self.ideal - self.mitigated) ** 2

    def best_strategy_id(self) -> int:
        """Returns the strategy id that corresponds to the strategy that
        maintained the smallest error across all ``BenchmarkProblem``
        instances."""
        errors = self.squared_errors()
        strategy_errors = np.sum(errors, axis=1)
        strategy_id = int(np.argmin(strategy_errors))
        return strategy_id

    def reset_data(self) -> None:
        """Reset all experiment result data using NaN values."""
        self.mitigated = np.full(
            (self.num_strategies, self.num_problems), np.nan
        )
        self.noisy = np.full((self.num_strategies, self.num_problems), np.nan)
        self.ideal = np.full((self.num_strategies, self.num_problems), np.nan)


class Calibrator:
    """An object used to orchestrate experiments for calibrating optimal error
    mitigation strategies.

    Args:
        executor: An unmitigated executor returning a
            :class:`.MeasurementResult`.
        settings: Optional ``Settings`` specifying the benchmark problems and
            mitigation strategies to explore.
        pipelines: Optional list of pipeline strings (e.g., ``"rem | zne"``)
            used to automatically construct calibration settings.
        frontend: The executor frontend as a string. For a list of supported
            frontends see ``mitiq.SUPPORTED_PROGRAM_TYPES.keys()``,
        ideal_executor: An optional simulated executor returning the ideal
            :class:`.MeasurementResult` without noise.
    """

    def __init__(
        self,
        executor: Executor | Callable[[QPROGRAM], QuantumResult],
        *,
        frontend: str,
        settings: Settings | None = None,
        pipelines: Sequence[str] | None = None,
        ideal_executor: Executor
        | Callable[[QPROGRAM], QuantumResult]
        | None = None,
    ):
        if settings is not None and pipelines is not None:
            raise ValueError(
                "Provide either an explicit Settings instance or a list of "
                "pipelines, but not both."
            )
        if pipelines is not None:
            settings = build_settings_from_pipelines(pipelines)

        self.executor = (
            executor if isinstance(executor, Executor) else Executor(executor)
        )
        self.ideal_executor = (
            Executor(ideal_executor)
            if ideal_executor and not isinstance(ideal_executor, Executor)
            else ideal_executor
        )
        self.settings = settings if settings is not None else ZNE_SETTINGS
        self.problems = self.settings.make_problems()
        self.strategies = self.settings.make_strategies()
        self.results = ExperimentResults(
            strategies=self.strategies, problems=self.problems
        )

        # Build an executor of Cirq circuits
        def cirq_execute(
            circuits: Sequence[cirq.Circuit],
        ) -> Sequence[MeasurementResult]:
            q_programs = [convert_from_mitiq(c, frontend) for c in circuits]
            results = cast(
                Sequence[MeasurementResult], self.executor.run(q_programs)
            )
            return results

        self._cirq_executor = Executor(cirq_execute)  # type: ignore [arg-type]

        # Executor for ideal circuits if provided
        if self.ideal_executor is not None:
            ideal_exec: Executor = self.ideal_executor

            def ideal_cirq_execute(
                circuits: Sequence[cirq.Circuit],
            ) -> Sequence[MeasurementResult]:
                q_programs = [
                    convert_from_mitiq(c, frontend) for c in circuits
                ]
                results = cast(
                    Sequence[MeasurementResult],
                    ideal_exec.run(q_programs),
                )
                return results

            self._ideal_cirq_executor: Executor | None = Executor(
                ideal_cirq_execute  # type: ignore[arg-type]
            )
        else:
            self._ideal_cirq_executor = None

        # Populate ideal distributions for custom circuits if needed
        if self._ideal_cirq_executor is not None:
            for problem in self.problems:
                if problem.type == "custom" and not problem.ideal_distribution:
                    circ = problem.circuit.copy()
                    circ.append(cirq.measure(sorted(circ.all_qubits())))
                    result = cast(
                        MeasurementResult,
                        self._ideal_cirq_executor.run([circ])[0],
                    )
                    problem.ideal_distribution = result.prob_distribution()

    @property
    def cirq_executor(self) -> Executor:
        """Returns an executor which is able to run Cirq circuits
        by converting them and calling self.executor.

        Args:
            executor: Executor which takes as input QPROGRAM circuits.

        Returns:
            Executor which takes as input a Cirq circuits.
        """
        return self._cirq_executor

    @property
    def ideal_cirq_executor(self) -> Executor | None:
        """Executor running ideal circuits if provided."""
        return self._ideal_cirq_executor

    def get_cost(self) -> dict[str, int]:
        """Returns the expected number of noisy and ideal expectation values
        required for calibration.

        Returns:
            A summary of the number of circuits to be run.
        """
        num_circuits = len(self.problems)
        num_options = sum(
            strategy.num_circuits_required()  # type: ignore
            for strategy in self.strategies  # type: ignore
        )

        noisy = num_circuits * (num_options + 1)
        ideal = 0  # TODO: ideal executor is currently unused
        return {
            "noisy_executions": noisy,
            "ideal_executions": ideal,
        }

    def run(self, log: OutputForm | None = None) -> None:
        """Runs all the circuits required for calibration.

        args:
             log: If set, detailed results of each experiment run by the
                calibrator are printed. The value corresponds to the format of
                the information and can be set to “flat” or “cartesian”.
        """
        if not self.results.is_missing_data():
            self.results.reset_data()

        for problem in self.problems:
            # Benchmark circuits have no measurements, so we append them.
            circuit = problem.circuit.copy()
            circuit.append(cirq.measure(sorted(circuit.all_qubits())))

            bitstring_to_measure = problem.most_likely_bitstring()
            expval_executor = convert_to_expval_executor(
                self.cirq_executor, bitstring_to_measure
            )
            noisy_value = expval_executor.evaluate(circuit)[0]
            for strategy in self.strategies:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if strategy.technique is MitigationTechnique.PIPELINE:
                        mitigated_value = self._evaluate_pipeline_strategy(
                            strategy,
                            circuit,
                            bitstring_to_measure,
                        )
                    else:
                        mitigated_value = strategy.mitigation_function(
                            circuit, expval_executor
                        )
                self.results.add_result(
                    strategy,
                    problem,
                    ideal_val=problem.largest_probability(),
                    noisy_val=noisy_value,
                    mitigated_val=mitigated_value,
                )

        self.results.ensure_full()

        if log is not None:
            if log == OutputForm.flat:
                self.results.log_results_flat()
            elif log == OutputForm.cartesian:
                self.results.log_results_cartesian()
            else:
                raise ValueError(
                    "log parameter must be one of: "
                    f"{', '.join(OutputForm._member_names_)}"
                )

    def best_strategy(self) -> Strategy:
        """Finds the best strategy by using the parameters that had the
        smallest error.

        Args:
            results: Calibration experiment results. Obtained by first running
                :func:`run`.

        Returns:
            A single :class:`Strategy` object specifying the technique and
            parameters that performed best.
        """
        self.results.ensure_full()

        strategy_id = self.results.best_strategy_id()
        return self.settings.get_strategy(strategy_id)

    def execute_with_mitigation(
        self,
        circuit: QPROGRAM,
        expval_executor: Executor | Callable[[QPROGRAM], QuantumResult],
        observable: Observable | None = None,
    ) -> QuantumResult | None:
        """See :func:`execute_with_mitigation` for signature and details."""
        return execute_with_mitigation(
            circuit, expval_executor, observable, calibrator=self
        )

    @staticmethod
    def _identity_inverse_confusion_matrix(
        num_qubits: int,
    ) -> npt.NDArray[np.float64]:
        dim = 2**num_qubits
        return np.eye(dim, dtype=np.float64)

    @staticmethod
    def _extract_expectation_from_result(
        result: QuantumResult, bitstring: str
    ) -> float:
        if isinstance(result, MeasurementResult) or hasattr(
            result, "prob_distribution"
        ):
            distribution = cast(MeasurementResult, result).prob_distribution()
            return float(distribution.get(bitstring, 0.0))

        if isinstance(result, (list, tuple)):
            if not result:
                raise ValueError("Executor returned an empty result list.")
            result = result[0]

        if isinstance(result, np.ndarray):
            if result.size == 0:
                raise ValueError("Executor returned an empty ndarray result.")
            result = result.flat[0]

        if isinstance(result, complex):
            return float(result.real)

        if isinstance(result, (np.floating, np.integer)):
            return float(result)

        if isinstance(result, float):
            return result

        raise TypeError(
            "Unsupported result type returned from mitigation pipeline: "
            f"{type(result)}"
        )

    def _measurement_executor_callable(
        self,
    ) -> Callable[[cirq.Circuit], MeasurementResult]:
        base_executor = self.cirq_executor

        def _execute(circ: cirq.Circuit) -> MeasurementResult:
            result = base_executor.run([circ])[0]
            return cast(MeasurementResult, result)

        return _execute

    @staticmethod
    def _factory_from_stage_params(params: dict[str, Any]) -> Any:
        factory = params.get("factory")
        if factory is not None:
            return factory.reset()
        factory_ctor = params.get("factory_ctor")
        if factory_ctor is None:
            raise ValueError(
                "Pipeline stage parameters must include either 'factory' or "
                "'factory_ctor'."
            )
        args = params.get("factory_args", ())
        kwargs = params.get("factory_kwargs", {})
        return factory_ctor(*args, **kwargs)

    def _evaluate_pipeline_strategy(
        self,
        strategy: Strategy,
        circuit: cirq.Circuit,
        bitstring: str,
    ) -> float:
        stages = strategy.technique_params.get("stages", [])
        if not stages:
            raise ValueError(
                "Pipeline strategy is missing stage definitions. "
                f"Strategy: {strategy}"
            )

        current_executor: Callable[[cirq.Circuit], Any] = (
            self._measurement_executor_callable()
        )
        current_type = "measurement"
        num_qubits = len(circuit.all_qubits())

        for stage in stages:
            name = stage["name"]
            params = stage.get("params", {})

            if name == "rem":
                icm = params.get("inverse_confusion_matrix")
                if isinstance(icm, str) and icm == "identity":
                    icm_array = self._identity_inverse_confusion_matrix(
                        num_qubits
                    )
                elif icm is None:
                    icm_array = self._identity_inverse_confusion_matrix(
                        num_qubits
                    )
                else:
                    icm_array = np.asarray(icm, dtype=np.float64)

                current_executor = cast(
                    Callable[[cirq.Circuit], MeasurementResult],
                    rem_mitigate_executor(
                        current_executor,
                        inverse_confusion_matrix=icm_array,
                    ),
                )
                current_type = "measurement"
            elif name == "zne":
                factory = self._factory_from_stage_params(params)
                scale_noise = params["scale_noise"]
                num_to_average = params.get("num_to_average", 1)

                if current_type == "measurement":
                    measurement_executor = current_executor

                    def expectation_executor(
                        circ: cirq.Circuit,
                    ) -> float:
                        measurement = measurement_executor(circ)
                        return (
                            cast(MeasurementResult, measurement)
                            .prob_distribution()
                            .get(bitstring, 0.0)
                        )

                    base_executor_for_zne: Callable[[cirq.Circuit], float] = (
                        expectation_executor
                    )
                else:
                    base_executor_for_zne = cast(
                        Callable[[cirq.Circuit], float], current_executor
                    )

                current_executor = zne_mitigate_executor(
                    base_executor_for_zne,
                    factory=factory,
                    scale_noise=scale_noise,
                    num_to_average=num_to_average,
                )
                current_type = "expectation"
            else:
                raise ValueError(
                    f"Unsupported pipeline stage '{name}' encountered."
                )

        final_result = current_executor(circuit)
        return self._extract_expectation_from_result(final_result, bitstring)


def convert_to_expval_executor(executor: Executor, bitstring: str) -> Executor:
    """Constructs a new executor returning an expectation value given by the
    probability that the circuit outputs the most likely state according to the
    ideal distribution.

    Args:
        executor: Executor which returns a :class:`.MeasurementResult`
            (bitstrings).
        bitstring: The bitstring to measure the probability of. Defaults to
            ground state bitstring "00...0".

    Returns:
        A tuple containing an executor returning expectation values and,
        the most likely bitstring, according to the passed ``distribution``
    """

    def expval_executor(circuit: cirq.Circuit) -> float:
        circuit_with_meas = circuit.copy()
        if not cirq.is_measurement(circuit_with_meas):
            circuit_with_meas.append(
                cirq.measure(sorted(circuit_with_meas.all_qubits()))
            )
        raw = cast(MeasurementResult, executor.run([circuit_with_meas])[0])
        distribution = raw.prob_distribution()
        return distribution.get(bitstring, 0.0)

    return Executor(expval_executor)  # type: ignore [arg-type]


def execute_with_mitigation(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    calibrator: Calibrator,
) -> QuantumResult | None:
    """Estimates the error-mitigated expectation value associated to the
    input circuit, via the application of the best mitigation strategy, as
    determined by calibration.

    Args:
        circuit: The input circuit to execute.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value.
            Otherwise, the ``QuantumResult`` returned by ``executor`` is used
            to compute the expectation of the observable.
        calibrator: ``Calibrator`` object with which to determine the error
            mitigation strategy to execute the circuit.

    Returns:
        The error mitigated expectation value.
    """

    if calibrator.results.is_missing_data():
        cost = calibrator.get_cost()
        answer = input(
            "Calibration experiments have not yet been run. You can run the "
            "experiments manually by calling `calibrator.run()`, or they can "
            f"be run now. The potential cost is:\n{cost}\n"
            "Would you like the experiments to be run automatically? (yes/no)"
        )
        if answer.lower() == "yes":
            calibrator.run()
        else:
            return None

    strategy = calibrator.best_strategy()
    if strategy.technique is MitigationTechnique.PIPELINE:
        raise NotImplementedError(
            "execute_with_mitigation does not yet support automatically "
            "running pipeline strategies on arbitrary circuits."
        )
    em_func = strategy.mitigation_function
    return em_func(circuit, executor=executor, observable=observable)
