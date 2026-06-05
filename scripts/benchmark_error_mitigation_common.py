# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for standalone error-mitigation benchmark scripts."""

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import cirq
import networkx as nx
import numpy as np
from tabulate import tabulate

from mitiq import MeasurementResult, Observable, PauliString
from mitiq.benchmarks import (
    generate_ghz_circuit,
    generate_mirror_circuit,
    generate_quantum_volume_circuit,
)
from mitiq.interface.mitiq_cirq import compute_density_matrix


@dataclass(frozen=True)
class BenchmarkResult:
    """Container for one benchmark row."""

    method: str
    ideal: float
    noisy: float
    mitigated: float | None
    improvement: float | None
    runtime: float
    executions: int
    notes: str = ""


class CountingFloatExecutor:
    """Wraps an expectation-value executor and counts calls."""

    def __init__(self, executor: Callable[[cirq.Circuit], float]) -> None:
        self._executor = executor
        self.calls = 0

    def __call__(self, circuit: cirq.Circuit) -> float:
        self.calls += 1
        return self._executor(circuit)


class CountingMeasurementExecutor:
    """Wraps a measurement executor and counts calls."""

    def __init__(
        self, executor: Callable[[cirq.Circuit], MeasurementResult]
    ) -> None:
        self._executor = executor
        self.calls = 0

    def __call__(self, circuit: cirq.Circuit) -> MeasurementResult:
        self.calls += 1
        return self._executor(circuit)


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds common benchmark CLI arguments."""

    parser.add_argument(
        "--circuit",
        choices=("ghz", "qv", "mirror"),
        default="ghz",
        help="Benchmark circuit family.",
    )
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)


def build_benchmark_circuit(
    circuit_name: str,
    n_qubits: int,
    depth: int,
    seed: int,
) -> cirq.Circuit:
    """Returns a Mitiq benchmark circuit as a Cirq circuit."""

    if circuit_name == "ghz":
        return strip_global_phase(
            generate_ghz_circuit(n_qubits, return_type="cirq")
        )

    if circuit_name == "qv":
        circuit, _ = generate_quantum_volume_circuit(
            n_qubits,
            depth,
            decompose=True,
            seed=seed,
            return_type="cirq",
        )
        return strip_global_phase(circuit)

    graph = nx.path_graph(n_qubits)
    circuit, _ = generate_mirror_circuit(
        depth,
        two_qubit_gate_prob=0.5,
        connectivity_graph=graph,
        seed=seed,
        return_type="cirq",
    )
    return strip_global_phase(circuit)


def strip_global_phase(circuit: cirq.Circuit) -> cirq.Circuit:
    """Removes global phases, which do not affect benchmark expectations."""

    return cirq.Circuit(
        op
        for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.GlobalPhaseGate)
    )


def benchmark_observable(circuit: cirq.Circuit) -> Observable:
    """Returns a simple Pauli-Z observable for the circuit qubits."""

    n_qubits = len(sorted(circuit.all_qubits()))
    return Observable(PauliString("Z" * n_qubits))


def zz_expectation_from_density_matrix(
    density_matrix: np.ndarray,
    n_qubits: int,
) -> float:
    """Computes the expectation of Z on all qubits."""

    expectation = 0.0
    probabilities = np.real(np.diag(density_matrix))
    for state, probability in enumerate(probabilities):
        eigenvalue = 1
        for qubit_index in range(n_qubits):
            bit = (state >> (n_qubits - qubit_index - 1)) & 1
            eigenvalue *= 1 if bit == 0 else -1
        expectation += eigenvalue * probability
    return float(expectation)


def zz_expectation_from_measurements(measurements: MeasurementResult) -> float:
    """Computes the expectation of Z on all measured qubits."""

    bitstrings = np.asarray(measurements.result)
    signs = 1 - 2 * bitstrings
    return float(np.mean(np.prod(signs, axis=1)))


def ideal_expectation(circuit: cirq.Circuit) -> float:
    """Computes the noiseless ZZ expectation value."""

    density_matrix = compute_density_matrix(circuit, noise_level=(0.0,))
    return zz_expectation_from_density_matrix(
        density_matrix, len(circuit.all_qubits())
    )


def improvement_factor(
    ideal: float,
    noisy: float,
    mitigated: float,
) -> float:
    """Computes |noisy - ideal| / |mitigated - ideal|."""

    mitigated_error = abs(mitigated - ideal)
    if mitigated_error == 0:
        return float("inf")
    return abs(noisy - ideal) / mitigated_error


def append_gate_noise(
    circuit: cirq.Circuit,
    noise_level: float,
) -> cirq.Circuit:
    """Returns a copy with depolarizing noise after non-measurement gates."""

    noisy_circuit = cirq.Circuit()
    for op in circuit.all_operations():
        noisy_circuit.append(op)
        if isinstance(op.gate, cirq.MeasurementGate):
            continue
        if len(op.qubits) == 1:
            noisy_circuit.append(cirq.depolarize(noise_level).on(*op.qubits))
        elif len(op.qubits) == 2:
            noisy_circuit.append(
                cirq.depolarize(noise_level, n_qubits=2).on(*op.qubits)
            )
    return noisy_circuit


def noisy_density_executor(
    noise_level: float,
) -> Callable[[cirq.Circuit], float]:
    """Builds an executor returning a noisy ZZ expectation value."""

    def execute(circuit: cirq.Circuit) -> float:
        noisy_circuit = append_gate_noise(circuit, noise_level)
        density_matrix = compute_density_matrix(
            noisy_circuit, noise_level=(0.0,)
        )
        return zz_expectation_from_density_matrix(
            density_matrix, len(circuit.all_qubits())
        )

    return execute


def add_terminal_measurements(circuit: cirq.Circuit) -> cirq.Circuit:
    """Returns a copy with one terminal measurement over all qubits."""

    measured = circuit.copy()
    qubits = sorted(measured.all_qubits())
    measured.append(cirq.measure(*qubits, key="m"))
    return measured


def noisy_readout_executor(
    p0: float,
    p1: float,
    shots: int,
    seed: int,
) -> Callable[[cirq.Circuit], MeasurementResult]:
    """Builds a readout-noise executor returning Mitiq measurements."""

    rng = np.random.default_rng(seed)
    simulator = cirq.Simulator(seed=seed)

    def execute(circuit: cirq.Circuit) -> MeasurementResult:
        measured = add_terminal_measurements(circuit)
        qubits = sorted(circuit.all_qubits())
        result = simulator.run(measured, repetitions=shots)
        bitstrings = np.array(result.measurements["m"], copy=True)

        zero_flips = (bitstrings == 0) & (rng.random(bitstrings.shape) < p0)
        one_flips = (bitstrings == 1) & (rng.random(bitstrings.shape) < p1)
        bitstrings[zero_flips] = 1
        bitstrings[one_flips] = 0

        return MeasurementResult(
            result=bitstrings,
            qubit_indices=tuple(int(q.x) for q in qubits),
        )

    return execute


def counts_from_measurements(
    measurements: MeasurementResult,
) -> dict[str, int]:
    """Converts Mitiq measurements into a counts dictionary."""

    counts: dict[str, int] = {}
    for row in measurements.result:
        key = "".join(str(int(bit)) for bit in row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-prints benchmark rows."""

    rows = []
    for result in results:
        mitigated = (
            "n/a" if result.mitigated is None else f"{result.mitigated:.6f}"
        )
        improvement = "n/a"
        if result.improvement is not None:
            improvement = f"{result.improvement:.3f}"
        rows.append(
            [
                result.method,
                f"{result.ideal:.6f}",
                f"{result.noisy:.6f}",
                mitigated,
                improvement,
                f"{result.runtime:.3f}",
                result.executions,
                result.notes,
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "method",
                "ideal",
                "noisy",
                "mitigated",
                "improvement",
                "runtime_s",
                "executions",
                "notes",
            ],
        )
    )


def timed_result(
    method: str,
    ideal: float,
    noisy: float,
    executor: CountingFloatExecutor | CountingMeasurementExecutor,
    run: Callable[[], float],
    notes: str = "",
) -> BenchmarkResult:
    """Runs a mitigation method and returns a benchmark row."""

    started_at = time.perf_counter()
    mitigated = float(np.real_if_close(run()))
    runtime = time.perf_counter() - started_at
    return BenchmarkResult(
        method=method,
        ideal=ideal,
        noisy=noisy,
        mitigated=mitigated,
        improvement=improvement_factor(ideal, noisy, mitigated),
        runtime=runtime,
        executions=executor.calls,
        notes=notes,
    )
