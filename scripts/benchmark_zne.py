# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark zero-noise extrapolation on Mitiq benchmark circuits.

Example:
    python scripts/benchmark_zne.py --circuit ghz --n-qubits 4
"""

from __future__ import annotations

import argparse
import time

import cirq
import numpy as np
from benchmark_error_mitigation_common import (
    BenchmarkResult,
    CountingFloatExecutor,
    add_benchmark_arguments,
    build_benchmark_circuit,
    ideal_expectation,
    improvement_factor,
    noisy_density_executor,
    print_results,
    timed_result,
    zz_expectation_from_density_matrix,
)

from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global


def qiskit_aer_density_executor(
    noise_level: float,
    seed: int,
) -> CountingFloatExecutor | None:
    """Builds a Qiskit Aer density-matrix executor when dependencies exist."""

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        from mitiq.interface import convert_from_mitiq
    except ImportError:
        return None

    def execute(circuit: cirq.Circuit) -> float:
        qiskit_circuit = convert_from_mitiq(circuit, "qiskit")

        one_qubit_gate_names = set()
        two_qubit_gate_names = set()
        for instruction in qiskit_circuit.data:
            num_qubits = instruction.operation.num_qubits
            if num_qubits == 1:
                one_qubit_gate_names.add(instruction.operation.name)
            elif num_qubits == 2:
                two_qubit_gate_names.add(instruction.operation.name)

        qiskit_circuit.save_density_matrix()
        noise_model = NoiseModel()
        if one_qubit_gate_names:
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(noise_level, 1),
                sorted(one_qubit_gate_names),
            )
        if two_qubit_gate_names:
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(noise_level, 2),
                sorted(two_qubit_gate_names),
            )

        simulator = AerSimulator(
            method="density_matrix",
            noise_model=noise_model,
            seed_simulator=seed,
        )
        result = simulator.run(qiskit_circuit).result()
        density_matrix = np.asarray(result.data(0)["density_matrix"])
        return zz_expectation_from_density_matrix(
            density_matrix, qiskit_circuit.num_qubits
        )

    return CountingFloatExecutor(execute)


def run_qiskit_aer_zne(
    circuit: cirq.Circuit,
    ideal: float,
    noisy: float,
    noise_level: float,
    seed: int,
) -> BenchmarkResult:
    """Runs a Qiskit Aer ZNE comparison over folded Mitiq circuits."""

    executor = qiskit_aer_density_executor(noise_level, seed)
    if executor is None:
        return BenchmarkResult(
            method="qiskit-aer.zne",
            ideal=ideal,
            noisy=noisy,
            mitigated=None,
            improvement=None,
            runtime=0.0,
            executions=0,
            notes="install qiskit extra to enable",
        )
    scale_factors = [1.0, 3.0, 5.0]
    started_at = time.perf_counter()
    values = [
        executor(fold_global(circuit, scale_factor))
        for scale_factor in scale_factors
    ]
    qiskit_noisy = values[0]
    coefficients = np.polyfit(scale_factors, values, deg=1)
    mitigated = float(np.polyval(coefficients, 0.0))
    runtime = time.perf_counter() - started_at
    return BenchmarkResult(
        method="qiskit-aer.zne",
        ideal=ideal,
        noisy=qiskit_noisy,
        mitigated=mitigated,
        improvement=improvement_factor(ideal, qiskit_noisy, mitigated),
        runtime=runtime,
        executions=executor.calls,
        notes="fold_global + linear fit",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser)
    args = parser.parse_args()

    circuit = build_benchmark_circuit(
        args.circuit, args.n_qubits, args.depth, args.seed
    )
    ideal = ideal_expectation(circuit)

    base_executor = CountingFloatExecutor(
        noisy_density_executor(args.noise_level)
    )
    noisy = base_executor(circuit)

    mitiq_executor = CountingFloatExecutor(
        noisy_density_executor(args.noise_level)
    )
    results = [
        timed_result(
            "mitiq.zne",
            ideal,
            noisy,
            mitiq_executor,
            lambda: execute_with_zne(
                circuit,
                mitiq_executor,
                factory=LinearFactory([1.0, 3.0, 5.0]),
                scale_noise=fold_global,
            ),
            notes="fold_global + LinearFactory",
        )
    ]
    results.append(
        run_qiskit_aer_zne(circuit, ideal, noisy, args.noise_level, args.seed)
    )

    print_results(results)


if __name__ == "__main__":
    main()
