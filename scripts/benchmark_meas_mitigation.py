# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark measurement mitigation with Mitiq REM, Mitiq TREX, and M3.

Example:
    python scripts/benchmark_meas_mitigation.py --circuit ghz --n-qubits 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from benchmark_error_mitigation_common import (
    BenchmarkResult,
    CountingMeasurementExecutor,
    add_benchmark_arguments,
    benchmark_observable,
    build_benchmark_circuit,
    counts_from_measurements,
    ideal_expectation,
    improvement_factor,
    noisy_readout_executor,
    print_results,
    timed_result,
    zz_expectation_from_measurements,
)

from mitiq.experimental.trex import execute_with_trex
from mitiq.rem import execute_with_rem
from mitiq.rem.inverse_confusion_matrix import (
    generate_inverse_confusion_matrix,
)


def run_mthree(
    circuit,
    ideal: float,
    noisy: float,
    p0: float,
    p1: float,
    shots: int,
    seed: int,
) -> BenchmarkResult:
    """Runs matrix-free measurement mitigation with mthree."""

    try:
        import mthree
    except ImportError:
        return BenchmarkResult(
            method="mthree",
            ideal=ideal,
            noisy=noisy,
            mitigated=None,
            improvement=None,
            runtime=0.0,
            executions=0,
            notes="install mthree to enable",
        )

    executor = CountingMeasurementExecutor(
        noisy_readout_executor(p0, p1, shots, seed)
    )
    started_at = time.perf_counter()
    measurements = executor(circuit)
    counts = counts_from_measurements(measurements)

    mitigation = mthree.M3Mitigation()
    calibration = [
        np.array([[1 - p0, p1], [p0, 1 - p1]])
        for _ in range(len(circuit.all_qubits()))
    ]
    mitigation.cals_from_matrices(calibration)
    quasi = mitigation.apply_correction(
        counts, list(range(len(circuit.all_qubits())))
    )
    mitigated = float(quasi.expval("Z" * len(circuit.all_qubits())))
    runtime = time.perf_counter() - started_at

    return BenchmarkResult(
        method="mthree",
        ideal=ideal,
        noisy=noisy,
        mitigated=mitigated,
        improvement=improvement_factor(ideal, noisy, mitigated),
        runtime=runtime,
        executions=executor.calls,
        notes="M3 correction on Z...Z",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser)
    args = parser.parse_args()

    circuit = build_benchmark_circuit(
        args.circuit, args.n_qubits, args.depth, args.seed
    )
    observable = benchmark_observable(circuit)
    ideal = ideal_expectation(circuit)

    p0 = args.noise_level
    p1 = args.noise_level
    base_executor = CountingMeasurementExecutor(
        noisy_readout_executor(p0, p1, args.shots, args.seed)
    )
    noisy = zz_expectation_from_measurements(base_executor(circuit))

    rem_executor = CountingMeasurementExecutor(
        noisy_readout_executor(p0, p1, args.shots, args.seed)
    )
    inverse_confusion_matrix = generate_inverse_confusion_matrix(
        args.n_qubits, p0=p0, p1=p1
    )
    results = [
        timed_result(
            "mitiq.rem",
            ideal,
            noisy,
            rem_executor,
            lambda: execute_with_rem(
                circuit,
                rem_executor,
                observable,
                inverse_confusion_matrix=inverse_confusion_matrix,
            ),
        )
    ]

    trex_executor = CountingMeasurementExecutor(
        noisy_readout_executor(p0, p1, args.shots, args.seed)
    )
    results.append(
        timed_result(
            "mitiq.trex",
            ideal,
            noisy,
            trex_executor,
            lambda: execute_with_trex(
                circuit,
                trex_executor,
                observable,
                num_randomizations=32,
                random_state=args.seed,
            ),
            notes="experimental",
        )
    )
    results.append(
        run_mthree(circuit, ideal, noisy, p0, p1, args.shots, args.seed)
    )

    print_results(results)


if __name__ == "__main__":
    main()
