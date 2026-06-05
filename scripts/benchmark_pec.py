# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark probabilistic error cancellation on Mitiq benchmark circuits.

Example:
    python scripts/benchmark_pec.py --circuit ghz --n-qubits 2 --shots 200
"""

from __future__ import annotations

import argparse

import cirq
from benchmark_error_mitigation_common import (
    BenchmarkResult,
    CountingFloatExecutor,
    add_benchmark_arguments,
    build_benchmark_circuit,
    ideal_expectation,
    noisy_density_executor,
    print_results,
    timed_result,
)

from mitiq.pec import execute_with_pec
from mitiq.pec.representations import (
    represent_operations_in_circuit_with_local_depolarizing_noise,
)


def pec_representations(circuit: cirq.Circuit, noise_level: float):
    """Constructs local depolarizing-noise representations for the circuit."""

    return represent_operations_in_circuit_with_local_depolarizing_noise(
        ideal_circuit=circuit,
        noise_level=noise_level,
    )


def run_qermit_placeholder(ideal: float, noisy: float) -> BenchmarkResult:
    """Records whether Qermit PEC is importable in this environment."""

    try:
        from qermit.probabilistic_error_cancellation import (  # noqa: F401
            gen_PEC_learning_based_MitEx,
        )
    except ImportError:
        return BenchmarkResult(
            method="qermit.pec",
            ideal=ideal,
            noisy=noisy,
            mitigated=None,
            improvement=None,
            runtime=0.0,
            executions=0,
            notes="install qermit to enable",
        )

    return BenchmarkResult(
        method="qermit.pec",
        ideal=ideal,
        noisy=noisy,
        mitigated=None,
        improvement=None,
        runtime=0.0,
        executions=0,
        notes="requires paired device/simulator backends",
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

    executor = CountingFloatExecutor(noisy_density_executor(args.noise_level))
    representations = pec_representations(circuit, args.noise_level)
    results = [
        timed_result(
            "mitiq.pec",
            ideal,
            noisy,
            executor,
            lambda: execute_with_pec(
                circuit,
                executor,
                representations=representations,
                num_samples=args.shots,
                random_state=args.seed,
            ),
            notes=f"{args.shots} samples",
        ),
        run_qermit_placeholder(ideal, noisy),
    ]

    print_results(results)


if __name__ == "__main__":
    main()
