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

from mitiq.interface import convert_from_mitiq
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global


def qiskit_noise_model_for_circuit(qiskit_circuit, noise_level: float):
    """Builds a Qiskit Aer noise model matching gates in a circuit."""

    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()
    seen_errors = set()
    for instruction in qiskit_circuit.data:
        qubits = tuple(
            qiskit_circuit.find_bit(q).index for q in instruction.qubits
        )
        if len(qubits) not in (1, 2):
            continue
        gate_name = instruction.operation.name
        error_key = (gate_name, qubits)
        if error_key in seen_errors:
            continue
        seen_errors.add(error_key)
        noise_model.add_quantum_error(
            depolarizing_error(noise_level, len(qubits)),
            gate_name,
            qubits,
        )
    return noise_model


def qiskit_aer_density_executor(
    noise_level: float,
    seed: int,
) -> CountingFloatExecutor | None:
    """Builds a Qiskit Aer density-matrix executor when dependencies exist."""

    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        return None

    def execute(circuit: cirq.Circuit) -> float:
        qiskit_circuit = convert_from_mitiq(circuit, "qiskit")

        qiskit_circuit.save_density_matrix()
        simulator = AerSimulator(
            method="density_matrix",
            noise_model=qiskit_noise_model_for_circuit(
                qiskit_circuit, noise_level
            ),
            seed_simulator=seed,
        )
        result = simulator.run(qiskit_circuit).result()
        density_matrix = np.asarray(result.data(0)["density_matrix"])
        return zz_expectation_from_density_matrix(
            density_matrix, qiskit_circuit.num_qubits
        )

    return CountingFloatExecutor(execute)


def run_qiskit_aer_manual_zne(
    circuit: cirq.Circuit,
    ideal: float,
    noisy: float,
    noise_level: float,
    seed: int,
) -> BenchmarkResult:
    """Runs a manual folded-circuit ZNE baseline with Qiskit Aer."""

    executor = qiskit_aer_density_executor(noise_level, seed)
    if executor is None:
        return BenchmarkResult(
            method="qiskit-aer.manual-zne",
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
        method="qiskit-aer.manual-zne",
        ideal=ideal,
        noisy=qiskit_noisy,
        mitigated=mitigated,
        improvement=improvement_factor(ideal, qiskit_noisy, mitigated),
        runtime=runtime,
        executions=executor.calls,
        notes="manual fold_global + linear fit",
    )


def run_qermit_zne(
    circuit: cirq.Circuit,
    ideal: float,
    noisy: float,
    noise_level: float,
    shots: int,
    seed: int,
) -> BenchmarkResult:
    """Runs Qermit zero-noise extrapolation with a noisy Aer backend."""

    try:
        from pytket.extensions.qiskit import AerBackend, qiskit_to_tk
        from pytket.pauli import Pauli, QubitPauliString
        from pytket.utils import QubitPauliOperator
        from qermit import (
            AnsatzCircuit,
            ObservableExperiment,
            ObservableTracker,
        )
        from qermit.zero_noise_extrapolation import (
            Fit,
            Folding,
            gen_ZNE_MitEx,
        )
    except ImportError:
        return BenchmarkResult(
            method="qermit.zne",
            ideal=ideal,
            noisy=noisy,
            mitigated=None,
            improvement=None,
            runtime=0.0,
            executions=0,
            notes="install qermit to enable",
        )

    scale_factors = [1, 3, 5]
    try:
        qiskit_circuit = convert_from_mitiq(circuit, "qiskit")
        pytket_circuit = qiskit_to_tk(qiskit_circuit)
        backend = AerBackend(
            noise_model=qiskit_noise_model_for_circuit(
                qiskit_circuit, noise_level
            ),
            n_qubits=len(pytket_circuit.qubits),
        )
        pauli_string = QubitPauliString(
            pytket_circuit.qubits,
            [Pauli.Z] * len(pytket_circuit.qubits),
        )
        observable = QubitPauliOperator({pauli_string: 1.0})
        experiment = ObservableExperiment(
            AnsatzCircuit(pytket_circuit, shots, {}),
            ObservableTracker(observable),
        )
        mitex = gen_ZNE_MitEx(
            backend,
            scale_factors,
            fit_type=Fit.linear,
            folding_type=Folding.circuit,
            seed=seed,
        )
        started_at = time.perf_counter()
        qermit_result = mitex.run([experiment])[0]
        runtime = time.perf_counter() - started_at
        mitigated = float(np.real_if_close(qermit_result.get(pauli_string, 0)))
    except Exception as error:
        return BenchmarkResult(
            method="qermit.zne",
            ideal=ideal,
            noisy=noisy,
            mitigated=None,
            improvement=None,
            runtime=0.0,
            executions=0,
            notes=f"qermit run failed: {type(error).__name__}",
        )

    return BenchmarkResult(
        method="qermit.zne",
        ideal=ideal,
        noisy=noisy,
        mitigated=mitigated,
        improvement=improvement_factor(ideal, noisy, mitigated),
        runtime=runtime,
        executions=len(scale_factors),
        notes="circuit folding + linear fit",
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
        run_qiskit_aer_manual_zne(
            circuit, ideal, noisy, args.noise_level, args.seed
        )
    )
    results.append(
        run_qermit_zne(
            circuit, ideal, noisy, args.noise_level, args.shots, args.seed
        )
    )

    print_results(results)


if __name__ == "__main__":
    main()
