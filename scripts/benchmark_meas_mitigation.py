#!/usr/bin/env python3
"""
Benchmark measurement error mitigation techniques.

Compares Mitiq's REM (Readout Error Mitigation) and TREX
(Twirled Readout Error eXtinction) against other measurement error
mitigation tools.

Usage:
    python scripts/benchmark_meas_mitigation.py --circuit ghz --n-qubits 4 \
        --noise-level 0.01 --shots 8192
"""

import argparse
import time
from typing import Any, Callable, Dict

import cirq
import numpy as np

from mitiq import MeasurementResult, Observable, PauliString
from mitiq.benchmarks import (
    generate_ghz_circuit,
    generate_mirror_circuit,
    generate_quantum_volume_circuit,
)
from mitiq.experimental.trex import execute_with_trex
from mitiq.rem import execute_with_rem


def noisy_executor(
    circuit: cirq.Circuit, noise_level: float = 0.01, shots: int = 8192
) -> MeasurementResult:
    """Execute circuit with depolarizing noise and return bitstring counts."""
    # Add measurements if not present
    has_measurements = any(
        isinstance(op.gate, cirq.MeasurementGate)
        for op in circuit.all_operations()
    )

    if not has_measurements:
        qubits = sorted(circuit.all_qubits())
        circuit = circuit + cirq.measure(*qubits, key="m")

    noisy_circuit = circuit.with_noise(cirq.depolarize(noise_level))
    result = cirq.DensityMatrixSimulator().run(
        noisy_circuit, repetitions=shots
    )
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


def ideal_executor(circuit: cirq.Circuit) -> float:
    """Execute circuit without noise and return ideal expectation value."""
    # Use density matrix simulator for ideal expectation
    density_matrix = (
        cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix
    )
    n_qubits = len(circuit.all_qubits())
    # For Z...Z observable on all qubits
    obs = np.eye(1, dtype=complex)
    for i in range(n_qubits):
        obs = np.kron(obs, cirq.Z._unitary_())
    ideal_exp = np.real(np.trace(density_matrix @ obs))
    return ideal_exp


def run_mitiq_rem(
    circuit: cirq.Circuit,
    executor: Callable,
    observable: Observable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run Mitiq REM benchmark."""
    start_time = time.time()

    # Generate inverse confusion matrix from calibration data
    # For simplicity, we use a simple depolarizing model
    n_qubits = len(circuit.all_qubits())
    # Confusion matrix: with probability p, flip to wrong outcome
    p = noise_level * 10  # Readout error is typically higher than gate error
    identity = np.eye(2**n_qubits)
    flip = np.ones((2**n_qubits, 2**n_qubits)) / (2**n_qubits - 1)
    np.fill_diagonal(flip, 0)
    confusion_matrix = (1 - p) * identity + p * flip
    inverse_confusion_matrix = np.linalg.inv(confusion_matrix)

    mitigated_value = execute_with_rem(
        circuit,
        executor,
        observable,
        inverse_confusion_matrix=inverse_confusion_matrix,
    )

    runtime = time.time() - start_time
    return {
        "tool": "Mitiq REM",
        "mitigated_value": mitigated_value,
        "runtime": runtime,
    }


def run_mitiq_trex(
    circuit: cirq.Circuit,
    executor: Callable,
    observable: Observable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run Mitiq TREX benchmark."""
    start_time = time.time()

    mitigated_value = execute_with_trex(
        circuit,
        executor,
        observable,
        num_randomizations=32,
        random_state=42,
    )

    runtime = time.time() - start_time
    return {
        "tool": "Mitiq TREX",
        "mitigated_value": mitigated_value,
        "runtime": runtime,
    }


def run_mthree(
    circuit: cirq.Circuit,
    executor: Callable,
    observable: Observable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run mthree benchmark (placeholder - requires mthree installation)."""
    # TODO: Implement mthree benchmark when mthree is added to dependencies
    return {
        "tool": "mthree",
        "mitigated_value": None,
        "runtime": 0,
        "note": (
            "mthree not yet implemented - requires dependency installation"
        ),
    }


def run_qermit(
    circuit: cirq.Circuit,
    executor: Callable,
    observable: Observable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run qermit benchmark (placeholder - requires qermit installation)."""
    # TODO: Implement qermit benchmark when qermit is added to dependencies
    return {
        "tool": "qermit",
        "mitigated_value": None,
        "runtime": 0,
        "note": (
            "qermit not yet implemented - requires dependency installation"
        ),
    }


def count_gates(circuit: cirq.Circuit) -> Dict[str, int]:
    """Count single and two-qubit gates in circuit."""
    single_qubit = 0
    two_qubit = 0
    for op in circuit.all_operations():
        if len(op.qubits) == 1:
            single_qubit += 1
        elif len(op.qubits) == 2:
            two_qubit += 1
    return {"single_qubit": single_qubit, "two_qubit": two_qubit}


def print_results_table(
    ideal_value: float,
    noisy_value: float,
    results: list[Dict[str, Any]],
    circuit: cirq.Circuit,
):
    """Print comparison table of results."""
    print("\n" + "=" * 100)
    print("Measurement Error Mitigation Benchmark Results")
    print("=" * 100)

    print(f"\nCircuit: {len(circuit.all_qubits())} qubits")
    gate_counts = count_gates(circuit)
    print(
        f"Gate counts: {gate_counts['single_qubit']} single-qubit, "
        f"{gate_counts['two_qubit']} two-qubit"
    )

    print(f"\nIdeal (noiseless) expectation value: {ideal_value:.6f}")
    print(f"Noisy (unmitigated) expectation value: {noisy_value:.6f}")
    noisy_error = abs(noisy_value - ideal_value)
    print(f"Noisy error: {noisy_error:.6f}")

    print("\n" + "-" * 100)
    print(
        f"{'Tool':<20} {'Mitigated Value':<20} "
        f"{'Error':<15} {'Improvement':<15} {'Runtime (s)':<15}"
    )
    print("-" * 100)

    for result in results:
        tool = result["tool"]
        mitigated = result["mitigated_value"]
        runtime = result.get("runtime", 0)

        if mitigated is None:
            print(
                f"{tool:<20} {'N/A':<20} {'N/A':<15} "
                f"{'N/A':<15} {runtime:<15.3f}"
            )
            if "note" in result:
                print(f"  Note: {result['note']}")
            continue

        mitigated_error = abs(mitigated - ideal_value)
        improvement = (
            noisy_error / mitigated_error
            if mitigated_error > 0
            else float("inf")
        )

        print(
            f"{tool:<20} {mitigated:<20.6f} {mitigated_error:<15.6f} "
            f"{improvement:<15.4f} {runtime:<15.3f}"
        )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark measurement error mitigation techniques"
    )
    parser.add_argument(
        "--circuit",
        type=str,
        choices=["ghz", "qv", "mirror"],
        default="ghz",
        help="Type of circuit to benchmark (default: ghz)",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="Number of qubits (default: 4)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.01,
        help="Depolarizing noise level (default: 0.01)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=8192,
        help="Number of shots (default: 8192)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Circuit depth for QV and mirror circuits (default: 5)",
    )

    args = parser.parse_args()

    # Generate circuit
    print(
        f"Generating {args.circuit.upper()} circuit with "
        f"{args.n_qubits} qubits..."
    )
    if args.circuit == "ghz":
        circuit = generate_ghz_circuit(args.n_qubits)
    elif args.circuit == "qv":
        circuit, _ = generate_quantum_volume_circuit(args.n_qubits, args.depth)
    elif args.circuit == "mirror":
        import networkx as nx

        # Use linear connectivity for simplicity
        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from(range(args.n_qubits))
        for i in range(args.n_qubits - 1):
            connectivity_graph.add_edge(i, i + 1)
        circuit, _ = generate_mirror_circuit(
            args.depth, 0.5, connectivity_graph, seed=42
        )

    # Define observable (Z...Z on all qubits)
    n_qubits = len(circuit.all_qubits())
    pauli_string = "Z" * n_qubits
    observable = Observable(PauliString(pauli_string))

    # Run ideal executor
    print("Computing ideal (noiseless) expectation value...")
    ideal_value = ideal_executor(circuit)

    # Run noisy executor
    print(f"Running noisy executor with noise level {args.noise_level}...")
    noisy_value = noisy_executor(
        circuit, noise_level=args.noise_level, shots=args.shots
    )
    noisy_exp = observable._expectation_from_measurements([noisy_value])

    # Create executor wrapper for mitiq functions
    def executor(circuit: cirq.Circuit) -> MeasurementResult:
        return noisy_executor(
            circuit, noise_level=args.noise_level, shots=args.shots
        )

    # Run benchmarks
    results = []

    print("\nRunning Mitiq REM...")
    results.append(
        run_mitiq_rem(
            circuit, executor, observable, args.noise_level, args.shots
        )
    )

    print("Running Mitiq TREX...")
    results.append(
        run_mitiq_trex(
            circuit, executor, observable, args.noise_level, args.shots
        )
    )

    print("Running mthree (placeholder)...")
    results.append(
        run_mthree(circuit, executor, observable, args.noise_level, args.shots)
    )

    print("Running qermit (placeholder)...")
    results.append(
        run_qermit(circuit, executor, observable, args.noise_level, args.shots)
    )

    # Print results
    print_results_table(ideal_value, noisy_exp.real, results, circuit)


if __name__ == "__main__":
    main()
