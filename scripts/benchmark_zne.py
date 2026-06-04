#!/usr/bin/env python3
"""
Benchmark Zero Noise Extrapolation (ZNE) techniques.

Compares Mitiq's ZNE against other ZNE implementations.

Usage:
    python scripts/benchmark_zne.py --circuit ghz --n-qubits 4 \
        --noise-level 0.01 --shots 8192
"""

import argparse
import time
from typing import Any, Callable, Dict

import cirq
import numpy as np

from mitiq import MeasurementResult, Observable, PauliString, zne
from mitiq.benchmarks import (
    generate_ghz_circuit,
    generate_mirror_circuit,
    generate_quantum_volume_circuit,
)
from mitiq.zne.inference import ExpFactory, LinearFactory, RichardsonFactory
from mitiq.zne.scaling import (
    fold_gates_at_random,
    fold_global,
    insert_id_layers,
)


def noisy_executor(
    circuit: cirq.Circuit, noise_level: float = 0.01, shots: int = 8192
) -> float:
    """Execute circuit with depolarizing noise and return expectation value."""
    # Add measurements if not present
    if not list(circuit.all_operations()):
        # Empty circuit, return 0
        return 0.0

    # Check if circuit has measurements
    has_measurements = any(
        isinstance(op.gate, cirq.MeasurementGate)
        for op in circuit.all_operations()
    )

    if not has_measurements:
        # Add measurements to all qubits
        qubits = sorted(circuit.all_qubits())
        circuit = circuit + cirq.measure(*qubits, key="m")

    noisy_circuit = circuit.with_noise(cirq.depolarize(noise_level))
    result = cirq.DensityMatrixSimulator().run(
        noisy_circuit, repetitions=shots
    )
    bitstrings = np.column_stack(list(result.measurements.values()))
    meas_result = MeasurementResult(bitstrings)

    # For Z...Z observable
    n_qubits = len(circuit.all_qubits())
    pauli_string = "Z" * n_qubits
    observable = Observable(PauliString(pauli_string))
    exp_value = observable._expectation_from_measurements([meas_result])
    return exp_value.real


def ideal_executor(circuit: cirq.Circuit) -> float:
    """Execute circuit without noise and return ideal expectation value."""
    # Use density matrix simulator for ideal expectation
    density_matrix = (
        cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix
    )
    n_qubits = len(circuit.all_qubits())
    obs = np.eye(1, dtype=complex)
    for i in range(n_qubits):
        obs = np.kron(obs, cirq.Z._unitary_())
    ideal_exp = np.real(np.trace(density_matrix @ obs))
    return ideal_exp


def run_mitiq_zne_linear(
    circuit: cirq.Circuit,
    executor: Callable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run Mitiq ZNE with LinearFactory."""
    start_time = time.time()

    mitigated_executor = zne.mitigate_executor(
        executor,
        scale_noise=fold_global,
        factory=LinearFactory([1.0, 3.0, 5.0]),
    )

    mitigated_value = mitigated_executor(circuit)

    runtime = time.time() - start_time
    return {
        "tool": "Mitiq ZNE (Linear)",
        "mitigated_value": mitigated_value,
        "runtime": runtime,
    }


def run_mitiq_zne_richardson(
    circuit: cirq.Circuit,
    executor: Callable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run Mitiq ZNE with RichardsonFactory."""
    start_time = time.time()

    mitigated_executor = zne.mitigate_executor(
        executor,
        scale_noise=fold_global,
        factory=RichardsonFactory([1.0, 3.0, 5.0]),
    )

    mitigated_value = mitigated_executor(circuit)

    runtime = time.time() - start_time
    return {
        "tool": "Mitiq ZNE (Richardson)",
        "mitigated_value": mitigated_value,
        "runtime": runtime,
    }


def run_mitiq_zne_exp(
    circuit: cirq.Circuit,
    executor: Callable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run Mitiq ZNE with ExpFactory."""
    start_time = time.time()

    # For depolarizing noise, asymptote = 1/2^n_qubits
    n_qubits = len(circuit.all_qubits())
    asymptote = 1.0 / (2**n_qubits)

    mitigated_executor = zne.mitigate_executor(
        executor,
        scale_noise=fold_global,
        factory=ExpFactory([1.0, 3.0, 5.0], asymptote=asymptote),
    )

    mitigated_value = mitigated_executor(circuit)

    runtime = time.time() - start_time
    return {
        "tool": "Mitiq ZNE (Exponential)",
        "mitigated_value": mitigated_value,
        "runtime": runtime,
    }


def run_mitiq_zne_random_folding(
    circuit: cirq.Circuit,
    executor: Callable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run Mitiq ZNE with random gate folding."""
    start_time = time.time()

    mitigated_executor = zne.mitigate_executor(
        executor,
        scale_noise=fold_gates_at_random,
        factory=LinearFactory([1.0, 3.0, 5.0]),
    )

    mitigated_value = mitigated_executor(circuit)

    runtime = time.time() - start_time
    return {
        "tool": "Mitiq ZNE (Random Folding)",
        "mitigated_value": mitigated_value,
        "runtime": runtime,
    }


def run_mitiq_zne_id_layers(
    circuit: cirq.Circuit,
    executor: Callable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run Mitiq ZNE with identity layer insertion."""
    start_time = time.time()

    mitigated_executor = zne.mitigate_executor(
        executor,
        scale_noise=insert_id_layers,
        factory=RichardsonFactory([1.0, 3.0, 5.0]),
    )

    mitigated_value = mitigated_executor(circuit)

    runtime = time.time() - start_time
    return {
        "tool": "Mitiq ZNE (ID Layers)",
        "mitigated_value": mitigated_value,
        "runtime": runtime,
    }


def run_qiskit_zne(
    circuit: cirq.Circuit,
    executor: Callable,
    observable: Observable,
    noise_level: float,
    shots: int,
) -> Dict[str, Any]:
    """Run manual ZNE benchmark using Cirq (simulating Qiskit-style ZNE)."""
    try:
        start_time = time.time()

        # Manual ZNE implementation using same simulator as Mitiq
        # for fair comparison. This simulates what Qiskit's ZNE would do
        from mitiq.zne.scaling import fold_global

        scale_factors = [1, 3, 5]
        scale_results = []

        for scale in scale_factors:
            # Scale circuit noise
            if scale > 1:
                scaled_circuit = fold_global(circuit, scale_factor=scale)
            else:
                scaled_circuit = circuit

            # Run with noisy executor
            exp_val = executor(scaled_circuit)
            scale_results.append(exp_val)

        # Linear extrapolation (same as Qiskit's default)
        x = np.array(scale_factors)
        y = np.array(scale_results)
        coeffs = np.polyfit(x, y, 1)
        mitigated_value = coeffs[1]  # y-intercept (scale=0)

        runtime = time.time() - start_time

        return {
            "tool": "Manual ZNE (Linear Extrapolation)",
            "mitigated_value": mitigated_value,
            "runtime": runtime,
        }

    except Exception as e:
        return {
            "tool": "Manual ZNE",
            "mitigated_value": None,
            "runtime": 0,
            "note": f"Runtime error: {e}",
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
    print("Zero Noise Extrapolation (ZNE) Benchmark Results")
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
        f"{'Tool':<30} {'Mitigated Value':<20} "
        f"{'Error':<15} {'Improvement':<15} {'Runtime (s)':<15}"
    )
    print("-" * 100)

    for result in results:
        tool = result["tool"]
        mitigated = result["mitigated_value"]
        runtime = result.get("runtime", 0)

        if mitigated is None:
            print(
                f"{tool:<30} {'N/A':<20} {'N/A':<15} "
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
            f"{tool:<30} {mitigated:<20.6f} {mitigated_error:<15.6f} "
            f"{improvement:<15.4f} {runtime:<15.3f}"
        )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Zero Noise Extrapolation (ZNE) techniques"
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
    print(
        f"Running noisy executor with noise level {args.noise_level}..."
    )
    noisy_value = noisy_executor(
        circuit, noise_level=args.noise_level, shots=args.shots
    )

    # Create executor wrapper
    def executor(circuit: cirq.Circuit) -> float:
        return noisy_executor(
            circuit, noise_level=args.noise_level, shots=args.shots
        )

    # Run benchmarks
    results = []

    print("\nRunning Mitiq ZNE (Linear)...")
    results.append(
        run_mitiq_zne_linear(
            circuit, executor, args.noise_level, args.shots
        )
    )

    print("Running Mitiq ZNE (Richardson)...")
    results.append(
        run_mitiq_zne_richardson(
            circuit, executor, args.noise_level, args.shots
        )
    )

    print("Running Mitiq ZNE (Exponential)...")
    results.append(
        run_mitiq_zne_exp(
            circuit, executor, args.noise_level, args.shots
        )
    )

    print("Running Mitiq ZNE (Random Folding)...")
    results.append(
        run_mitiq_zne_random_folding(
            circuit, executor, args.noise_level, args.shots
        )
    )

    print("Running Mitiq ZNE (ID Layers)...")
    results.append(
        run_mitiq_zne_id_layers(
            circuit, executor, args.noise_level, args.shots
        )
    )

    print("Running Qiskit ZNE (placeholder)...")
    results.append(
        run_qiskit_zne(
            circuit, executor, observable, args.noise_level, args.shots
        )
    )

    # Print results
    print_results_table(ideal_value, noisy_value, results, circuit)


if __name__ == "__main__":
    main()
