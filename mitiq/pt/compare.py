# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools for comparing PT vs PT+QEM performance."""

from collections.abc import Callable
from typing import Any

import numpy as np
import cirq
from cirq import Circuit

from mitiq import QPROGRAM, zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory
from mitiq.pt import generate_pauli_twirl_variants


def compare_performance(
    circuit: QPROGRAM,
    executor: Callable[[QPROGRAM], float],
    noise_levels: list[float] | None = None,
    num_twirled_circuits: int = 10,
    qem_strategy: str | None = "zne",
    observable: Any | None = None,
) -> dict[str, Any]:
    """Compare performance of PT alone vs PT + QEM techniques.

    Generates a plot showing when Pauli Twirling (PT) alone is useful vs
    when combining PT with Quantum Error Mitigation (QEM) techniques
    (like ZNE) provides better results.

    Args:
        circuit: The input circuit to benchmark
        executor: Function that executes a circuit and returns expectation value
        noise_levels: List of noise probabilities to test (default: [0.001, 0.005, 0.01, 0.02, 0.05])
        num_twirled_circuits: Number of Pauli twirled variants to generate
        qem_strategy: QEM technique to combine with PT ("zne", "ddd", or None for PT only)
        observable: Observable to measure (if None, assumes executor returns expectation)

    Returns:
        Dictionary containing:
        - 'noise_levels': The noise levels tested
        - 'ideal_value': Ideal noise-free expectation value
        - 'raw_errors': Error without any mitigation
        - 'pt_errors': Error with PT alone
        - 'qem_errors': Error with QEM alone (if qem_strategy specified)
        - 'pt_qem_errors': Error with PT + QEM combined
        - 'best_strategy': Which strategy performed best at each noise level
    """
    if noise_levels is None:
        noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]

    # Get ideal value (noise-free simulation)
    ideal_executor = _create_noisy_executor(executor, noise_prob=0.0)
    ideal_value = ideal_executor(circuit)

    results = {
        'noise_levels': noise_levels,
        'ideal_value': ideal_value,
        'raw_errors': [],
        'pt_errors': [],
        'qem_errors': [] if qem_strategy else None,
        'pt_qem_errors': [] if qem_strategy else None,
        'best_strategy': [],
    }

    for noise_prob in noise_levels:
        # Create noisy executor
        noisy_executor = _create_noisy_executor(executor, noise_prob)

        # Raw execution (no mitigation)
        raw_result = noisy_executor(circuit)
        raw_error = abs(raw_result - ideal_value)
        results['raw_errors'].append(raw_error)

        # PT only: average over twirled circuits
        twirled_circuits = generate_pauli_twirl_variants(
            circuit, num_circuits=num_twirled_circuits
        )
        pt_results = [noisy_executor(c) for c in twirled_circuits]
        pt_result = np.mean(pt_results)
        pt_error = abs(pt_result - ideal_value)
        results['pt_errors'].append(pt_error)

        # QEM only (if specified)
        if qem_strategy:
            if qem_strategy.lower() == 'zne':
                # Apply ZNE without PT
                qem_result = zne.execute_with_zne(
                    circuit,
                    noisy_executor,
                    factory=RichardsonFactory([1.0, 2.0, 3.0])
                )
                qem_error = abs(qem_result - ideal_value)
                results['qem_errors'].append(qem_error)

                # PT + QEM combined
                pt_qem_results = []
                for twirled_circuit in twirled_circuits:
                    mitigated = zne.execute_with_zne(
                        twirled_circuit,
                        noisy_executor,
                        factory=RichardsonFactory([1.0, 2.0, 3.0])
                    )
                    pt_qem_results.append(mitigated)
                pt_qem_result = np.mean(pt_qem_results)
                pt_qem_error = abs(pt_qem_result - ideal_value)
                results['pt_qem_errors'].append(pt_qem_error)

                # Determine best strategy
                errors = {
                    'raw': raw_error,
                    'pt': pt_error,
                    'zne': qem_error,
                    'pt+zne': pt_qem_error,
                }
            else:
                # For other QEM strategies, just compare PT vs raw
                errors = {
                    'raw': raw_error,
                    'pt': pt_error,
                }

            best = min(errors, key=errors.get)
            results['best_strategy'].append(best)
        else:
            # PT only comparison
            if pt_error < raw_error:
                results['best_strategy'].append('pt')
            else:
                results['best_strategy'].append('raw')

    return results


def plot_comparison(results: dict[str, Any], title: str = "PT vs PT+QEM Performance") -> None:
    """Plot the comparison results from compare_performance().

    Creates a visualization showing which technique performs best at
    different noise levels. Lower error is better.

    Args:
        results: Dictionary output from compare_performance()
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    noise_levels = results['noise_levels']

    plt.figure(figsize=(10, 6))

    # Plot each strategy
    plt.plot(noise_levels, results['raw_errors'], 'o-', label='Raw (no mitigation)', linewidth=2, markersize=8)
    plt.plot(noise_levels, results['pt_errors'], 's-', label='PT only', linewidth=2, markersize=8)

    if results['qem_errors']:
        plt.plot(noise_levels, results['qem_errors'], '^-', label='ZNE only', linewidth=2, markersize=8)
        plt.plot(noise_levels, results['pt_qem_errors'], 'd-', label='PT + ZNE', linewidth=2, markersize=8)

    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Ideal (zero error)')

    plt.xlabel('Noise Level (depolarizing probability)', fontsize=12)
    plt.ylabel('Absolute Error |⟨O⟩ - ⟨O⟩_ideal|', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add text annotation showing best strategy summary
    strategy_counts = {}
    for s in results['best_strategy']:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    best_overall = max(strategy_counts, key=strategy_counts.get)
    annotation = f"Best overall: {best_overall.upper()} ({strategy_counts[best_overall]}/{len(results['best_strategy'])} noise levels)"
    plt.figtext(0.5, 0.02, annotation, ha='center', fontsize=10, style='italic')

    plt.show()


def _create_noisy_executor(
    base_executor: Callable[[QPROGRAM], float],
    noise_prob: float
) -> Callable[[QPROGRAM], float]:
    """Create an executor that adds depolarizing noise before execution."""
    def noisy_executor(circuit: QPROGRAM) -> float:
        if noise_prob > 0:
            # Add depolarizing noise to the circuit
            noisy_circuit = _add_depolarizing_noise(circuit, noise_prob)
            return base_executor(noisy_circuit)
        return base_executor(circuit)
    return noisy_executor


def _add_depolarizing_noise(circuit: QPROGRAM, noise_prob: float) -> QPROGRAM:
    """Add depolarizing noise after each gate in the circuit."""
    if isinstance(circuit, Circuit):
        noisy_circuit = cirq.Circuit()
        for moment in circuit:
            noisy_circuit.append(moment)
            # Add noise to all qubits in this moment
            for op in moment:
                for qubit in op.qubits:
                    noisy_circuit.append(cirq.depolarize(p=noise_prob)(qubit))
        return noisy_circuit
    # For other circuit types, return as-is (assume executor handles noise)
    return circuit
