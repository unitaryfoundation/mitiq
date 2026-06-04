# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level debiasing and sharpening error mitigation tools."""

from typing import Callable, Dict

from mitiq import QPROGRAM, Executor, QuantumResult
from mitiq.experimental.deb.symmetrization import construct_circuits
from mitiq.experimental.deb.sharpening import sharpen


def execute_with_debiasing(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    num_variants: int = 10,
    random_state: int | None = None,
) -> Dict[str, float]:
    """Execute circuit with debiasing (symmetrization) error mitigation.

    Generates multiple Pauli-symmetrized circuit variants, executes them,
    and returns the averaged probability distribution.

    Args:
        circuit: The input circuit to execute with debiasing.
        executor: A Mitiq executor that executes a circuit and returns
            measurement results as a bitstring count dictionary.
        num_variants: Number of circuit variants to generate.
        random_state: Seed for random number generation.

    Returns:
        The debiased probability distribution (averaged across variants).
    """
    # Generate symmetrized circuit variants
    variants = construct_circuits(circuit, num_variants, random_state)

    # Execute each variant
    results = []
    for variant in variants:
        result = executor(variant)
        results.append(result)

    # Average the probability distributions
    averaged_dist: Dict[str, float] = {}
    for counts in results:
        total_shots = sum(counts.values())
        if total_shots > 0:
            for bitstring, count in counts.items():
                prob = count / total_shots
                averaged_dist[bitstring] = averaged_dist.get(bitstring, 0) + prob

    # Normalize by number of variants
    for bitstring in averaged_dist:
        averaged_dist[bitstring] /= len(results)

    return averaged_dist


def execute_with_debiasing_and_sharpening(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    num_variants: int = 10,
    random_state: int | None = None,
    sharpen_threshold: int = 2,
) -> Dict[str, float]:
    """Execute circuit with debiasing and sharpening error mitigation.

    Generates multiple Pauli-symmetrized circuit variants, executes them,
    and applies plurality voting (sharpening) to the results.

    Args:
        circuit: The input circuit to execute with debiasing and sharpening.
        executor: A Mitiq executor that executes a circuit and returns
            measurement results as a bitstring count dictionary.
        num_variants: Number of circuit variants to generate.
        random_state: Seed for random number generation.
        sharpen_threshold: Minimum number of votes required for a bitstring
            to win in plurality voting.

    Returns:
        The debiased and sharpened probability distribution.
    """
    # Generate symmetrized circuit variants
    variants = construct_circuits(circuit, num_variants, random_state)

    # Execute each variant
    results = []
    for variant in variants:
        result = executor(variant)
        results.append(result)

    # Apply sharpening (plurality voting)
    sharpened_dist = sharpen(results, threshold=sharpen_threshold)

    return sharpened_dist
