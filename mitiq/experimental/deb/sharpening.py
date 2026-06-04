# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Sharpening (plurality voting) for debiasing error mitigation."""

from typing import Dict, List

import numpy as np


def sharpen(
    results: List[Dict[str, int]],
    threshold: int = 2,
) -> Dict[str, float]:
    """Apply plurality voting (sharpening) to bitstring count dictionaries.

    For each shot, the most commonly occurring bitstring across all
    symmetrized variants is selected. This is particularly effective when
    the target answer is concentrated in a small number of bitstrings.

    Args:
        results: List of bitstring count dictionaries, one per variant.
        threshold: Minimum number of votes required for a bitstring to win.
            If no winner is found, threshold is reduced by 1 until threshold=2.
            If still no winner, componentwise average is returned.

    Returns:
        A single sharpened probability distribution (normalized).
    """
    if not results:
        return {}

    # Convert each count dict to list of bitstrings (expand by counts)
    all_bitstrings_per_variant = []
    for counts in results:
        bitstrings = []
        for bitstring, count in counts.items():
            bitstrings.extend([bitstring] * count)
        all_bitstrings_per_variant.append(bitstrings)

    # Ensure all variants have the same number of shots
    n_shots = len(all_bitstrings_per_variant[0])
    for bitstrings in all_bitstrings_per_variant:
        if len(bitstrings) != n_shots:
            raise ValueError("All variants must have the same number of shots")

    # Perform plurality voting for each shot
    winning_bitstrings = []
    for shot_idx in range(n_shots):
        # Collect bitstrings from all variants for this shot
        shot_bitstrings = [
            variant[shot_idx] for variant in all_bitstrings_per_variant
        ]

        # Count votes
        vote_counts: Dict[str, int] = {}
        for bitstring in shot_bitstrings:
            vote_counts[bitstring] = vote_counts.get(bitstring, 0) + 1

        # Find winner with votes >= threshold
        winner = None
        current_threshold = threshold
        while current_threshold >= 2:
            for bitstring, count in vote_counts.items():
                if count >= current_threshold:
                    winner = bitstring
                    break
            if winner:
                break
            current_threshold -= 1

        # If no winner, use componentwise average fallback
        if not winner:
            # Average the histograms
            total_counts: Dict[str, float] = {}
            for counts in results:
                for bitstring, count in counts.items():
                    total_counts[bitstring] = total_counts.get(bitstring, 0) + count

            # Normalize
            total = sum(total_counts.values())
            if total > 0:
                return {k: v / total for k, v in total_counts.items()}
            else:
                return {}

        winning_bitstrings.append(winner)

    # Build final histogram from winning bitstrings
    final_counts: Dict[str, int] = {}
    for bitstring in winning_bitstrings:
        final_counts[bitstring] = final_counts.get(bitstring, 0) + 1

    # Normalize
    total = sum(final_counts.values())
    if total > 0:
        return {k: v / total for k, v in final_counts.items()}
    else:
        return {}
