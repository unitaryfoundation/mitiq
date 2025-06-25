# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools for constructing the noise-amplified representations of ideal
operations.
"""

from collections.abc import Sequence

from cirq import Circuit

from mitiq.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec import OperationRepresentation


def scale_circuit_amplifications(
    ideal_circuit: Circuit,
    scale_factor: float,
    noise_model: str,
    epsilon: float,
) -> Sequence[OperationRepresentation]:
    r"""Generates a list of implementable sequences from the noise-amplified
    representation of the input ideal circuit based on the input noise model
    and baseline noise level.

    Args:
        ideal_circuit: The ideal circuit from which an implementable
            sequence is sampled.
        scale_factor: A (positive) number by which the baseline noise
            level is to be amplified.
        noise_model: A string describing the noise model to be used for the
            noise-scaled representations, e.g. "local_depolarizing" or
            "global_depolarizing".
        epsilon: Baseline noise level.

    Returns:
        A list of noise-amplified circuits, corresponding to each scale
        factor multiplied by the baseline noise level."""

    if noise_model == "local_depolarizing":
        amp_fn = amplify_noisy_ops_in_circuit_with_local_depolarizing_noise
        # TODO add other existing noise models from Mitiq
    elif noise_model == "global_depolarizing":
        amp_fn = amplify_noisy_ops_in_circuit_with_global_depolarizing_noise
    else:
        raise ValueError("Noise model not supported")
        # TODO allow use of custom noise model

    return amp_fn(ideal_circuit, scale_factor * epsilon)
