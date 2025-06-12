# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level probabilistic error amplification tools."""

import warnings

import numpy as np
from cirq import Circuit

from mitiq.pea.scale_amplifications import scale_circuit_amplifications
from mitiq.pec.pec import (
    _LARGE_SAMPLE_WARN,
    LargeSampleWarning,
    sample_circuit,
)
from mitiq.typing import QPROGRAM


def construct_circuits(
    circuit: Circuit,
    scale_factors: list[float],
    noise_model: str,
    epsilon: float,
    random_state: int | np.random.RandomState | None = None,
    precision: float = 0.1,
    num_samples: int | None = None,
    full_output: bool = True,
) -> (
    list[list[QPROGRAM]]
    | tuple[list[list[QPROGRAM]], list[list[int]], list[float]]
):
    """Samples lists of implementable circuits from the noise-amplified
    representation of the input ideal circuit at each input noise scale
    factor.

    Note that the ideal operation can be a sequence of operations (circuit),
    for instance U = V W, as long as a representation is known. Similarly, A
    and B can be sequences of operations (circuits) or just single operations.

    Args:
        circuit: The ideal circuit from which an implementable
            sequence is sampled.
        scale_factors: A list of (positive) numbers by which the baseline
            noise level is to be amplified.
        noise_model: A string describing the noise model to be used for the
            noise-scaled representations, e.g. "local_depolarizing" or
            "global_depolarizing".
        epsilon: Baseline noise level.
        random_state: The random state or seed for reproducibility.
        precision: The desired precision for the sampling process.
            Default is 0.1.
        num_samples: The number of noisy circuits to be sampled for PEA.
            If not given, this is deduced from the 'precision'.
        full_output: If ``full_output`` is True, a list of lists of signs and a
            list of norms, corresponding to each noise scale factor are
            returned.

    Returns:
        A list of lists of sampled circuits, where each list of circuits
        corresponds to an input noise scale factor times the baseline noise
        level. If ``full_output`` is True, also returns a list of lists of
        signs and a list of norms, corresponding to each noise scale factor.

    Raises:
        ValueError: If the precision is not within the interval (0, 1].
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if not (0 < precision <= 1):
        raise ValueError(
            "The value of 'precision' should be within the interval (0, 1],"
            f" but precision is {precision}."
        )

    # Get the 1-norm of the circuit quasi-probability representation
    _, _, norm = sample_circuit(
        circuit,
        scale_circuit_amplifications(circuit, 1.0, noise_model, epsilon),
        num_samples=1,
    )

    # Deduce the number of samples (if not given by the user)
    if num_samples is None:
        num_samples = int((norm / precision) ** 2)

    if num_samples > 10**5:
        warnings.warn(_LARGE_SAMPLE_WARN, LargeSampleWarning)

    scaled_sampled_circuits = []
    scaled_signs = []
    scaled_norms = []
    for s in scale_factors:
        sampled_circuits, signs, norm = sample_circuit(
            circuit,
            scale_circuit_amplifications(circuit, s, noise_model, epsilon),
            num_samples=num_samples,
            random_state=random_state,
        )
        scaled_sampled_circuits.append(sampled_circuits)
        scaled_signs.append(signs)
        scaled_norms.append(norm)

    if full_output:
        return scaled_sampled_circuits, scaled_signs, scaled_norms
    return scaled_sampled_circuits
