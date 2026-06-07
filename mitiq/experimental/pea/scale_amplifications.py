# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools for constructing the noise-amplified representations of ideal
operations.
"""

from collections.abc import Sequence

from cirq import Circuit

from mitiq.experimental.pea.amplifications.amplify_depolarizing import (
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


def scale_representations(
    representations: Sequence[OperationRepresentation],
    scale_factor: float,
) -> list[OperationRepresentation]:
    r"""Amplifies a list of quasi-probability representations by a scale factor
    using canonical noise scaling.

    This is the representation-based counterpart of
    :func:`scale_circuit_amplifications`. Instead of rebuilding the
    amplifications from a known noise model, it scales user-supplied
    representations (e.g. learned from hardware characterization) so that PEA
    can be applied to arbitrary noise.

    The scaling follows the "Canonical noise scaling" prescription of Section D
    of :cite:`Mari_2021_PRA`. That construction shows a quasi-probability
    representation of an ideal gate induces a canonical noise channel that
    depends linearly on a single noise parameter, so noise scaling is always
    well defined for an arbitrary representation. Each representation is a
    quasi-probability decomposition of an ideal operation into the ideal
    operation itself (the "identity" term, coefficient :math:`a_0`) plus a set
    of error terms (coefficients :math:`a_k`). The deviation from the identity
    is amplified linearly by the scale factor :math:`s`:

    .. math::

        a_0 \rightarrow 1 - s\,(1 - a_0), \qquad a_k \rightarrow s\,a_k .

    This preserves the unit sum of the coefficients and, for the global
    depolarizing model (and any single-qubit depolarizing channel), reproduces
    the analytic rebuild at noise level :math:`s\,\epsilon` exactly at every
    scale factor.

    Args:
        representations: The base (unscaled) quasi-probability representations,
            one per unique operation in the circuit.
        scale_factor: A (positive) number by which the baseline noise is
            amplified.

    Returns:
        A new list of representations with coefficients scaled by
        ``scale_factor``. The ideal operations and the implementable noisy
        operations are unchanged; only the coefficients differ.

    Raises:
        ValueError: If a representation has no identity term (a noisy operation
            equal to its ideal operation) or that term has zero weight, so the
            deviation from the identity cannot be determined.

    .. note::
        For the two-qubit ``local_depolarizing`` model the per-qubit channels
        combine as a tensor product, so the identity coefficient depends
        quadratically on the noise level. Linear deviation scaling cannot
        reproduce that exactly; use ``global_depolarizing`` representations (or
        single-qubit operations) when exact equivalence with the legacy
        ``noise_model`` path is required.
    """
    scaled = []
    for rep in representations:
        ideal = rep.ideal
        identity_indices = [
            i
            for i, op in enumerate(rep.noisy_operations)
            if op.circuit == ideal
        ]
        identity_weight = sum(rep.coeffs[i] for i in identity_indices)
        if not identity_indices or identity_weight == 0:
            raise ValueError(
                "Cannot scale a representation without a non-zero identity "
                "term (a noisy operation equal to the ideal operation). The "
                "canonical noise scaling needs it to measure the deviation "
                "from the identity."
            )

        scaled_identity_weight = 1.0 - scale_factor * (1.0 - identity_weight)

        new_coeffs = []
        for i, coeff in enumerate(rep.coeffs):
            if i in identity_indices:
                # Split the scaled identity weight across the identity term(s)
                # in their original proportions (a single term in practice).
                new_coeffs.append(
                    float(coeff / identity_weight * scaled_identity_weight)
                )
            else:
                new_coeffs.append(float(scale_factor * coeff))

        scaled.append(
            OperationRepresentation(
                rep.ideal,
                rep.noisy_operations,
                new_coeffs,
                rep.is_qubit_dependent,
            )
        )
    return scaled
