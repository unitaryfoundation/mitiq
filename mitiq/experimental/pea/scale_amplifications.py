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


# Coefficients with magnitude below this are treated as zero when classifying
# the sign structure of a representation, so floating-point dust does not
# misroute the scaling rule.
_SIGN_TOL = 1e-12


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

    The scaling follows the "Canonical noise scaling" prescription of Section
    VI D (Eqs. 38-44) of :cite:`Mari_2021_PRA`. That construction splits a
    quasi-probability representation into a positive and a negative volume and
    rescales them so that noise scaling is well defined for an arbitrary
    representation. The exact rule applied to each representation depends on
    its sign structure:

    * **All-positive** representations (a genuine probability distribution,
      e.g. an ``amplify_*`` decomposition or a noise model learned from
      hardware, with an explicit identity term) are scaled by deviation from
      the identity. See :func:`_scale_positive_representation`.
    * **Signed** representations (containing negative coefficients, e.g. a
      PEC-style ``represent_operation_with_*`` decomposition) are scaled by
      the canonical sign-partition rule of Eq. (43). See
      :func:`_scale_signed_representation`. Scaling these by deviation from the
      identity would amplify the error cancellation instead of the noise.

    Both rules preserve the unit sum of the coefficients.

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
        ValueError: If an all-positive representation has no identity term or a
            zero-weight identity term, or if a signed representation has no
            positive volume or ``scale_factor`` exceeds its canonical limit.
    """
    return [
        _scale_signed_representation(rep, scale_factor)
        if any(coeff < -_SIGN_TOL for coeff in rep.coeffs)
        else _scale_positive_representation(rep, scale_factor)
        for rep in representations
    ]


def _scale_positive_representation(
    representation: OperationRepresentation,
    scale_factor: float,
) -> OperationRepresentation:
    r"""Scales an all-positive representation by deviation from the identity.

    The representation is a quasi-probability decomposition of an ideal
    operation into the ideal operation itself (the "identity" term,
    coefficient :math:`a_0`) plus a set of error terms (coefficients
    :math:`a_k`). The deviation from the identity is amplified linearly by the
    scale factor :math:`s`:

    .. math::

        a_0 \rightarrow 1 - s\,(1 - a_0), \qquad a_k \rightarrow s\,a_k .

    This preserves the unit sum of the coefficients and, for the global
    depolarizing model (and any single-qubit depolarizing channel), reproduces
    the analytic rebuild at noise level :math:`s\,\epsilon` exactly at every
    scale factor. It is the :math:`\gamma^- = 0` special case of the canonical
    sign-partition rule (:func:`_scale_signed_representation`).

    Raises:
        ValueError: If the representation has no identity term (a noisy
            operation equal to its ideal operation) or that term has zero
            weight, so the deviation from the identity cannot be determined.

    .. note::
        For the two-qubit ``local_depolarizing`` model the per-qubit channels
        combine as a tensor product, so the identity coefficient depends
        quadratically on the noise level. Linear deviation scaling cannot
        reproduce that exactly; use ``global_depolarizing`` representations (or
        single-qubit operations) when exact equivalence with the legacy
        ``noise_model`` path is required.
    """
    ideal = representation.ideal
    identity_indices = [
        i
        for i, op in enumerate(representation.noisy_operations)
        if op.circuit == ideal
    ]
    identity_weight = sum(representation.coeffs[i] for i in identity_indices)
    if not identity_indices or identity_weight == 0:
        raise ValueError(
            "Cannot scale a representation without a non-zero identity "
            "term (a noisy operation equal to the ideal operation). The "
            "canonical noise scaling needs it to measure the deviation "
            "from the identity."
        )

    scaled_identity_weight = 1.0 - scale_factor * (1.0 - identity_weight)

    new_coeffs = []
    for i, coeff in enumerate(representation.coeffs):
        if i in identity_indices:
            # Split the scaled identity weight across the identity term(s)
            # in their original proportions (a single term in practice).
            new_coeffs.append(
                float(coeff / identity_weight * scaled_identity_weight)
            )
        else:
            new_coeffs.append(float(scale_factor * coeff))

    return OperationRepresentation(
        representation.ideal,
        representation.noisy_operations,
        new_coeffs,
        representation.is_qubit_dependent,
    )


def _scale_signed_representation(
    representation: OperationRepresentation,
    scale_factor: float,
) -> OperationRepresentation:
    r"""Scales a signed representation by canonical sign-partition scaling.

    This implements the canonical noise scaling of Section VI D (Eq. 43) of
    :cite:`Mari_2021_PRA`. The coefficients are split by sign into a positive
    volume :math:`\gamma^+` and a negative volume :math:`\gamma^-` (with
    :math:`\gamma^+ - \gamma^- = 1`), and each group is rescaled by

    .. math::

        a_i^+ \rightarrow a_i^+\,
            \frac{\gamma^+ - s\,\gamma^-}{\gamma^+}, \qquad
        a_i^- \rightarrow a_i^-\,(1 - s) .

    This preserves the unit sum. In the amplification band
    :math:`1 \le s \le \gamma^+/\gamma^-` the negative volume vanishes, so the
    representation becomes a genuine probability distribution (one-norm 1):
    amplifying it adds real noise rather than amplifying the error
    cancellation. The scaling is defined up to the canonical limit
    :math:`s \le \gamma^+/\gamma^-`, where the positive volume reaches zero.

    Raises:
        ValueError: If the representation has no positive volume, or if
            ``scale_factor`` exceeds the canonical limit
            :math:`\gamma^+/\gamma^-`.
    """
    coeffs = representation.coeffs
    pos_volume = sum(coeff for coeff in coeffs if coeff > _SIGN_TOL)
    neg_volume = -sum(coeff for coeff in coeffs if coeff < -_SIGN_TOL)

    if pos_volume <= 0:
        raise ValueError(
            "Cannot scale a signed representation without positive total "
            "mass (gamma+ > 0)."
        )
    if neg_volume > 0 and scale_factor > pos_volume / neg_volume:
        raise ValueError(
            f"scale_factor={scale_factor} exceeds the canonical limit "
            f"gamma+/gamma- = {pos_volume / neg_volume}, beyond which the "
            "noise-scaled representation is undefined."
        )

    pos_scaling = (pos_volume - scale_factor * neg_volume) / pos_volume
    neg_scaling = 1.0 - scale_factor

    new_coeffs = []
    for coeff in coeffs:
        if coeff > _SIGN_TOL:
            new_coeffs.append(float(coeff * pos_scaling))
        elif coeff < -_SIGN_TOL:
            new_coeffs.append(float(coeff * neg_scaling))
        else:
            new_coeffs.append(0.0)

    return OperationRepresentation(
        representation.ideal,
        representation.noisy_operations,
        new_coeffs,
        representation.is_qubit_dependent,
    )
