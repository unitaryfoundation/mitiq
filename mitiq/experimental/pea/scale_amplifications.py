# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools for constructing the noise-amplified representations of ideal
operations.
"""

import copy
import math
import warnings
from collections.abc import Callable, Sequence

from cirq import Circuit

from mitiq.experimental.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec import OperationRepresentation
from mitiq.utils import _equal

# Coefficients with magnitude below this are treated as zero when classifying
# the sign structure of a representation, so floating-point dust does not
# misroute the scaling rule or perturb the positive/negative volumes.
_SIGN_TOL = 1e-12


def _validate_scale_factor(scale_factor: float) -> None:
    if not math.isfinite(scale_factor):
        raise ValueError("Scale factor must be finite.")

    if scale_factor <= 0:
        raise ValueError("The scale factor must be a positive number.")


def _make_scaled_representation_factory(
    ideal_circuit: Circuit,
    noise_model: str | None = None,
    epsilon: float | None = None,
    representations: Sequence[OperationRepresentation] | None = None,
) -> Callable[[float], Sequence[OperationRepresentation]]:
    """Returns a scale-factor-to-representations provider for PEA.

    The legacy ``noise_model`` path still rebuilds analytic depolarizing
    amplifications at the target noise level. The direct ``representations``
    path scales the provided operation representations.
    """

    if (noise_model is None) == (representations is None):
        raise ValueError(
            "Either ``noise_model`` and ``epsilon`` or ``representations`` "
            "must be provided."
        )

    if representations is not None:
        if len(representations) == 0:
            raise ValueError(
                "``representations`` must be a non-empty list of "
                "``OperationRepresentation`` objects."
            )

        def scaled_representations(
            scale_factor: float,
        ) -> Sequence[OperationRepresentation]:
            _validate_scale_factor(scale_factor)
            return [
                scale_representation(rep, scale_factor)
                for rep in representations
            ]

        return scaled_representations

    if epsilon is None:
        raise ValueError(
            "``epsilon`` must be provided when using the ``noise_model`` "
            "path."
        )

    if noise_model == "local_depolarizing":
        amp_fn = amplify_noisy_ops_in_circuit_with_local_depolarizing_noise
        # TODO add other existing noise models from Mitiq
    elif noise_model == "global_depolarizing":
        amp_fn = amplify_noisy_ops_in_circuit_with_global_depolarizing_noise
    else:
        raise ValueError("Noise model not supported")
        # TODO allow use of custom noise model

    def scaled_representations(
        scale_factor: float,
    ) -> Sequence[OperationRepresentation]:
        _validate_scale_factor(scale_factor)
        return amp_fn(ideal_circuit, scale_factor * epsilon)

    return scaled_representations


def scale_circuit_amplifications(
    ideal_circuit: Circuit,
    scale_factor: float,
    noise_model: str | None = None,
    epsilon: float | None = None,
    representations: Sequence[OperationRepresentation] | None = None,
) -> Sequence[OperationRepresentation]:
    r"""Generates a list of implementable sequences from the noise-amplified
    representation of the input ideal circuit based on the input noise model
    and baseline noise level.

    Args:
        ideal_circuit: The ideal circuit from which an implementable
            sequence is sampled.
        scale_factor: A (positive) number by which the baseline noise
            level is to be amplified.
        noise_model: A legacy string describing the noise model to be used for
            the noise-scaled representations, e.g. "local_depolarizing" or
            "global_depolarizing". The deprecated alternative to
            ``representations`` (the two are mutually exclusive).
        epsilon: Baseline noise level. Used only for the legacy ``noise_model``
            path; ignored when ``representations`` is provided.
        representations: Optional precomputed list of operation representations
            from which noise-scaled representations are derived.

    Returns:
        A list of noise-amplified circuits, corresponding to each scale
        factor multiplied by the baseline noise level."""

    return _make_scaled_representation_factory(
        ideal_circuit,
        noise_model=noise_model,
        epsilon=epsilon,
        representations=representations,
    )(scale_factor)


def _find_identity_term_index(
    representation: OperationRepresentation,
) -> int:
    identity_term_index = None
    for idx, (coeff, noisy_op) in enumerate(
        representation.basis_expansion
    ):
        if coeff > 0 and _equal(noisy_op.circuit, representation.ideal):
            identity_term_index = idx
            break

    if identity_term_index is None:
        raise ValueError(
            "Could not identify the identity (un-noisy) term in this "
            "all-positive representation."
        )

    return identity_term_index


def _scale_positive_representation(
    representation: OperationRepresentation,
    scale_factor: float,
) -> OperationRepresentation:
    """Scales an all-positive representation.

    This corresponds to a decomposition
    ``G = a_0 O_0 + a_1 O_1 + ...``
    where ``O_0`` is the implementation of the ideal gate.
    For such representations we use:

    .. math::
        a_0' = 1 - s(1-a_0)
        a_i' = s a_i, i > 0
    """

    coeffs = representation.coeffs
    identity_term_index = _find_identity_term_index(representation)

    identity_weight = coeffs[identity_term_index]
    if 1 - identity_weight > 0 and scale_factor > 1 / (1 - identity_weight):
        warnings.warn(
            f"scale_factor={scale_factor} drives the identity coefficient "
            f"negative (deviation-scaling limit {1 / (1 - identity_weight)}); "
            "the amplified representation is no longer a valid probability "
            "distribution.",
            stacklevel=2,
        )

    scaled_coeffs = copy.copy(coeffs)
    scaled_coeffs[identity_term_index] = 1 - scale_factor * (
        1 - scaled_coeffs[identity_term_index]
    )

    for idx, coeff in enumerate(coeffs):
        if idx == identity_term_index:
            continue
        scaled_coeffs[idx] = scale_factor * coeff

    return OperationRepresentation(
        representation.ideal,
        list(representation.noisy_operations),
        scaled_coeffs,
        representation.is_qubit_dependent,
    )


def _scale_signed_representation(
    representation: OperationRepresentation,
    scale_factor: float,
) -> OperationRepresentation:
    r"""Scales a signed representation using canonical noise scaling.

    This implements the canonical two-channel rescaling described in
    Section D ("Canonical noise scaling") of arXiv:2108.02237. The
    coefficients are split by sign into positive/negative volumes
    ``gamma_plus``/``gamma_minus`` (with ``gamma_plus - gamma_minus = 1``)
    and rescaled by

    .. math::
        \eta_+ \to \eta_+ (\gamma^+ - s\,\gamma^-)/\gamma^+, \qquad
        \eta_- \to \eta_- (1 - s),

    which is sign-free with one-norm 1 in the amplification band, up to the
    canonical limit ``s <= gamma_plus / gamma_minus``.
    """

    coeffs = representation.coeffs
    pos_volume = sum(coeff for coeff in coeffs if coeff > _SIGN_TOL)
    neg_volume = -sum(coeff for coeff in coeffs if coeff < -_SIGN_TOL)

    if pos_volume <= 0:
        raise ValueError("Canonical scaling requires a representation "
                         "with positive total mass.")
    if neg_volume > 0:
        max_scale = pos_volume / neg_volume
        if scale_factor > max_scale:
            raise ValueError(
                f"scale_factor={scale_factor} is above the canonical limit "
                f"{max_scale}."
            )

    pos_scaling = (pos_volume - scale_factor * neg_volume) / pos_volume
    neg_scaling = 1 - scale_factor

    scaled_coeffs = []
    for coeff in coeffs:
        if coeff > _SIGN_TOL:
            scaled_coeffs.append(coeff * pos_scaling)
        elif coeff < -_SIGN_TOL:
            scaled_coeffs.append(coeff * neg_scaling)
        else:
            scaled_coeffs.append(0.0)

    return OperationRepresentation(
        representation.ideal,
        list(representation.noisy_operations),
        scaled_coeffs,
        representation.is_qubit_dependent,
    )


def scale_representation(
    representation: OperationRepresentation,
    scale_factor: float,
) -> OperationRepresentation:
    """Scales a single representation for a given noise scale factor.

    The scaling rule is selected based on the sign structure of the input
    representation (see Section D of arXiv:2108.02237):

    * Signed representation (contains negative coefficients, e.g. a PEC-style
      ``represent_operation_with_*`` decomposition): canonical sign-partition
      noise scaling, valid up to the limit ``s <= gamma_plus / gamma_minus``.
    * All-positive representation (e.g. a learned noise model or an
      ``amplify_*`` decomposition with an explicit identity term):
      deviation-from-identity scaling ``a_0 -> 1 - s(1 - a_0)``,
      ``a_k -> s a_k``.

    Both rules preserve the unit sum of the coefficients. This function is
    designed to be used by ``scale_circuit_amplifications``.
    """

    _validate_scale_factor(scale_factor)

    if all(coeff >= -_SIGN_TOL for coeff in representation.coeffs):
        return _scale_positive_representation(representation, scale_factor)

    return _scale_signed_representation(representation, scale_factor)
