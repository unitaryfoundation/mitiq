# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level probabilistic error amplification tools."""

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import numpy as np
from cirq import Circuit

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.experimental.pea.scale_amplifications import (
    scale_circuit_amplifications,
)
from mitiq.pec.pec import (
    _LARGE_SAMPLE_WARN,
    LargeSampleWarning,
    sample_circuit,
)
from mitiq.pec.types import OperationRepresentation


def _canonical_scaled_representations(
    representations: Sequence[OperationRepresentation],
    scale_factor: float,
) -> list[OperationRepresentation]:
    r"""Returns canonically scaled representations for PEA.

    A quasi-probability representation can be split into positive and negative
    parts as ``G = gamma_plus * Phi_plus - gamma_minus * Phi_minus``. PEA's
    canonical noise scaling uses
    ``G(lambda) = (gamma_plus - lambda * gamma_minus) * Phi_plus
    - (1 - lambda) * gamma_minus * Phi_minus``.
    """
    scaled_representations = []
    for representation in representations:
        gamma_plus = sum(c for c in representation.coeffs if c > 0)
        gamma_minus = sum(-c for c in representation.coeffs if c < 0)

        if np.isclose(gamma_plus, 0.0):
            raise ValueError(
                "Cannot canonically scale a representation without positive "
                "coefficients."
            )

        if np.isclose(gamma_minus, 0.0):
            if not np.isclose(scale_factor, 1.0):
                raise ValueError(
                    "Cannot canonically scale an all-positive "
                    "representation away from scale factor 1."
                )
            scaled_coeffs = list(representation.coeffs)
        else:
            positive_scale = (
                gamma_plus - scale_factor * gamma_minus
            ) / gamma_plus
            negative_scale = 1.0 - scale_factor
            scaled_coeffs = [
                float(c * positive_scale if c > 0 else c * negative_scale)
                for c in representation.coeffs
            ]

        scaled_representations.append(
            OperationRepresentation(
                getattr(representation, "_native_ideal", representation.ideal),
                representation.noisy_operations,
                scaled_coeffs,
                representation.is_qubit_dependent,
            )
        )

    return scaled_representations


def _resolve_representations(
    circuit: Circuit,
    scale_factor: float,
    noise_model: str | None,
    epsilon: float | None,
    representations: Sequence[OperationRepresentation] | None,
) -> Sequence[OperationRepresentation]:
    if representations is not None and noise_model is not None:
        raise ValueError(
            "Exactly one of 'representations' or 'noise_model' should be "
            "provided."
        )
    if representations is None and noise_model is None:
        raise ValueError(
            "Either 'representations' or 'noise_model' must be provided."
        )
    if representations is not None:
        return _canonical_scaled_representations(representations, scale_factor)

    if epsilon is None:
        raise ValueError(
            "'epsilon' must be provided when using the legacy 'noise_model' "
            "argument."
        )
    assert noise_model is not None
    warnings.warn(
        "The 'noise_model' argument is deprecated. Pass "
        "'representations' directly instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return scale_circuit_amplifications(
        circuit, scale_factor, noise_model, epsilon
    )


def construct_circuits(
    circuit: Circuit,
    scale_factors: list[float],
    noise_model: str | None = None,
    epsilon: float | None = None,
    random_state: int | np.random.RandomState | None = None,
    precision: float = 0.1,
    num_samples: int | None = None,
    representations: Sequence[OperationRepresentation] | None = None,
) -> tuple[list[list[QPROGRAM]], list[list[int]], list[float]]:
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
        representations: Quasi-probability representations of the circuit
            operations to be canonically noise-scaled for PEA.
        noise_model: A legacy string describing the noise model to be used for
            noise-scaled representations, e.g. "local_depolarizing" or
            "global_depolarizing". Deprecated in favor of ``representations``.
        epsilon: Baseline noise level. Required only with ``noise_model``.
        random_state: The random state or seed for reproducibility.
        precision: The desired precision for the sampling process.
            Default is 0.1.
        num_samples: The number of noisy circuits to be sampled for PEA.
            If not given, this is deduced from the 'precision'.

    Returns:
        The scaled circuits, their signs and norms at each scale factor.
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

    scaled_representations = [
        _resolve_representations(
            circuit, s, noise_model, epsilon, representations
        )
        for s in scale_factors
    ]

    # Deduce the number of samples (if not given by the user)
    if num_samples is None:
        _, _, norm = sample_circuit(
            circuit,
            _resolve_representations(
                circuit, 1.0, noise_model, epsilon, representations
            ),
            num_samples=1,
        )
        for scaled_representation in scaled_representations:
            _, _, scaled_norm = sample_circuit(
                circuit, scaled_representation, num_samples=1
            )
            norm = max(norm, scaled_norm)
        num_samples = int((norm / precision) ** 2)

    if num_samples > 10**5:
        warnings.warn(_LARGE_SAMPLE_WARN, LargeSampleWarning)

    scaled_sampled_circuits = []
    scaled_signs = []
    scaled_norms = []
    for scaled_representation in scaled_representations:
        sampled_circuits, signs, norm = sample_circuit(
            circuit,
            scaled_representation,
            num_samples=num_samples,
            random_state=random_state,
        )
        scaled_sampled_circuits.append(sampled_circuits)
        scaled_signs.append(signs)
        scaled_norms.append(norm)

    return scaled_sampled_circuits, scaled_signs, scaled_norms


def combine_results(
    scale_factors: list[float],
    scaled_results: Iterable[Iterable[float]],
    scaled_norms: Iterable[float],
    scaled_signs: Iterable[Iterable[int]],
    extrapolation_method: Callable[[Sequence[float], Sequence[float]], float],
) -> float:
    """Combine expectation values coming from probabilistically sampled
    circuits at each of the input noise `scale_factors` and extrapolate
    the resulting expectation values to the zero noise limit to obtain the
    error-mitigated expectation value.

    Warning:
        The ``scaled_results`` must be in the same order as the circuits were
        generated.

    Args:
        scale_factors: A list of (positive) numbers by which the baseline
            noise level is to be amplified.
        scaled_results: Results as obtained from running circuits at each scale
            factor.
        scaled_norms: The one-norm of the circuit representations at each scale
            factor.
        scaled_signs: The signs corresponding to the positivity of the sampled
            circuits at each scale factor.
        extrapolation_method: The method of extrapolation to use when fitting
            the measured results. A list of built-in functions can be found
            in ``mitiq.zne.inference``.

    Returns:
        The PEA estimate of the expectation value.
    """
    pea_values = []
    for results, norm, signs in zip(
        scaled_results, scaled_norms, scaled_signs
    ):
        unbiased_estimators = [
            norm * s * val for s, val in zip(signs, results)
        ]
        pea_values.append(cast(float, np.average(unbiased_estimators)))
    pea_result = extrapolation_method(scale_factors, pea_values)
    return pea_result


def execute_with_pea(
    circuit: Circuit,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    scale_factors: list[float],
    noise_model: str | None = None,
    epsilon: float | None = None,
    extrapolation_method: Callable[[Sequence[float], Sequence[float]], float]
    | None = None,
    observable: Observable | None = None,
    random_state: int | np.random.RandomState | None = None,
    precision: float = 0.1,
    num_samples: int | None = None,
    full_output: bool = False,
    force_run_all: bool = False,
    representations: Sequence[OperationRepresentation] | None = None,
) -> float | tuple[float, dict[str, Any]]:
    r"""Estimates the error-mitigated expectation value associated to the
    input circuit, via the application of probabilistic error amplification
    (PEA). :cite:`Kim_2023_Nature`.

    This function implements PEA by:

    1. Sampling different implementable circuits from the quasi-probability
       representation of the input circuit at each of the input noise
       `scale_factors`;
    2. Evaluating the noisy expectation values associated to the sampled
       circuits (through the "executor" function provided by the user);
    3. Estimating the ideal expectation value from a suitable linear
       combination of the noisy ones at each noise scale factor;
    4. Extrapolating the expectation values obtained at each noise
       scale factor to the zero noise limit.

    Args:
        circuit: The input circuit to execute with error-mitigation.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        scale_factors: A list of (positive) numbers by which the baseline
            noise level is to be amplified.
        representations: Quasi-probability representations of the circuit
            operations to be canonically noise-scaled for PEA.
        noise_model: A legacy string describing the noise model to be used for
            noise-scaled representations, e.g. "local_depolarizing" or
            "global_depolarizing". Deprecated in favor of ``representations``.
        epsilon: Baseline noise level. Required only with ``noise_model``.
        extrapolation_method: The method of extrapolation to use when fitting
            the measured results. A list of built-in functions can be found
            in ``mitiq.zne.inference``.
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        random_state: The random state or seed for reproducibility.
        precision: The desired precision for the sampling process.
            Default is 0.1.
        num_samples: The number of noisy circuits to be sampled for PEA.
            If not given, this is deduced from the argument 'precision'.
        full_output: If False only the average PEA value is returned.
            If True a dictionary containing all PEA data is returned too.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.

    Returns:
        The tuple ``(pea_value, pea_data)`` where ``pea_value`` is the
        expectation value estimated with PEA and ``pea_data`` is a dictionary
        which contains all the raw data involved in the PEA process. If
        ``full_output`` is ``False``, only ``pea_value`` is
        returned.
    """
    if extrapolation_method is None:
        raise ValueError("'extrapolation_method' must be provided.")

    scaled_circuits, scaled_signs, scaled_norms = construct_circuits(
        circuit,
        scale_factors,
        noise_model=noise_model,
        epsilon=epsilon,
        random_state=random_state,
        precision=precision,
        num_samples=num_samples,
        representations=representations,
    )
    # Execute all sampled circuits
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    scaled_results = [
        executor.evaluate(sc, observable, force_run_all)
        for sc in scaled_circuits
    ]

    pea_value = combine_results(
        scale_factors,
        scaled_results,
        scaled_norms,
        scaled_signs,
        extrapolation_method,
    )

    if not full_output:
        return pea_value

    num_circuits = len(scaled_circuits[0])

    # Build dictionary with additional results and data
    pea_data: dict[str, Any] = {
        "num_samples": num_circuits,
        "precision": precision,
        "pea_value": pea_value,
        "scaled_expectation_values": scaled_results,
        "scaled_sampled_circuits": scaled_circuits,
    }

    return pea_value, pea_data
