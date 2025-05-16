from typing import Callable, List, Optional, Sequence, Union, cast

import numpy as np
from cirq import Circuit

from mitiq import QPROGRAM, Executor, QuantumResult
from mitiq.observable.observable import Observable
from mitiq.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec import OperationRepresentation
from mitiq.pec.sampling import sample_circuit


def scale_circuit_amplifications(
    ideal_circuit: Circuit,
    scale_factors: float,
    noise_model: str,
    epsilon: float,
) -> Sequence[OperationRepresentation]:
    r"""Generates a list of implementable sequences from the noise-amplified
    representation of the input ideal circuit based on the input noise model
    and baseline noise level.

    Args:
        ideal_circuit: The ideal circuit from which an implementable
            sequence is sampled.
        scale_factors: A list of (positive) numbers by which the baseline
            noise level is to be amplified.
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
        raise ValueError("Must specify supported noise model")
        # TODO allow use of custom noise model

    return amp_fn(ideal_circuit, (scale_factors - 1) * epsilon)


def sample_circuit_amplifications(
    ideal_circuit: Circuit,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    scale_factors: List[float],
    noise_model: str,
    epsilon: float,
    observable: Observable | None = None,
    num_samples: Optional[int] = None,
    force_run_all: bool = True,
) -> List[float]:
    """Samples a list of implementable circuits from the noise-amplified
    representation of the input ideal circuit.
    Returns a list of expectation values, evaluated at each scaled noise level.

    Note that the ideal operation can be a sequence of operations (circuit),
    for instance U = V W, as long as a representation is known. Similarly, A
    and B can be sequences of operations (circuits) or just single operations.

    Args:
        ideal_circuit: The ideal circuit from which an implementable
            sequence is sampled.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        scale_factors: A list of (positive) numbers by which the baseline
            noise level is to be amplified.
        noise_model: A string describing the noise model to be used for the
            noise-scaled representations, e.g. "local_depolarizing" or
            "global_depolarizing".
        epsilon: Baseline noise level.
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        num_samples: The number of noisy circuits to be sampled for PEA.
            If not given, this is deduced from the 'precision'.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.

    Returns:
        A list of expectation values, evaluated at each noise scale
        factor times the baseline noise level.
    """

    if not isinstance(executor, Executor):
        executor = Executor(executor)

    precision = 0.1  # TODO make configurable?
    amp_values = []
    for s in scale_factors:
        scaled_amplification = scale_circuit_amplifications(
            ideal_circuit, s, noise_model, epsilon
        )
        if num_samples is None:
            amp_norms = [amp.norm for amp in scaled_amplification]
            num_samples = int(
                sum([(a_norm / precision) ** 2 for a_norm in amp_norms])
            )

        sampled_circuits, signs, norm = sample_circuit(
            ideal_circuit,
            scaled_amplification,
            num_samples=num_samples,
        )
        scaled_results = executor.evaluate(
            sampled_circuits, observable, force_run_all
        )

        # Evaluate unbiased estimators
        unbiased_estimators = [
            norm * s * val for s, val in zip(signs, scaled_results)
        ]

        amp_values.append(cast(float, np.average(unbiased_estimators)))

    return amp_values
