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
    scale_factor: float,
    noise_model: str,
    epsilon: float,
) -> Sequence[OperationRepresentation]:
    """Returns a list of noise-amplified circuits, corresponding to each scale
    factor multiplied by the baseline noise level."""

    if noise_model == "local_depolarizing":
        amp_fn = amplify_noisy_ops_in_circuit_with_local_depolarizing_noise
        # TODO add other existing noise models from Mitiq
    elif noise_model == "global_depolarizing":
        amp_fn = amplify_noisy_ops_in_circuit_with_global_depolarizing_noise
    else:
        raise ValueError("Must specify supported noise model")
        # TODO allow use of custom noise model

    return amp_fn(ideal_circuit, (scale_factor - 1) * epsilon)


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
    """Returns a list of expectation values, evaluated at each noise scale
    factor times the baseline noise level."""

    if not isinstance(executor, Executor):
        executor = Executor(executor)

    precision = 0.1
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
