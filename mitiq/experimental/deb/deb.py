# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level debiasing and sharpening helpers."""

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from mitiq import QPROGRAM, MeasurementResult
from mitiq.executor.executor import Executor
from mitiq.experimental.deb.sharpening import BitstringDistribution, sharpen
from mitiq.experimental.deb.symmetrization import (
    _average_distributions,
    _construct_cirq_symmetrizations,
    _counts_to_distribution,
    _distribution_to_counts,
)
from mitiq.interface.conversions import convert_from_mitiq


def execute_with_debiasing(
    circuit: QPROGRAM,
    executor: Executor
    | Callable[[QPROGRAM], MeasurementResult | BitstringDistribution],
    num_variants: int,
    *,
    random_state: int | np.random.Generator | None = None,
    full_output: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    """Executes Pauli-symmetrized variants and averages distributions.

    Args:
        circuit: Input circuit supported by Mitiq.
        executor: Executor returning a ``MeasurementResult`` or a bitstring
            count/probability dictionary.
        num_variants: Number of symmetrized variants to execute.
        random_state: Optional NumPy seed or generator for reproducibility.
        full_output: If ``True``, include circuits and intermediate
            distributions in the return value.

    Returns:
        The averaged bitstring probability distribution. If ``full_output`` is
        ``True``, returns ``(distribution, metadata)``.
    """
    (
        averaged,
        data,
    ) = _execute_and_collect(
        circuit,
        executor,
        num_variants,
        random_state=random_state,
    )
    return (averaged, data) if full_output else averaged


def execute_with_debiasing_and_sharpening(
    circuit: QPROGRAM,
    executor: Executor
    | Callable[[QPROGRAM], MeasurementResult | BitstringDistribution],
    num_variants: int,
    *,
    random_state: int | np.random.Generator | None = None,
    full_output: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, object]]:
    """Executes debiasing and applies plurality-vote sharpening.

    Args:
        circuit: Input circuit supported by Mitiq.
        executor: Executor returning a ``MeasurementResult`` or a bitstring
            count/probability dictionary.
        num_variants: Number of symmetrized variants to execute.
        random_state: Optional NumPy seed or generator for reproducibility.
        full_output: If ``True``, include intermediate data in the return
            value.

    Returns:
        The sharpened bitstring probability distribution. If ``full_output`` is
        ``True``, returns ``(distribution, metadata)``.
    """
    _, data = _execute_and_collect(
        circuit,
        executor,
        num_variants,
        random_state=random_state,
    )
    sharpened = sharpen(
        cast(list[dict[str, float]], data["variant_distributions"])
    )
    data["sharpened_distribution"] = sharpened
    return (sharpened, data) if full_output else sharpened


def _execute_and_collect(
    circuit: QPROGRAM,
    executor: Executor
    | Callable[[QPROGRAM], MeasurementResult | BitstringDistribution],
    num_variants: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    """Executes symmetrized variants and returns averaged data."""
    symmetrizations, input_type = _construct_cirq_symmetrizations(
        circuit, num_variants, random_state=random_state
    )
    circuits = [
        convert_from_mitiq(symmetrization.circuit, input_type)
        for symmetrization in symmetrizations
    ]
    results = _run_executor(executor, circuits)

    variant_distributions = []
    for result in results:
        counts = _result_to_counts(result)
        variant_distributions.append(_counts_to_distribution(counts))

    averaged = _average_distributions(variant_distributions)
    data: dict[str, object] = {
        "circuits": circuits,
        "pauli_layers": [s.paulis for s in symmetrizations],
        "variant_distributions": variant_distributions,
    }
    return averaged, data


def _run_executor(
    executor: Executor
    | Callable[[QPROGRAM], MeasurementResult | BitstringDistribution],
    circuits: list[QPROGRAM],
) -> Sequence[MeasurementResult | BitstringDistribution]:
    """Runs an executor over all circuits."""
    if isinstance(executor, Executor):
        return cast(
            Sequence[MeasurementResult | BitstringDistribution],
            executor.run(circuits, force_run_all=True),
        )
    return [executor(circuit) for circuit in circuits]


def _result_to_counts(
    result: MeasurementResult | BitstringDistribution,
) -> dict[str, int | float]:
    """Converts supported executor results to a count dictionary."""
    if isinstance(result, MeasurementResult):
        return {key: value for key, value in result.get_counts().items()}

    if not result:
        raise ValueError("Executor returned an empty result.")

    if any(value < 0 for value in result.values()):
        raise ValueError("Executor result weights must be non-negative.")

    total = sum(float(value) for value in result.values())
    if total <= 0:
        raise ValueError(
            "Executor result weights must sum to a positive value."
        )

    if all(
        0.0 <= float(value) <= 1.0 for value in result.values()
    ) and np.isclose(
        total,
        1.0,
    ):
        rounded_counts = _distribution_to_counts(
            {key: float(value) for key, value in result.items()},
            shots=10_000,
        )
        return {key: value for key, value in rounded_counts.items()}

    return dict(result)
