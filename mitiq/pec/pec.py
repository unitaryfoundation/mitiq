# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level probabilistic error cancellation tools."""

import warnings
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Any, cast

import numpy as np

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.pec import OperationRepresentation, sample_circuit


class LargeSampleWarning(Warning):
    """Warning is raised when PEC sample size is greater than 10 ** 5"""

    pass


_LARGE_SAMPLE_WARN = (
    "The number of PEC samples is very large. It may take several minutes."
    " It may be necessary to reduce 'precision' or 'num_samples'."
)


def construct_circuits(
    circuit: QPROGRAM,
    representations: Sequence[OperationRepresentation],
    precision: float = 0.03,
    num_samples: int | None = None,
    random_state: int | np.random.RandomState | None = None,
    full_output: bool = False,
) -> list[QPROGRAM] | tuple[list[QPROGRAM], list[int], float]:
    """Generates a list of sampled circuits based on the given
    quasi-probability representations.

    Args:
        circuit: The quantum circuit to be sampled.
        representations: The quasi-probability representations of the circuit
            operations.
        precision: The desired precision for the sampling process.
            Default is 0.03.
        num_samples: The number of samples to generate. If None, the number of
            samples is deduced based on the precision. Default is None.
        random_state: The random state or seed for reproducibility.
        full_output: If True, returns the signs and the norm along with the
            sampled circuits. Default is False.

    Returns:
        A list of sampled circuits. If ``full_output`` is True, also returns a
        list of signs, the norm.

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
        representations,
        num_samples=1,
    )

    # Deduce the number of samples (if not given by the user)
    if num_samples is None:
        num_samples = int((norm / precision) ** 2)

    if num_samples > 10**5:
        warnings.warn(_LARGE_SAMPLE_WARN, LargeSampleWarning)

    sampled_circuits, signs, _ = sample_circuit(
        circuit,
        representations,
        random_state=random_state,
        num_samples=num_samples,
    )

    if full_output:
        return sampled_circuits, signs, norm
    return sampled_circuits


def combine_results(
    results: Iterable[float], norm: float, signs: Iterable[int]
) -> float:
    """Combine expectation values coming from probabilistically sampled
    circuits.

    Warning:
        The ``results`` must be in the same order as the circuits were
        generated.

    Args:
        results: Results as obtained from running circuits.
        norm: The one-norm of the circuit representation.
        signs: The signs corresponding to the positivity of the sampled
            circuits.

    Returns:
        The PEC estimate of the expectation value.
    """
    unbiased_estimators = [norm * s * val for s, val in zip(signs, results)]

    pec_value = cast(float, np.average(unbiased_estimators))
    return pec_value


def execute_with_pec(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    representations: Sequence[OperationRepresentation],
    precision: float = 0.03,
    num_samples: int | None = None,
    force_run_all: bool = True,
    random_state: int | np.random.RandomState | None = None,
    full_output: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    r"""Estimates the error-mitigated expectation value associated to the
    input circuit, via the application of probabilistic error cancellation
    (PEC). :cite:`Temme_2017_PRL` :cite:`Endo_2018_PRX`.

    This function implements PEC by:

    1. Sampling different implementable circuits from the quasi-probability
       representation of the input circuit;
    2. Evaluating the noisy expectation values associated to the sampled
       circuits (through the "executor" function provided by the user);
    3. Estimating the ideal expectation value from a suitable linear
       combination of the noisy ones.

    Args:
        circuit: The input circuit to execute with error-mitigation.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        representations: Representations (basis expansions) of each operation
            in the input circuit.
        precision: The desired estimation precision (assuming the observable
            is bounded by 1). The number of samples is deduced according
            to the formula (one_norm / precision) ** 2, where 'one_norm'
            is related to the negativity of the quasi-probability
            representation :cite:`Temme_2017_PRL`. If 'num_samples' is
            explicitly set by the user, 'precision' is ignored and has no
            effect.
        num_samples: The number of noisy circuits to be sampled for PEC.
            If not given, this is deduced from the argument 'precision'.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.
        random_state: Seed for sampling circuits.
        full_output: If False only the average PEC value is returned.
            If True a dictionary containing all PEC data is returned too.

    Returns:
        The tuple ``(pec_value, pec_data)`` where ``pec_value`` is the
        expectation value estimated with PEC and ``pec_data`` is a dictionary
        which contains all the raw data involved in the PEC process (including
        the PEC estimation error).
        The error is estimated as ``pec_std / sqrt(num_samples)``, where
        ``pec_std`` is the standard deviation of the PEC samples, i.e., the
        square root of the mean squared deviation of the sampled values from
        ``pec_value``. If ``full_output`` is ``False``, only ``pec_value`` is
        returned.
    """
    sampled_circuits, signs, norm = construct_circuits(
        circuit,
        representations,
        precision,
        num_samples,
        random_state=random_state,
        full_output=True,
    )

    # Execute all sampled circuits
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    results = executor.evaluate(sampled_circuits, observable, force_run_all)

    # Evaluate unbiased estimators [Temme2017] [Endo2018] [Takagi2020]
    unbiased_estimators = [norm * s * val for s, val in zip(signs, results)]

    pec_value = cast(float, np.average(unbiased_estimators))

    if not full_output:
        return pec_value

    num_circuits = len(sampled_circuits)
    # Build dictionary with additional results and data
    pec_data: dict[str, Any] = {
        "num_samples": num_circuits,
        "precision": precision,
        "pec_value": pec_value,
        "pec_error": np.std(unbiased_estimators) / np.sqrt(num_circuits),
        "unbiased_estimators": unbiased_estimators,
        "measured_expectation_values": results,
        "sampled_circuits": sampled_circuits,
    }

    return pec_value, pec_data


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    representations: Sequence[OperationRepresentation],
    precision: float = 0.03,
    num_samples: int | None = None,
    force_run_all: bool = True,
    random_state: int | np.random.RandomState | None = None,
    full_output: bool = False,
) -> Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with probabilistic error cancellation (PEC).

    Args:
        executor: A function that executes a circuit and returns the
            unmitigated `QuantumResult` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        representations: Representations (basis expansions) of each operation
            in the input circuit.
        precision: The desired estimation precision (assuming the observable
            is bounded by 1). The number of samples is deduced according
            to the formula (one_norm / precision) ** 2, where 'one_norm'
            is related to the negativity of the quasi-probability
            representation :cite:`Temme_2017_PRL`. If 'num_samples' is
            explicitly set, 'precision' is ignored and has no effect.
        num_samples: The number of noisy circuits to be sampled for PEC.
            If not given, this is deduced from the argument 'precision'.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.
        random_state: Seed for sampling circuits.
        full_output: If False only the average PEC value is returned.
            If True a dictionary containing all PEC data is returned too.

    Returns:
        The error-mitigated version of the input executor.
    """
    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(
            circuit: QPROGRAM,
        ) -> float | tuple[float, dict[str, Any]]:
            return execute_with_pec(
                circuit,
                executor,
                observable,
                representations=representations,
                precision=precision,
                num_samples=num_samples,
                force_run_all=force_run_all,
                random_state=random_state,
                full_output=full_output,
            )

    else:

        @wraps(executor)
        def new_executor(
            circuits: list[QPROGRAM],
        ) -> list[float | tuple[float, dict[str, Any]]]:
            return [
                execute_with_pec(
                    circuit,
                    executor,
                    observable,
                    representations=representations,
                    precision=precision,
                    num_samples=num_samples,
                    force_run_all=force_run_all,
                    random_state=random_state,
                    full_output=full_output,
                )
                for circuit in circuits
            ]

    return new_executor


def pec_decorator(
    observable: Observable | None = None,
    *,
    representations: Sequence[OperationRepresentation],
    precision: float = 0.03,
    num_samples: int | None = None,
    force_run_all: bool = True,
    random_state: int | np.random.RandomState | None = None,
    full_output: bool = False,
) -> Callable[
    [Callable[[QPROGRAM], QuantumResult]],
    Callable[
        [QPROGRAM],
        float | tuple[float, dict[str, Any]],
    ],
]:
    """Decorator which adds an error-mitigation layer based on probabilistic
    error cancellation (PEC) to an executor function, i.e., a function which
    executes a quantum circuit with an arbitrary backend and returns a
    ``QuantumResult`` (e.g. an expectation value).


    Args:
        observable: Observable to compute the expectation value of. If None,
            the `executor` function being decorated must return an expectation
            value. Otherwise, the `QuantumResult` returned by this `executor`
            is used to compute the expectation of the observable.
        representations: Representations (basis expansions) of each operation
            in the input circuit.
        precision: The desired estimation precision (assuming the observable
            is bounded by 1). The number of samples is deduced according
            to the formula (one_norm / precision) ** 2, where 'one_norm'
            is related to the negativity of the quasi-probability
            representation :cite:`Temme_2017_PRL`. If 'num_samples' is
            explicitly set by the user, 'precision' is ignored and has no
            effect.
        num_samples: The number of noisy circuits to be sampled for PEC.
            If not given, this is deduced from the argument 'precision'.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.
        random_state: Seed for sampling circuits.
        full_output: If False only the average PEC value is returned.
            If True a dictionary containing all PEC data is returned too.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult],
    ) -> Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]]:
        return mitigate_executor(
            executor,
            observable,
            representations=representations,
            precision=precision,
            num_samples=num_samples,
            force_run_all=force_run_all,
            random_state=random_state,
            full_output=full_output,
        )

    return decorator
