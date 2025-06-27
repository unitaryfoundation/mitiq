# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level zero-noise extrapolation tools."""

from collections.abc import Callable, Sequence
from functools import wraps

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.zne.inference import (
    Factory,
    RichardsonFactory,
)
from mitiq.zne.scaling import fold_gates_at_random


def construct_circuits(
    circuit: QPROGRAM,
    scale_factors: list[float],
    scale_method: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,  # type:ignore [has-type]
) -> list[QPROGRAM]:
    """Given a circuit, scale_factors and a scale_method, outputs a list
       of circuits that will be used in ZNE.

    Args:
        circuit: The input circuit to execute with ZNE.
        scale_factors: An array of noise scale factors.
        scale_method: The function for scaling the noise of a quantum circuit.
            A list of built-in functions can be found in ``mitiq.zne.scaling``.

    Returns:
        The scaled circuits using the scale_method.
    """
    circuits = []
    for scale_factor in scale_factors:
        circuits.append(scale_method(circuit, scale_factor))

    return circuits


def combine_results(
    scale_factors: Sequence[float],
    results: Sequence[float],
    extrapolation_method: Callable[[Sequence[float], Sequence[float]], float],
) -> float:
    """Computes the error-mitigated expectation value associated to the
    input results from executing the scaled circuits, via the application
    of zero-noise extrapolation (ZNE).

    Args:
        scale_factors: An array of noise scale factors.
        results: An array storing the results of running the scaled circuits.
        extrapolation_method: The function for scaling the noise of a
            quantum circuit. A list of built-in functions can be found
            in ``mitiq.zne.inference``.

    Returns:
        The expectation value estimated with ZNE.
    """
    res = extrapolation_method(scale_factors, results)

    return res


def execute_with_zne(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    factory: Factory | None = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,  # type: ignore [has-type]
    num_to_average: int = 1,
) -> float:
    """Estimates the error-mitigated expectation value associated to the
    input circuit, via the application of zero-noise extrapolation (ZNE).

    Args:
        circuit: The input circuit to execute with ZNE.
        executor: A Mitiq executor that executes a circuit and returns the
            unmitigated ``QuantumResult`` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If
            ``None``, the ``executor`` must return an expectation value.
            Otherwise, the ``QuantumResult`` returned by ``executor`` is used
            to compute the expectation of the observable.
        factory: ``Factory`` object that determines the zero-noise
            extrapolation method.
        scale_noise: The function for scaling the noise of a quantum circuit.
            A list of built-in functions can be found in ``mitiq.zne.scaling``.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to ``scale_noise``, then averaged.

    Returns:
        The expectation value estimated with ZNE.
    """
    if not factory:
        factory = RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])

    if not isinstance(factory, Factory):
        raise TypeError(
            f"Argument `factory` must be of type mitiq.factories.Factory "
            f"but type(factory) is {type(factory)}."
        )

    if not callable(scale_noise):
        raise TypeError("Argument `scale_noise` must be callable.")

    if num_to_average < 1:
        raise ValueError("Argument `num_to_average` must be a positive int.")

    return factory.run(
        circuit, executor, observable, scale_noise, int(num_to_average)
    ).reduce()


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    factory: Factory | None = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,  # type:ignore [has-type]
    num_to_average: int = 1,
) -> Callable[[QPROGRAM], float]:
    """Returns a modified version of the input 'executor' which is
    error-mitigated with zero-noise extrapolation (ZNE).

    Args:
        executor: A function that executes a circuit and returns the
            unmitigated `QuantumResult` (e.g. an expectation value).
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        factory: Factory object determining the zero-noise extrapolation
            method.
        scale_noise: Function for scaling the noise of a quantum circuit.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to scale_noise, then averaged.

    Returns:
        The error-mitigated version of the input executor.
    """
    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(circuit: QPROGRAM) -> float:
            return execute_with_zne(
                circuit,
                executor,
                observable,
                factory=factory,
                scale_noise=scale_noise,
                num_to_average=num_to_average,
            )

    else:

        @wraps(executor)
        def new_executor(circuits: list[QPROGRAM]) -> list[float]:
            return [
                execute_with_zne(
                    circuit,
                    executor,
                    observable,
                    factory=factory,
                    scale_noise=scale_noise,
                    num_to_average=num_to_average,
                )
                for circuit in circuits
            ]

    return new_executor


def zne_decorator(
    observable: Observable | None = None,
    *,
    factory: Factory | None = None,
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,  # type: ignore [has-type]
    num_to_average: int = 1,
) -> Callable[
    [Callable[[QPROGRAM], QuantumResult]], Callable[[QPROGRAM], float]
]:
    """Decorator which adds an error-mitigation layer based on zero-noise
    extrapolation (ZNE) to an executor function, i.e., a function which
    executes a quantum circuit with an arbitrary backend and returns a
    ``QuantumResult`` (e.g. an expectation value).

    Args:
        observable: Observable to compute the expectation value of. If None,
            the `executor` being decorated must return an expectation value.
            Otherwise, the `QuantumResult` returned by the `executor` is used
            to compute the expectation of the observable.
        factory: Factory object determining the zero-noise extrapolation
            method.
        scale_noise: Function for scaling the noise of a quantum circuit.
        num_to_average: Number of times expectation values are computed by
            the executor after each call to scale_noise, then averaged.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """
    # Raise an error if the decorator is used without parenthesis
    if callable(observable):
        raise TypeError(
            "Decorator must be used with parentheses (i.e., @zne_decorator()) "
            "even if no explicit arguments are passed."
        )

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult],
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(
            executor,
            observable,
            factory=factory,
            scale_noise=scale_noise,
            num_to_average=num_to_average,
        )

    return decorator
