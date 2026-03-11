# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""API for using Clifford Data Regression (CDR) error mitigation."""

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.cdr import (
    generate_training_circuits,
    linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.cdr.clifford_utils import is_clifford
from mitiq.zne.scaling import fold_gates_at_random


def execute_with_cdr(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    simulator: Executor | Callable[[QPROGRAM], QuantumResult],
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.1,
    fit_function: Callable[..., float] = linear_fit_function,
    num_fit_parameters: int | None = None,
    scale_factors: Sequence[float] = (1,),
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,  # type: ignore
    **kwargs: Any,
) -> float:
    """Function for the calculation of an observable from some circuit of
    interest to be mitigated with CDR (or vnCDR) based on
    Ref. :cite:`Czarnik_2021_Quantum` and Ref. :cite:`Lowe_2021_PRR`.

    The circuit of interest must be compiled in the native basis of the IBM
    quantum computers, that is {Rz, sqrt(X), CNOT}, or such that all the
    non-Clifford gates are contained in the Rz rotations.

    The observable/s to be calculated should be input as an array or a list of
    arrays representing the diagonal of the observables to be measured. Note
    these observables MUST be diagonal in z-basis measurements corresponding to
    the circuit of interest.

    Returns mitigated observables list of raw observables (at noise scale
    factors).

    This function returns the mitigated observable/s.

    Args:
        circuit: Quantum program to execute with error mitigation.
        executor: Executes a circuit and returns a `QuantumResult`.
        observable: Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        simulator: Executes a circuit without noise and returns a
            `QuantumResult`. For CDR to be efficient, the simulator must
            be able to efficiently simulate near-Clifford circuits.
        num_training_circuits: Number of training circuits to be used in the
            mitigation.
        fraction_non_clifford: The fraction of non-Clifford gates to be
            substituted in the training circuits.
        fit_function: The function to map noisy to exact data. Takes array of
            noisy and data and parameters returning a float. See
            ``cdr.linear_fit_function`` for an example.
        num_fit_parameters: The number of parameters the fit_function takes.
        scale_noise: Function for scaling the noise of a quantum circuit.
        scale_factors: Factors by which to scale the noise.

            - When 1.0 is the only scale factor, the method is known as CDR.

            - Note: When scale factors larger than 1.0 are provided, the method
              is known as "variable-noise CDR."

        kwargs: Available keyword arguments are:

            - method_select (string): Specifies the method used to select the
              non-Clifford gates to replace when constructing the near-Clifford
              training circuits. Can be 'uniform' or 'gaussian'.

            - method_replace (string): Specifies the method used to replace the
              selected non-Clifford gates with a Clifford when constructing the
              near-Clifford training circuits. Can be 'uniform', 'gaussian', or
              'closest'.

            - sigma_select (float): Width of the Gaussian distribution used for
              ``method_select='gaussian'``.

            - sigma_replace (float): Width of the Gaussian distribution used
              for ``method_replace='gaussian'``.

            - random_state (int): Seed for sampling.
    """
    # Handle keyword arguments for generating training circuits.

    method_select = kwargs.get("method_select", "uniform")
    method_replace = kwargs.get("method_replace", "closest")
    random_state = kwargs.get("random_state", None)
    kwargs_for_training_set_generation = {
        "sigma_select": kwargs.get("sigma_select"),
        "sigma_replace": kwargs.get("sigma_replace"),
    }

    if num_fit_parameters is None:
        if fit_function is linear_fit_function:
            num_fit_parameters = 1 + len(scale_factors)
        elif fit_function is linear_fit_function_no_intercept:
            num_fit_parameters = len(scale_factors)
        else:
            raise ValueError(
                "Must provide `num_fit_parameters` for custom fit function."
            )

    # cast executor and simulator inputs to Executor type
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    if not isinstance(simulator, Executor):
        simulator = Executor(simulator)

    # Check if circuit is already Clifford
    if is_clifford(circuit):
        return simulator.evaluate(circuit, observable)[0].real

    # Generate training circuits.
    training_circuits = generate_training_circuits(
        circuit,
        num_training_circuits,
        fraction_non_clifford,
        method_select,
        method_replace,
        random_state,
        kwargs=kwargs_for_training_set_generation,
    )

    # [Optionally] Scale noise in circuits.
    all_circuits = [
        [scale_noise(c, s) for s in scale_factors]
        for c in [circuit] + training_circuits  # type: ignore
    ]

    to_run = [circuit for circuits in all_circuits for circuit in circuits]
    all_circuits_shape = (len(all_circuits), len(all_circuits[0]))

    results = executor.evaluate(to_run, observable)
    noisy_results = np.array(results).reshape(all_circuits_shape)

    results = simulator.evaluate(training_circuits, observable)
    ideal_results = np.array(results)

    # Do the regression.
    fitted_params, _ = curve_fit(
        lambda x, *params: fit_function(x, params),
        noisy_results[1:, :].T,
        ideal_results,
        p0=np.zeros(num_fit_parameters),
    )
    return fit_function(noisy_results[0, :], fitted_params)


def mitigate_executor(
    executor: Callable[[QPROGRAM], QuantumResult],
    observable: Observable | None = None,
    *,
    simulator: Executor | Callable[[QPROGRAM], QuantumResult],
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.1,
    fit_function: Callable[..., float] = linear_fit_function,
    num_fit_parameters: int | None = None,
    scale_factors: Sequence[float] = (1,),
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,  # type: ignore
    **kwargs: Any,
) -> Callable[[QPROGRAM], float]:
    """Returns a clifford data regression (CDR) mitigated version of
    the input 'executor'.

    The input `executor` executes a circuit with an arbitrary backend and
    produces an expectation value (without any error mitigation). The returned
    executor executes the circuit with the same backend but uses clifford
    data regression to produce the CDR estimate of the ideal expectation
    value associated to the input circuit.

    Args:
        executor: Executes a circuit and returns a `QuantumResult`.
        observable: Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        simulator: Executes a circuit without noise and returns a
            `QuantumResult`. For CDR to be efficient, the simulator must
            be able to efficiently simulate near-Clifford circuits.
        num_training_circuits: Number of training circuits to be used in the
            mitigation.
        fraction_non_clifford: The fraction of non-Clifford gates to be
            substituted in the training circuits.
        fit_function: The function to map noisy to exact data. Takes array of
            noisy and data and parameters returning a float. See
            ``cdr.linear_fit_function`` for an example.
        num_fit_parameters: The number of parameters the fit_function takes.
        scale_noise: Function for scaling the noise of a quantum circuit.
        scale_factors: Factors by which to scale the noise.

            - When 1.0 is the only scale factor, the method is known as CDR.

            - Note: When scale factors larger than 1.0 are provided, the method
              is known as "variable-noise CDR."

        kwargs: Available keyword arguments are:

            - method_select (string): Specifies the method used to select the
              non-Clifford gates to replace when constructing the near-Clifford
              training circuits. Can be 'uniform' or 'gaussian'.

            - method_replace (string): Specifies the method used to replace
              the selected non-Clifford gates with a Clifford when constructing
              the near-Clifford training circuits. Can be 'uniform', 'gaussian'
              , or 'closest'.

            - sigma_select (float): Width of the Gaussian distribution used for
              ``method_select='gaussian'``.

            - sigma_replace (float): Width of the Gaussian distribution used
              for ``method_replace='gaussian'``.

            - random_state (int): Seed for sampling."""
    executor_obj = Executor(executor)
    if not executor_obj.can_batch:

        @wraps(executor)
        def new_executor(
            circuit: QPROGRAM,
        ) -> float:
            return execute_with_cdr(
                circuit,
                executor,
                observable,
                simulator=simulator,
                num_training_circuits=num_training_circuits,
                fraction_non_clifford=fraction_non_clifford,
                fit_function=fit_function,
                num_fit_parameters=num_fit_parameters,
                scale_factors=scale_factors,
                scale_noise=scale_noise,
                **kwargs,
            )

    else:

        @wraps(executor)
        def new_executor(
            circuits: list[QPROGRAM],
        ) -> list[float]:
            return [
                execute_with_cdr(
                    circuit,
                    executor,
                    observable,
                    simulator=simulator,
                    num_training_circuits=num_training_circuits,
                    fraction_non_clifford=fraction_non_clifford,
                    fit_function=fit_function,
                    num_fit_parameters=num_fit_parameters,
                    scale_factors=scale_factors,
                    scale_noise=scale_noise,
                    **kwargs,
                )
                for circuit in circuits
            ]

    return new_executor


def cdr_decorator(
    observable: Observable | None = None,
    *,
    simulator: Executor | Callable[[QPROGRAM], QuantumResult],
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.1,
    fit_function: Callable[..., float] = linear_fit_function,
    num_fit_parameters: int | None = None,
    scale_factors: Sequence[float] = (1,),
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,  # type: ignore
    **kwargs: Any,
) -> Callable[
    [Callable[[QPROGRAM | Any], QuantumResult]],
    Callable[
        [QPROGRAM | Any],
        float,
    ],
]:
    """Decorator which adds clifford data regression (CDR) mitigation
    to an executor function, i.e., a function which executes a quantum circuit
    with an arbitrary backend and returns the CDR estimate of the ideal
    expectation value associated to the input circuit.

    Args:
        executor: Executes a circuit and returns a `QuantumResult`.
        observable: Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.
        simulator: Executes a circuit without noise and returns a
            `QuantumResult`. For CDR to be efficient, the simulator must
            be able to efficiently simulate near-Clifford circuits.
        num_training_circuits: Number of training circuits to be used in the
            mitigation.
        fraction_non_clifford: The fraction of non-Clifford gates to be
            substituted in the training circuits.
        fit_function: The function to map noisy to exact data. Takes array of
            noisy and data and parameters returning a float. See
            ``cdr.linear_fit_function`` for an example.
        num_fit_parameters: The number of parameters the fit_function takes.
        scale_noise: Function for scaling the noise of a quantum circuit.
        scale_factors: Factors by which to scale the noise.

            - When 1.0 is the only scale factor, the method is known as CDR.

            - Note: When scale factors larger than 1.0 are provided, the method
              is known as "variable-noise CDR."

        kwargs: Available keyword arguments are:

            - method_select (string): Specifies the method used to select the
              non-Clifford gates to replace when constructing the near-Clifford
              training circuits. Can be 'uniform' or 'gaussian'.

            - method_replace (string): Specifies the method used to replace the
              selected non-Clifford gates with a Clifford when constructing the
              near-Clifford training circuits. Can be 'uniform', 'gaussian', or
              'closest'.

            - sigma_select (float): Width of the Gaussian distribution used for
              ``method_select='gaussian'``.

            - sigma_replace (float): Width of the Gaussian distribution used
              for ``method_replace='gaussian'``.

            - random_state (int): Seed for sampling.
    """

    def decorator(
        executor: Callable[[QPROGRAM], QuantumResult],
    ) -> Callable[[QPROGRAM], float]:
        return mitigate_executor(
            executor,
            observable,
            simulator=simulator,
            num_training_circuits=num_training_circuits,
            fraction_non_clifford=fraction_non_clifford,
            fit_function=fit_function,
            num_fit_parameters=num_fit_parameters,
            scale_factors=scale_factors,
            scale_noise=scale_noise,
            **kwargs,
        )

    return decorator
