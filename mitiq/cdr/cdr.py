# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""API for using Clifford Data Regression (CDR) error mitigation."""

from functools import wraps
from typing import Any, Callable, cast, Optional, Sequence, Union

import numpy as np
from scipy.optimize import curve_fit

from mitiq import Executor, Observable, QPROGRAM, QuantumResult
from mitiq.interface import accept_any_qprogram_as_input
from mitiq.cdr import (
    generate_training_circuits,
    linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.zne.scaling import fold_gates_at_random


@wraps(accept_any_qprogram_as_input)
def execute_with_cdr(
    circuit: QPROGRAM,
    executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    observable: Optional[Observable] = None,
    *,
    simulator: Union[Executor, Callable[[QPROGRAM], QuantumResult]],
    num_training_circuits: int = 10,
    fraction_non_clifford: float = 0.1,
    fit_function: Callable[..., float] = linear_fit_function,
    num_fit_parameters: Optional[int] = None,
    scale_factors: Sequence[float] = (1,),
    scale_noise: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
    **kwargs: Any,
) -> float:
    """Function for the calculation of an observable from some circuit of
    interest to be mitigated with CDR (or vnCDR) based on [Czarnik2020]_ and
    [Lowe2020]_.

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
        observable: Observable to compute the expectation value of. If None,
            the `executor` must return an expectation value. Otherwise,
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
        scale_noise: scale_noise: Function for scaling the noise of a quantum
            circuit.
        scale_factors: Factors by which to scale the noise, should not
            include 1 as this is just the original circuit. Note: When
            scale_factors is provided, the method is known as "variable-noise
            Clifford data regression."
        kwargs: Available keyword arguments are:
            - method_select (string): Specifies the method used to select the
                non-Clifford gates to replace when constructing the
                near-Clifford training circuits. Can be 'uniform' or
                'gaussian'.
            - method_replace (string): Specifies the method used to replace
                the selected non-Clifford gates with a Clifford when
                constructing the near-Clifford training circuits. Can be
                'uniform', 'gaussian', or 'closest'.
            - sigma_select (float): Width of the Gaussian distribution used for
                ``method_select='gaussian'``.
            - sigma_replace (float): Width of the Gaussian distribution used
                for ``method_replace='gaussian'``.
            - random_state (int): Seed for sampling.

    .. [Czarnik2020] : Piotr Czarnik, Andrew Arramsmith, Patrick Coles,
        Lukasz Cincio, "Error mitigation with Clifford quantum circuit
        data," (https://arxiv.org/abs/2005.10189).
    .. [Lowe2020] : Angus Lowe, Max Hunter Gordon, Piotr Czarnik,
        Andrew Arramsmith, Patrick Coles, Lukasz Cincio,
        "Unified approach to data-driven error mitigation,"
        (https://arxiv.org/abs/2011.01157).
    """
    # Handle keyword arguments for generating training circuits.
    method_select = kwargs.get("method_select", "uniform")
    method_replace = kwargs.get("method_replace", "closest")
    random_state = kwargs.get("random_state", None)
    kwargs_for_training_set_generation = {
        "sigma_select": kwargs.get("sigma_select"),
        "sigma_replace": kwargs.get("sigma_replace"),
    }

    if num_fit_parameters is None and fit_function not in (
        linear_fit_function,
        linear_fit_function_no_intercept,
    ):
        raise ValueError(
            "Must provide arg `num_fit_parameters` for custom fit function."
        )
    num_fit_parameters = cast(int, num_fit_parameters)

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

    # Execute all circuits.
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    if not isinstance(simulator, Executor):
        simulator = Executor(simulator)

    to_run = [circuit for circuits in all_circuits for circuit in circuits]
    all_circuits_shape = (len(all_circuits), len(all_circuits[0]))

    results = executor.evaluate(to_run, observable)
    noisy_results = np.array(results).reshape(all_circuits_shape)

    results = simulator.evaluate(all_circuits[0], observable)
    ideal_results = np.array(results)

    # Do the regression.
    fitted_params, _ = curve_fit(
        lambda x, *params: fit_function(x, params),
        noisy_results,
        ideal_results,
        p0=np.zeros(num_fit_parameters),
    )
    return fit_function(noisy_results[:, 0], fitted_params)
