# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Functions to calculate parameters for depolarizing noise and biased noise
models via a learning-based technique."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from mitiq import QPROGRAM, Executor, Observable
from mitiq.cdr import generate_training_circuits
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)
from mitiq.pec.representations.depolarizing import (
    represent_operation_with_local_depolarizing_noise,
)


def learn_biased_noise_parameters(
    operations_to_learn: list[QPROGRAM],
    circuit: QPROGRAM,
    ideal_executor: Executor,
    noisy_executor: Executor,
    pec_kwargs: dict[str, Any] | None = None,
    num_training_circuits: int = 5,
    fraction_non_clifford: float = 0.2,
    training_random_state: np.random.RandomState | None = None,
    epsilon0: float = 0.05,
    eta0: float = 1,
    observable: Observable | None = None,
    **learning_kwargs: dict["str", Any],
) -> tuple[bool, float, float]:
    r"""This function learns the biased noise parameters epsilon and eta
    associated to a set of input operations. The learning process is based on
    the execution of a set of training circuits on a noisy backend and on a
    classical simulator. The training circuits are near-Clifford approximations
    of the input circuit. A biased noise model characterization is assumed.

    Args:
        operations_to_learn: The ideal operations to learn the noise model of.
        circuit: The full quantum program as defined by the user.
        ideal_executor: Simulates a circuit and returns
            a noiseless ``QuantumResult``.
        noisy_executor: Executes a circuit on a noisy backend
            and returns a ``QuantumResult``.
        pec_kwargs (optional): Options to pass to ``execute_w_pec`` for the
            error-mitigated expectation value obtained from executing
            the training circuits.
        num_training_circuits: Number of near-Clifford circuits to be
            generated for training.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            gates in each training circuit.
        training_random_state: Seed for sampling the training circuits.
        epsilon0: Initial guess for noise strength.
        eta0: Initial guess for noise bias.
        observable (optional): Observable to compute the expectation value of.
            If None, the ``executor`` must return an expectation value.
            Otherwise the `QuantumResult` returned by `executor` is used to
            compute the expectation of the observable.
        learning_kwargs (optional): Additional data and options including
            ``method`` an optimization method supported by
            ``scipy.optimize.minimize`` and settings for the chosen
            optimization method.

    Returns:
        A 3-tuple containing a flag indicating whether or not the optimizer
        exited successfully, the optimized noise strength epsilon, and the
        optimized noise bias, eta.

    .. note:: Using this function may require some tuning. One of the main
        challenges is setting a good value of ``num_samples`` in the PEC
        options ``pec_kwargs``. Setting a small value of ``num_samples`` is
        typically necessary to obtain a reasonable execution time. On the other
        hand, using a number of PEC samples that is too small can result in a
        large statistical error, ultimately causing the optimization process to
        fail.
    """
    if pec_kwargs is None:
        pec_kwargs = {}

    training_circuits = generate_training_circuits(
        circuit=circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
        random_state=training_random_state,
    )

    ideal_values = np.array(
        ideal_executor.evaluate(training_circuits, observable)
    )

    pec_data, method, minimize_kwargs = _parse_learning_kwargs(
        learning_kwargs=learning_kwargs
    )

    result = minimize(
        biased_noise_loss_function,
        [epsilon0, eta0],
        args=(
            operations_to_learn,
            training_circuits,
            ideal_values,
            noisy_executor,
            pec_kwargs,
            pec_data,
            observable,
        ),
        method=method,
        **minimize_kwargs,
    )

    success = result.success
    epsilon_opt, eta_opt = result.x

    return success, epsilon_opt, eta_opt


def learn_depolarizing_noise_parameter(
    operations_to_learn: list[QPROGRAM],
    circuit: QPROGRAM,
    ideal_executor: Executor,
    noisy_executor: Executor,
    pec_kwargs: dict[str, Any] | None = None,
    num_training_circuits: int = 5,
    fraction_non_clifford: float = 0.2,
    training_random_state: np.random.RandomState | None = None,
    epsilon0: float = 0.05,
    observable: Observable | None = None,
    **learning_kwargs: dict["str", Any],
) -> tuple[bool, float]:
    r"""This function learns the depolarizing noise parameter (epsilon)
    associated to a set of input operations. The learning process is based on
    the execution of a set of training circuits on a noisy backend and on a
    classical simulator. The training circuits are near-Clifford approximations
    of the input circuit. A depolarizing noise model characterization is
    assumed.

    Args:
        operations_to_learn: The ideal operations to learn the noise model of.
        circuit: The full quantum program as defined by the user.
        ideal_executor: Simulates a circuit and returns
            a noiseless ``QuantumResult``.
        noisy_executor: Executes a circuit on a noisy backend
            and returns a ``QuantumResult``.
        pec_kwargs (optional): Options to pass to ``execute_w_pec`` for the
            error-mitigated expectation value obtained from executing
            the training circuits.
        num_training_circuits: Number of near-Clifford circuits to be
            generated for training.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            gates in each training circuit.
        training_random_state: Seed for sampling the training circuits.
        epsilon0: Initial guess for noise strength.
        observable (optional): Observable to compute the expectation value of.
            If None, the ``executor`` must return an expectation value.
            Otherwise the `QuantumResult` returned by `executor` is used to
            compute the expectation of the observable.
        learning_kwargs (optional): Additional data and options including
            ``method`` an optimization method supported by
            ``scipy.optimize.minimize`` and settings for the chosen
            optimization method.

    Returns:
        A 2-tuple containing flag indicating whether or not the optimizer
        exited successfully and the optimized noise strength epsilon.

    .. note:: Using this function may require some tuning. One of the main
        challenges is setting a good value of ``num_samples`` in the PEC
        options ``pec_kwargs``. Setting a small value of ``num_samples`` is
        typically necessary to obtain a reasonable execution time. On the other
        hand, using a number of PEC samples that is too small can result in a
        large statistical error, ultimately causing the optimization process to
        fail.
    """
    if pec_kwargs is None:
        pec_kwargs = {}

    training_circuits = generate_training_circuits(
        circuit=circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=fraction_non_clifford,
        random_state=training_random_state,
    )

    ideal_values = np.array(
        ideal_executor.evaluate(training_circuits, observable)
    )

    pec_data, method, minimize_kwargs = _parse_learning_kwargs(
        learning_kwargs=learning_kwargs
    )

    result = minimize(
        depolarizing_noise_loss_function,
        epsilon0,
        args=(
            operations_to_learn,
            training_circuits,
            ideal_values,
            noisy_executor,
            pec_kwargs,
            pec_data,
            observable,
        ),
        method=method,
        **minimize_kwargs,
    )

    success = result.success
    epsilon_opt = result.x[0]

    return success, epsilon_opt


def depolarizing_noise_loss_function(
    epsilon: npt.NDArray[np.float64],
    operations_to_mitigate: list[QPROGRAM],
    training_circuits: list[QPROGRAM],
    ideal_values: npt.NDArray[np.float64],
    noisy_executor: Executor,
    pec_kwargs: dict[str, Any],
    pec_data: npt.NDArray[np.float64] | None = None,
    observable: Observable | None = None,
) -> float:
    r"""Loss function for optimizing quasi-probability representations
    assuming a depolarizing noise model depending on one real parameter.

    Args:
        epsilon: Array of optimization parameters epsilon
            (local noise strength) and eta (noise bias between reduced
            dephasing and depolarizing channels).
        operations_to_mitigate: List of ideal operations to be represented by
            a combination of noisy operations.
        training_circuits: List of training circuits for generating the
            error-mitigated expectation values.
        ideal_values: Expectation values obtained by noiseless simulations.
        noisy_executor: Executes the circuit with noise and returns a
            `QuantumResult`.
        pec_kwargs: Options to pass to `execute_w_pec` for the error-mitigated
            expectation value obtained from executing the training circuits.
        pec_data (optional): 2-D array of error-mitigated expection values for
            model training.
        observable (optional): Observable to compute the expectation value of.
            If None, the ``executor`` must return an expectation value.
            Otherwise the ``QuantumResult`` returned by ``executor`` is used to
            compute the expectation of the observable.

    Returns: Mean squared error between the error-mitigated values and
        the ideal values, over the training set.
    """
    if pec_data is not None:
        ind = np.abs(pec_data[:, 0] - epsilon).argmin()
        mitigated_values = pec_data[ind, 1:]

    else:
        representations = [
            represent_operation_with_local_depolarizing_noise(
                operation,
                epsilon[0],
            )
            for operation in operations_to_mitigate
        ]
        mitigated_values = np.array(
            [
                execute_with_pec(
                    circuit=training_circuit,
                    observable=observable,
                    executor=noisy_executor,
                    representations=representations,
                    full_output=False,
                    **pec_kwargs,
                )
                for training_circuit in training_circuits
            ]
        )

    return np.mean((mitigated_values - ideal_values) ** 2)


def biased_noise_loss_function(
    params: npt.NDArray[np.float64],
    operations_to_mitigate: list[QPROGRAM],
    training_circuits: list[QPROGRAM],
    ideal_values: npt.NDArray[np.float64],
    noisy_executor: Executor,
    pec_kwargs: dict[str, Any],
    pec_data: npt.NDArray[np.float64] | None = None,
    observable: Observable | None = None,
) -> float:
    r"""Loss function for optimizing quasi-probability representations
    assuming a biased noise model depending on two real parameters.

    Args:
        params: Array of optimization parameters epsilon
            (local noise strength) and eta (noise bias between reduced
            dephasing and depolarizing channels).
        operations_to_mitigate: List of ideal operations to be represented by
            a combination of noisy operations.
        training_circuits: List of training circuits for generating the
            error-mitigated expectation values.
        ideal_values: Expectation values obtained by noiseless simulations.
        noisy_executor: Executes the circuit with noise and returns a
            `QuantumResult`.
        pec_kwargs: Options to pass to `execute_w_pec` for the error-mitigated
            expectation value obtained from executing the training circuits.
        pec_data (optional): 3-D array of error-mitigated expection values for
            model training.
        observable (optional): Observable to compute the expectation value of.
            If None, the ``executor`` must return an expectation value.
            Otherwise the ``QuantumResult`` returned by ``executor`` is used to
            compute the expectation of the observable.

    Returns: Mean squared error between the error-mitigated values and
        the ideal values, over the training set.
    """
    epsilon = params[0]
    eta = params[1]

    if pec_data is not None:
        ind_eps = np.abs(pec_data[:, 0, 0] - epsilon).argmin()
        ind_eta = np.abs(pec_data[0, :, 0] - eta).argmin()
        mitigated_values = pec_data[ind_eps, ind_eta, :]

    else:
        representations = [
            represent_operation_with_local_biased_noise(
                operation,
                epsilon,
                eta,
            )
            for operation in operations_to_mitigate
        ]
        mitigated_values = np.array(
            [
                execute_with_pec(
                    circuit=training_circuit,
                    observable=observable,
                    executor=noisy_executor,
                    representations=representations,
                    full_output=False,
                    **pec_kwargs,
                )
                for training_circuit in training_circuits
            ]
        )

    return np.mean((mitigated_values - ideal_values) ** 2)


def _parse_learning_kwargs(
    learning_kwargs: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], str, dict[str, Any]]:
    r"""Function for handling additional options and data for the learning
    functions.

    Args:
        learning_kwargs: Additional data and options including
            ``pec_data`` from pre-executed runs of PEC on training circuits,
            ``method`` an optimization method supported by
            ``scipy.optimize.minimize``, and settings for the chosen
            optimization method.

    Returns:
       Values contained in learning_kwargs or defaults if not specified.
    """
    minimize_kwargs = learning_kwargs.get("learning_kwargs")

    if minimize_kwargs is None:
        minimize_kwargs = {}

    pec_data = minimize_kwargs.pop("pec_data", None)
    method = minimize_kwargs.pop("method", "Nelder-Mead")

    return pec_data, method, minimize_kwargs
