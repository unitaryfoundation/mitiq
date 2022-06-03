# Copyright (C) 2022 Unitary Fund
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
"""Function to calculate parameters for biased noise model via a
learning-based technique."""

from typing import Optional, List
from typing import cast
import numpy as np
from scipy.optimize import minimize
from cirq import Circuit, LineQubit, Gate
from mitiq import QPROGRAM, Executor, Observable
from mitiq.cdr import generate_training_circuits
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)


def learn_biased_noise_parameters(
    operation: Gate,
    circuit: QPROGRAM,
    ideal_executor: Executor,
    noisy_executor: Executor,
    num_training_circuits: int = 10,
    epsilon0: float = 0,
    eta0: float = 1,
    observable: Optional[Observable] = None,
) -> List[float]:
    r"""Loss function: optimize the quasiprobability representation using
    the method of least squares

    Args:
        operation: ideal operation to be represented by a (learning-optmized)
            combination of noisy operations.
        circuit: the full quantum program as defined by the user.
        ideal_executor: Executes the ideal circuit and returns a
            `QuantumResult`.
        noisy_executor: Executes the noisy circuit and returns a
            `QuantumResult`.
        num_training_circuits: number of Clifford circuits to be generated for
            training data.
        epsilon0: initial guess for noise strength.
        eta0: initial guess for noise bias.
        observable (optional): Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.

    Returns:
        Optimized noise strength epsilon and noise bias eta.
    """
    training_circuits = generate_training_circuits(
        circuit=circuit,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=0,
        method_select="uniform",
        method_replace="closest",
    )

    ideal_values = np.array(
        [ideal_executor.evaluate(t, observable) for t in training_circuits]
    )

    x0 = np.array(
        [epsilon0, eta0]
    )  # initial parameter values for optimization
    result = minimize(
        biased_noise_loss_function,
        x0,
        args=(
            operation,
            circuit,
            ideal_values,
            noisy_executor,
            observable,
        ),
        method="BFGS",
    )
    x_result = result.x
    epsilon = x_result[0]
    eta = x_result[1]

    return [epsilon, eta]


def biased_noise_loss_function(
    x: np.ndarray,
    operation: Gate,
    circuit: QPROGRAM,
    ideal_values: np.ndarray,
    noisy_executor: Executor,
    observable: Optional[Observable] = None,
) -> float:
    r"""Loss function for optimizing the quasiprobability representation using
    the method of least squares

    Args:
        x: array of optimization parameters epsilon
            (local noise strength) and eta (noise bias between reduced
            dephasing and depolarizing
            channels)
        operation: ideal operation to be represented by a (learning-optmized)
            combination of noisy operations
        ideal_values: expectation values obtained by simulations run on the
            Clifford training circuit
        noisy_executor: Executes the noisy circuit and returns a
            `QuantumResult`.
        observable (optional): Observable to compute the expectation value of.
            If None, the `executor` must return an expectation value. Otherwise
            the `QuantumResult` returned by `executor` is used to compute the
            expectation of the observable.

    Returns: Square of the difference between the error-mitigated value and
        the ideal value, over the training set
    """
    epsilon = x[0]
    eta = x[1]
    qreg = LineQubit.range(operation.num_qubits())
    representations = [
        represent_operation_with_local_biased_noise(
            Circuit(operation.on(*qreg)),
            epsilon,
            eta,
        )
    ]
    mitigated = cast(
        float,
        execute_with_pec(
            circuit=circuit,
            observable=observable,
            executor=noisy_executor,
            representations=representations,
            full_output=False,
        ),
    )

    return np.sum(
        (mitigated * np.ones(len(ideal_values)) - ideal_values) ** 2
    ) / len(ideal_values)
