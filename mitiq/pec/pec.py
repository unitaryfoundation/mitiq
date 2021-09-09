# Copyright (C) 2020 Unitary Fund
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

"""High-level probabilistic error cancellation tools."""

from typing import Optional, Callable, List, Union, Tuple, Dict, Any
from functools import wraps
import warnings

import numpy as np

from mitiq import QPROGRAM
from mitiq.executor import generate_collected_executor
from mitiq.pec import sample_circuit, OperationRepresentation
from mitiq.interface import convert_to_mitiq


class LargeSampleWarning(Warning):
    """Warning is raised when PEC sample size is greater than 10 ** 5
    """

    pass


_LARGE_SAMPLE_WARN = (
    "The number of PEC samples is very large. It may take several minutes."
    " It may be necessary to reduce 'precision' or 'num_samples'."
)


def execute_with_pec(
    circuit: QPROGRAM,
    executor: Callable[[QPROGRAM], float],
    representations: List[OperationRepresentation],
    precision: float = 0.03,
    num_samples: Optional[int] = None,
    force_run_all: bool = True,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    full_output: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    r"""Evaluates the expectation value associated to the input circuit
    using probabilistic error cancellation (PEC) [Temme2017]_ [Endo2018]_.

    This function implements PEC by:

    1. Sampling different implementable circuits from the quasi-probability
       representation of the input circuit;
    2. Evaluating the noisy expectation values associated to the sampled
       circuits (through the "executor" function provided by the user);
    3. Estimating the ideal expectation value from a suitable linear
       combination of the noisy ones.

    Args:
        circuit: The input circuit to execute with error-mitigation.
        executor: A function which executes a circuit (sequence of circuits)
            and returns an expectation value (sequence of expectation values).
        representations: Representations (basis expansions) of each operation
            in the input circuit.
        precision: The desired estimation precision (assuming the observable
            is bounded by 1). The number of samples is deduced according
            to the formula (one_norm / precision) ** 2, where 'one_norm'
            is related to the negativity of the quasi-probability
            representation [Temme2017]_. If 'num_samples' is explicitly set
            by the user, 'precision' is ignored and has no effect.
        num_samples: The number of noisy circuits to be sampled for PEC.
            If not given, this is deduced from the argument 'precision'.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.
        random_state: Seed for sampling circuits.
        full_output: If False only the average PEC value is returned.
            If True a dictionary containing all PEC data is returned too.

    Returns:
        pec_value: The PEC estimate of the ideal expectation value associated
            to the input circuit.
        pec_data: A dictionary which contains all the raw data involved in the
            PEC process (including the PEC estimation error). The error is
            estimated as pec_std / sqrt(num_samples), where 'pec_std' is the
            standard deviation of the PEC samples, i.e., the square root of the
            mean squared deviation of the sampled values from 'pec_value'.
            This is returned only if ``full_output`` is ``True``.

    .. [Endo2018] : Suguru Endo, Simon C. Benjamin, Ying Li,
        "Practical Quantum Error Mitigation for Near-Future Applications"
        *Phys. Rev. **X 8**, 031027 (2018),
        (https://arxiv.org/abs/1712.09271).

    .. [Takagi2020] : Ryuji Takagi,
        "Optimal resource cost for error mitigation,"
        (https://arxiv.org/abs/2006.12509).
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if not (0 < precision <= 1):
        raise ValueError(
            "The value of 'precision' should be within the interval (0, 1],"
            f" but precision is {precision}."
        )

    converted_circuit, _ = convert_to_mitiq(circuit)

    # Get the 1-norm of the circuit quasi-probability representation
    _, _, norm = sample_circuit(
        converted_circuit, representations, num_samples=1,
    )

    # Deduce the number of samples (if not given by the user)
    if not isinstance(num_samples, int):
        num_samples = int((norm / precision) ** 2)

    # Issue warning for very large sample size
    if num_samples > 10 ** 5:
        warnings.warn(_LARGE_SAMPLE_WARN, LargeSampleWarning)

    # Sample all the circuits
    sampled_circuits, signs, _ = sample_circuit(
        converted_circuit,
        representations,
        random_state=random_state,
        num_samples=num_samples,
    )

    # Execute all sampled circuits
    collected_executor = generate_collected_executor(
        executor, force_run_all=force_run_all
    )
    results = collected_executor(sampled_circuits)

    # Evaluate unbiased estimators [Temme2017] [Endo2018] [Takagi2020]
    unbiased_estimators = [
        norm * s * val  # type: ignore[operator]
        for s, val in zip(signs, results)
    ]

    pec_value = np.average(unbiased_estimators)

    if not full_output:
        return pec_value

    # Build dictionary with additional results and data
    pec_data: Dict[str, Any] = {}

    pec_data = {
        "num_samples": num_samples,
        "precision": precision,
        "pec_value": pec_value,
        "pec_error": np.std(unbiased_estimators) / np.sqrt(num_samples),
        "unbiased_estimators": unbiased_estimators,
        "measured_expectation_values": results,
        "sampled_circuits": sampled_circuits,
    }

    return pec_value, pec_data


def mitigate_executor(
    executor: Callable[[QPROGRAM], float],
    representations: List[OperationRepresentation],
    precision: float = 0.03,
    num_samples: Optional[int] = None,
    force_run_all: bool = True,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    full_output: bool = False,
) -> Callable[[QPROGRAM], Union[float, Tuple[float, Dict[str, Any]]]]:
    """Returns a probabilistic error cancellation (PEC) mitigated version of
    the input 'executor'.

    The input `executor` executes a circuit with an arbitrary backend and
    produces an expectation value (without any error mitigation). The returned
    executor executes the circuit with the same backend but uses probabilistic
    error cancellation to produce the PEC estimate of the ideal expectation
    value associated to the input circuit as well as A dictionary which
    contains all the raw data involved in the PEC process.

    Args:
        executor: A function which executes a circuit (sequence of circuits)
            and returns an expectation value (sequence of expectation values).
        representations: Representations (basis expansions) of each operation
            in the input circuit.
        precision: The desired estimation precision (assuming the observable
            is bounded by 1). The number of samples is deduced according
            to the formula (one_norm / precision) ** 2, where 'one_norm'
            is related to the negativity of the quasi-probability
            representation [Temme2017]_. If 'num_samples' is explicitly set
            by the user, 'precision' is ignored and has no effect.
        num_samples: The number of noisy circuits to be sampled for PEC.
            If not given, this is deduced from the argument 'precision'.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.
        random_state: Seed for sampling circuits.
        full_output: If False only the average PEC value is returned.
            If True a dictionary containing all PEC data is returned too.
    """

    @wraps(executor)
    def new_executor(
        circuit: QPROGRAM,
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        return execute_with_pec(
            circuit,
            executor,
            representations,
            precision,
            num_samples,
            force_run_all,
            random_state,
            full_output,
        )

    return new_executor


def pec_decorator(
    representations: List[OperationRepresentation],
    precision: float = 0.03,
    num_samples: Optional[int] = None,
    force_run_all: bool = True,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    full_output: bool = False,
) -> Callable[
    [Callable[[Union[QPROGRAM, Any, Any, Any]], float]],
    Callable[
        [Union[QPROGRAM, Any, Any, Any]],
        Union[float, Tuple[float, Dict[str, Any]]],
    ],
]:
    """Decorator which adds probabilistic error cancellation (PEC) mitigation
    to an executor function, i.e., a function which executes a quantum circuit
    with an arbitrary backend and returns the PEC estimate of the ideal
    expectation value associated to the input circuit as well as A dictionary
    which contains all the raw data involved in the PEC process.

    Args:
        representations: Representations (basis expansions) of each operation
            in the input circuit.
        precision: The desired estimation precision (assuming the observable
            is bounded by 1). The number of samples is deduced according
            to the formula (one_norm / precision) ** 2, where 'one_norm'
            is related to the negativity of the quasi-probability
            representation [Temme2017]_. If 'num_samples' is explicitly set
            by the user, 'precision' is ignored and has no effect.
        num_samples: The number of noisy circuits to be sampled for PEC.
            If not given, this is deduced from the argument 'precision'.
        force_run_all: If True, all sampled circuits are executed regardless of
            uniqueness, else a minimal unique set is executed.
        random_state: Seed for sampling circuits.
        full_output: If False only the average PEC value is returned.
            If True a dictionary containing all PEC data is returned too.
    """

    def decorator(
        executor: Callable[[QPROGRAM], float]
    ) -> Callable[[QPROGRAM], Union[float, Tuple[float, Dict[str, Any]]]]:
        return mitigate_executor(
            executor,
            representations,
            precision,
            num_samples,
            force_run_all,
            random_state,
            full_output,
        )

    return decorator
