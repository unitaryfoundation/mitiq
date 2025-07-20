# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utilities for efficiently running collections of circuits generated
by error mitigation techniques to compute expectation values."""

import inspect
import typing
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any, List, Tuple, cast, get_args

import numpy as np
import numpy.typing as npt

from mitiq import QPROGRAM, MeasurementResult, QuantumResult
from mitiq.interface import convert_from_mitiq, convert_to_mitiq
from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString

DensityMatrixLike = [
    np.ndarray,
    Iterable[np.ndarray],  # type: ignore
    list[np.ndarray],  # type: ignore
    List[np.ndarray],  # type: ignore
    Sequence[np.ndarray],  # type: ignore
    tuple[np.ndarray],  # type: ignore
    Tuple[np.ndarray],  # type: ignore
    npt.NDArray[np.complex64],
    list[npt.NDArray[np.complex64]],
    list[np.ndarray],  # type: ignore
    tuple[npt.NDArray[np.complex64]],
]
FloatLike = [
    None,  # Untyped executors are assumed to return floats.
    float,
    Iterable[float],
    list[float],
    Sequence[float],
    tuple[float],
    list[float],
    tuple[float],
]
MeasurementResultLike = [
    MeasurementResult,
    Iterable[MeasurementResult],
    list[MeasurementResult],
    Sequence[MeasurementResult],
    tuple[MeasurementResult],
    list[MeasurementResult],
    tuple[MeasurementResult],
]


class Executor:
    """Tool for efficiently scheduling/executing quantum programs and storing
    the results.

    Args:
        executor: A function which inputs a program and outputs a
            ``mitiq.QuantumResult``, or inputs a sequence of programs and
            outputs a sequence of ``mitiq.QuantumResult`` s.
        max_batch_size: Maximum number of programs that can be sent in a
            single batch (if the executor is batched).
    """

    def __init__(
        self,
        executor: Callable[[QPROGRAM | Sequence[QPROGRAM]], Any],
        max_batch_size: int = 75,
    ) -> None:
        self._executor = executor

        executor_annotation = inspect.getfullargspec(executor).annotations
        self._executor_return_type = executor_annotation.get("return")
        self._max_batch_size = max_batch_size

        self._executed_circuits: list[QPROGRAM] = []
        self._quantum_results: list[QuantumResult] = []

        self._calls_to_executor: int = 0

    @property
    def can_batch(self) -> bool:
        """Returns True if the executor is recognized as a "batched executor",
        else False.

        The executor is detected as "batched" if and only if it is annotated
        with a return type that is a subclass of ``Iterable``. Common examples
        include:

        * ``Iterable[QuantumResult]``
        * ``List[QuantumResult]``/``list[QuantumResult]``
        * ``Sequence[QuantumResult]``
        * ``Tuple[QuantumResult]``/``tuple[QuantumResult]``

        Otherwise, it is considered "serial".

        Batched executors can *run several quantum programs in a single call*.

        Returns:
            True if the executor is detected as batched, else False.
        """
        return_type = self._executor_return_type

        if return_type is None:
            return False

        return return_type in (
            BatchedType[T]  # type: ignore[index]
            for BatchedType in [
                Iterable,
                List,
                typing.Sequence,
                Tuple,
                list,
                tuple,
                Sequence,
            ]
            for T in get_args(QuantumResult)
        )

    @property
    def executed_circuits(self) -> list[QPROGRAM]:
        return self._executed_circuits

    @property
    def quantum_results(self) -> list[QuantumResult]:
        return self._quantum_results

    @property
    def calls_to_executor(self) -> int:
        return self._calls_to_executor

    def evaluate(
        self,
        circuits: QPROGRAM | list[QPROGRAM],
        observable: Observable | None = None,
        force_run_all: bool = True,
        **kwargs: Any,
    ) -> list[float]:
        """Returns the expectation value Tr[ρ O] for each circuit in
        ``circuits`` where O is the observable provided or implicitly defined
        by the ``executor``. (The observable is implicitly defined when the
        ``executor`` returns float(s).)

        All executed circuits are stored in ``self.executed_circuits``, and all
        quantum results are stored in ``self.quantum_results``.

        Args:
            circuits: A single circuit or list of circuits.
            observable: Observable O in the expression Tr[ρ O]. If None,
                the ``executor`` must return a float (which corresponds to
                Tr[ρ O] for a specific, fixed observable O).
            force_run_all: If True, force every circuit in the input sequence
                to be executed (if some are identical). Else, detects identical
                circuits and runs a minimal set.

        Returns:
            List of real valued expectation values.
        """
        if not isinstance(circuits, list):
            circuits = [circuits]

        warn_non_hermitian = False
        if observable:
            if isinstance(observable, PauliString):
                if observable.coeff.imag > 0.0001:
                    warn_non_hermitian = True
            elif isinstance(observable, Observable):
                if any(
                    pauli.coeff.imag > 0.0001 for pauli in observable._paulis
                ):
                    warn_non_hermitian = True
        if warn_non_hermitian:
            warnings.warn(
                "Expected observable to be hermitian. Continue with caution."
            )

        # Check executor and observable compatibility with type hinting
        # If FloatLike is specified as a return and observable is used
        if self._executor_return_type in FloatLike and observable is not None:
            # Type hinted as FloatLike and observable passed
            if self._executor_return_type is not None:
                raise ValueError(
                    "When using an executor which returns a float-like "
                    "result, measurements should be added before the circuit "
                    "is executed instead of with an observable."
                )
            else:
                # Using an observable but no type hinting
                raise ValueError(
                    "When using an observable, the return type of the "
                    "executor must be specified using typehinting."
                )
        elif observable is None:
            # Type hinted as DensityMatrixLike but no observable is set
            if self._executor_return_type in DensityMatrixLike:
                raise ValueError(
                    "When using a density matrix result, an observable "
                    "is required."
                )
            # Type hinted as MeasurementResultLike but no observable is set
            elif self._executor_return_type in MeasurementResultLike:
                raise ValueError(
                    "When using a measurement, or bitstring, like result, an "
                    "observable is required."
                )

        # Get all required circuits to run.
        if (
            observable is not None
            and self._executor_return_type in MeasurementResultLike
        ):
            all_circuits = [
                circuit_with_measurements
                for circuit in circuits
                for circuit_with_measurements in observable.measure_in(circuit)
            ]
            result_step = observable.ngroups
        else:
            all_circuits = circuits
            result_step = 1

        # Run all required circuits.
        all_results = self.run(all_circuits, force_run_all, **kwargs)

        # Parse the results.
        if self._executor_return_type in FloatLike:
            results = np.real_if_close(
                cast(Sequence[float], all_results)
            ).tolist()

        elif self._executor_return_type in DensityMatrixLike:
            observable = cast(Observable, observable)
            all_results = cast(list[npt.NDArray[np.complex64]], all_results)
            results = [
                observable._expectation_from_density_matrix(density_matrix)
                for density_matrix in all_results
            ]

        elif self._executor_return_type in MeasurementResultLike:
            observable = cast(Observable, observable)
            all_results = cast(list[MeasurementResult], all_results)
            results = [
                observable._expectation_from_measurements(
                    all_results[i : i + result_step]
                )
                for i in range(len(all_results) // result_step)
            ]

        else:
            raise ValueError(
                f"Could not parse executed results from executor with type "
                f"{self._executor_return_type}."
            )

        return results

    def run(
        self,
        circuits: QPROGRAM | Sequence[QPROGRAM],
        force_run_all: bool = True,
        **kwargs: Any,
    ) -> Sequence[QuantumResult]:
        """Runs all input circuits using the least number of possible calls to
        the executor.

        Args:
            circuits: Circuit or sequence thereof to execute with the executor.
            force_run_all: If True, force every circuit in the input sequence
                to be executed (if some are identical). Else, detects identical
                circuits and runs a minimal set.
        """
        if not isinstance(circuits, Sequence):
            circuits = [circuits]

        start_result_index = len(self._quantum_results)

        if force_run_all:
            to_run = circuits
        else:
            # Make circuits hashable.
            # Note: Assumes all circuits are the same type.
            # TODO: Bug! These conversions to/from Mitiq are not safe in that,
            #  e.g., they do not preserve classical register structure in
            #  Qiskit circuits, potentially causing executed results to be
            #  incorrect. Safe conversions should follow the logic in
            #  mitiq.interface.noise_scaling_converter.
            _, conversion_type = convert_to_mitiq(circuits[0])
            hashable_circuits = [
                convert_to_mitiq(circ)[0].freeze() for circ in circuits
            ]

            # Get the unique circuits and counts
            collection = Counter(hashable_circuits)
            to_run = [
                convert_from_mitiq(circ.unfreeze(), conversion_type)
                for circ in collection.keys()
            ]

        if not self.can_batch:
            for circuit in to_run:
                self._call_executor(circuit, **kwargs)

        else:
            stop = len(to_run)
            step = self._max_batch_size
            for i in range(int(np.ceil(stop / step))):
                batch = to_run[i * step : (i + 1) * step]
                self._call_executor(batch, **kwargs)

        results = self._quantum_results[start_result_index:]

        if not force_run_all:
            # Expand computed results to all results using counts.
            results_dict = dict(zip(collection.keys(), results))
            results = [results_dict[key] for key in hashable_circuits]

        return self._post_run(results)

    def _post_run(
        self, results: Sequence[QuantumResult]
    ) -> Sequence[QuantumResult]:
        """Post-processes the measurement results.
        For example, this method can be overridden by a
        readout error mitigation function.
        """
        return results

    def _call_executor(
        self, to_run: QPROGRAM | Sequence[QPROGRAM], **kwargs: Any
    ) -> None:
        """Calls the executor on the input circuit(s) to run. Stores the
        executed circuits in ``self._executed_circuits`` and the quantum
        results in ``self._quantum_results``.

        Args:
            to_run: Circuit(s) to run.
        """
        result = self._executor(to_run, **kwargs)
        self._calls_to_executor += 1

        if self.can_batch:
            self._quantum_results.extend(result)
            self._executed_circuits.extend(to_run)
        else:
            self._quantum_results.append(result)
            self._executed_circuits.append(to_run)
