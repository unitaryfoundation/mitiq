# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, cast

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import QPROGRAM, MeasurementResult, QuantumResult
from mitiq.observable.pauli import PauliString, PauliStringCollection
from mitiq.typing import Numeric


class Observable:
    """A quantum observable typically used to compute its mitigated expectation
    value.

    Args:
        paulis: PauliStrings used to define the observable.
    """

    def __init__(self, *paulis: PauliString) -> None:
        self._paulis = _combine_duplicate_pauli_strings(paulis)
        self._groups: list[PauliStringCollection]
        self._ngroups: int
        self.partition()

    @staticmethod
    def from_pauli_string_collections(
        *pauli_string_collections: PauliStringCollection,
    ) -> "Observable":
        """Creates an ``Observable`` from one or more ``PauliStringCollection``
            instances.

        Args:
            pauli_string_collections: One or more collections of Pauli strings
                used to define the observable.

        Returns:
            An ``Observable`` containing the given Pauli string collections.
        """
        obs = Observable()
        obs._groups = list(pauli_string_collections)
        obs._ngroups = len(pauli_string_collections)
        obs._paulis = [
            pauli
            for pauli_string_collection in pauli_string_collections
            for pauli in pauli_string_collection.elements
        ]
        return obs

    @property
    def nterms(self) -> int:
        return len(self._paulis)

    def _qubits(self) -> set[cirq.Qid]:
        """Returns all qubits acted on by the Observable."""
        return {q for pauli in self._paulis for q in pauli._pauli.qubits}

    @property
    def paulis(self) -> list[PauliString]:
        return self._paulis

    @property
    def qubit_indices(self) -> list[int]:
        return [cast(cirq.LineQubit, q).x for q in sorted(self._qubits())]

    @property
    def nqubits(self) -> int:
        return len(self.qubit_indices)

    def __mul__(
        self, other: "Observable | PauliString | Numeric"
    ) -> "Observable":
        if isinstance(other, (PauliString, int, float, complex)):
            return Observable(*[pauli * other for pauli in self._paulis])
        elif isinstance(other, Observable):
            return Observable(
                *[
                    pauli * other_pauli
                    for pauli in self._paulis
                    for other_pauli in other._paulis
                ]
            )
        return NotImplemented

    def __rmul__(self, other: "PauliString | Numeric") -> "Observable":
        if isinstance(other, (PauliString, int, float, complex)):
            return Observable(*[other * pauli for pauli in self._paulis])
        return NotImplemented

    @property
    def groups(self) -> list[PauliStringCollection]:
        return self._groups

    @property
    def ngroups(self) -> int:
        return self._ngroups

    def partition(self, seed: int | None = None) -> None:
        """Partitions the observable's Pauli strings into commuting groups.

        This method groups the ``PauliStringCollection`` instances such that
        each group consists of mutually commuting operators, which can be
        measured together in a quantum circuit.

        Note:
            This method randomizes the way in which the list of
            paulis is partitioned.

        Args:
            seed: An optional seed for shuffling to ensure deterministic
                behavior when partitioning.
        """
        rng = np.random.RandomState(seed)

        psets: list[PauliStringCollection] = []
        paulis = copy.deepcopy(self._paulis)
        rng.shuffle(paulis)  # type: ignore

        while paulis:
            pauli = paulis.pop()
            added = False
            for i, pset in enumerate(psets):
                if pset.can_add(pauli):
                    pset.add(pauli)
                    added = True
                    break

            if not added:
                psets.append(PauliStringCollection(pauli))

        self._groups = psets
        self._ngroups = len(self._groups)

    def measure_in(self, circuit: QPROGRAM) -> list[QPROGRAM]:
        """Given a quantum circuit, this method returns a list of circuits
        where each circuit corresponds to a different group of commuting Pauli
        strings, which allows measurement in the appropriate basis.

        Args:
            circuit: The quantum circuit in which the observable should be
            measured.

        Returns:
            A list of quantum circuits with the appropriate measurement
            settings for each group of Pauli strings.
        """
        return [pset.measure_in(circuit) for pset in self._groups]

    def matrix(
        self,
        qubit_indices: list[int] | None = None,
    ) -> npt.NDArray[np.complex64]:
        """Returns the (potentially very large) matrix of the ``Observable``.

        Args:
            qubit_indices: Optional list of qubit indices specifying the order
            of qubits in the matrix representation. If None, the default
            ordering from `self.qubit_indices` is used.

        Returns:
            A ``NumPy`` array representing the matrix form of the observable.
        """
        if qubit_indices is None:
            qubit_indices = self.qubit_indices
        n = len(qubit_indices)

        obs_matrix = np.zeros(shape=(2**n, 2**n), dtype=np.complex64)
        for pauli in self._paulis:
            obs_matrix += pauli.matrix(qubit_indices_to_include=qubit_indices)

        return obs_matrix

    def expectation(
        self, circuit: QPROGRAM, execute: Callable[[QPROGRAM], QuantumResult]
    ) -> complex:
        """Computes the expectation value of the observable.

        This method executes the given quantum circuit and estimates the
        expectation value of the observable based on the measurement results.

        Args:
            circuit: The quantum circuit to be executed.
            execute: A function that takes a quantum circuit as input and
            returns a QuantumResult containing measurement outcomes.

        Returns:
            The expectation value of the observable.
        """
        from mitiq.executor import Executor  # Avoid circular import.

        return Executor(execute).evaluate(circuit, observable=self)[0]

    def _expectation_from_measurements(
        self, measurements: list[MeasurementResult]
    ) -> float:
        return sum(
            pset._expectation_from_measurements(bitstrings)
            for (pset, bitstrings) in zip(self._groups, measurements)
        )

    def _expectation_from_density_matrix(
        self, density_matrix: npt.NDArray[np.complex64]
    ) -> float:
        observable_matrix = self.matrix()

        if density_matrix.shape != observable_matrix.shape:
            nqubits = int(np.log2(density_matrix.shape[0]))
            density_matrix = cirq.partial_trace(
                np.reshape(density_matrix, newshape=[2, 2] * nqubits),
                keep_indices=self.qubit_indices,
            ).reshape(observable_matrix.shape)

        return np.real_if_close(
            np.trace(density_matrix @ observable_matrix)
        ).item()

    def __str__(self) -> str:
        return " + ".join(map(str, self._paulis))

    def __eq__(self, other: Any) -> bool:
        return np.allclose(self.matrix(), other.matrix())


def _combine_duplicate_pauli_strings(
    paulis: Iterable[PauliString],
) -> list[PauliString]:
    """Combines duplicate PauliStrings by adding their coefficients.
    Discards paulis with zero coefficients.

    Returns: deduped list of PauliStrings.
    """
    pauli_string_coefficients: defaultdict[PauliString, complex] = defaultdict(
        complex
    )
    for pauli_string in paulis:
        cache_key = pauli_string.with_coeff(1)
        pauli_string_coefficients[cache_key] += pauli_string.coeff
    return [
        pauli_string.with_coeff(coeff)
        for (pauli_string, coeff) in pauli_string_coefficients.items()
        if not np.isclose(coeff, 0.0)
    ]
