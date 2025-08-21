# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines input / output types for a quantum computer (simulator):

* SUPPORTED_PROGRAM_TYPES: All supported packages / circuits which Mitiq can
     interface with,
* QPROGRAM: All supported packages / circuits which are installed in the
     environment Mitiq is run in, and
* QuantumResult: An object returned by a quantum computer (simulator) running
     a quantum program from which expectation values to be mitigated can be
     computed. Note this includes expectation values themselves.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Any, TypeAlias, Union, cast

import numpy as np
import numpy.typing as npt
from cirq import Circuit as _Circuit


class EnhancedEnumMeta(EnumMeta):
    def __str__(cls) -> str:
        return ", ".join(
            [member.name.lower() for member in cast(type[Enum], cls)]
        )


class EnhancedEnum(Enum, metaclass=EnhancedEnumMeta):
    # This is for backwards compatibility with the old representation
    # of SUPPORTED_PROGRAM_TYPES, which was a dictionary
    @classmethod
    def keys(cls) -> list[str]:
        return [member.name.lower() for member in cls]


try:
    from pyquil import Program as _Program
except ImportError:  # pragma: no cover
    _Program = _Circuit  # type: ignore

try:
    from qiskit import QuantumCircuit as _QuantumCircuit
except ImportError:  # pragma: no cover
    _QuantumCircuit = _Circuit

try:
    from braket.circuits import Circuit as _BKCircuit
except ImportError:  # pragma: no cover
    _BKCircuit = _Circuit

try:
    from pennylane.tape import QuantumTape as _QuantumTape
except ImportError:  # pragma: no cover
    _QuantumTape = _Circuit

try:
    from qibo.models.circuit import Circuit as _QiboCircuit
except ImportError:  # pragma: no cover
    _QiboCircuit = _Circuit


# Supported + installed quantum programs.
QPROGRAM = Union[
    _Circuit, _Program, _QuantumCircuit, _BKCircuit, _QuantumTape, _QiboCircuit
]


# Supported quantum programs.
class SUPPORTED_PROGRAM_TYPES(EnhancedEnum):
    BRAKET = _BKCircuit
    CIRQ = _Circuit
    PENNYLANE = _QuantumTape
    PYQUIL = _Program
    QIBO = _QiboCircuit
    QISKIT = _QuantumCircuit


# Define MeasurementResult, a result obtained by measuring qubits on a quantum
# computer.
Bitstring = str | list[int]


@dataclass
class MeasurementResult:
    """Mitiq object for collecting the bitstrings sampled from a quantum
    computer when executing a circuit. This is one of the possible types
    (see :class:`~mitiq.typing.QuantumResult`) that an
    :class:`.Executor` can return.

    Args:
        result:
            The sequence of measured bitstrings.
        qubit_indices:
            The qubit indices associated to each
            bit in a bitstring (from left to right).
            If not given, Mitiq assumes the default ordering
            ``tuple(range(self.nqubits))``, where ``self.nqubits``
            is the bitstring length deduced from ``result``.

    Example:
        >>> mr = MeasurementResult(["001", "010", "001"])
        >>> mr.get_counts()
        {'001': 2, '010': 1}

    Warning:
        Use caution when selecting the default option for ``qubit_indices``,
        especially when estimating an :class:`.Observable`
        acting on a subset of qubits. In this case Mitiq
        only applies measurement gates to the specific qubits and, therefore,
        it is essential to specify the corresponding ``qubit_indices``.
    """

    result: Sequence[Bitstring]
    qubit_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Validate arguments
        if isinstance(self.result, dict):
            raise TypeError(
                "Use the MeasurementResult.from_counts method to instantiate "
                "a MeasurementResult object from a dictionary."
            )
        symbols = set(b for bits in self.result for b in bits)
        if not (symbols.issubset({0, 1}) or symbols.issubset({"0", "1"})):
            raise ValueError("Bitstrings should look like '011' or [0, 1, 1].")

        if symbols.issubset({"0", "1"}):
            # Convert to list of integers
            int_result = [[int(b) for b in bits] for bits in self.result]
            self.result: list[list[int]] = list(int_result)

        if isinstance(self.result, np.ndarray):
            self.result = self.result.tolist()

        self._bitstrings = np.array(self.result)

        if not self.qubit_indices:
            self.qubit_indices = tuple(range(self.nqubits))
        else:
            if len(self.qubit_indices) != self.nqubits:
                raise ValueError(
                    f"MeasurementResult has {self.nqubits} qubit(s) but there "
                    f"are {len(self.qubit_indices)} `qubit_indices`."
                )

        self._measurements = dict(zip(self.qubit_indices, self._bitstrings.T))

    @property
    def shots(self) -> int:
        return self._bitstrings.shape[0]

    @property
    def nqubits(self) -> int:
        return (
            self._bitstrings.shape[1]
            if len(self._bitstrings.shape) >= 2
            else 0
        )

    @property
    def asarray(self) -> npt.NDArray[np.int64]:
        return self._bitstrings

    @classmethod
    def from_counts(
        cls,
        counts: dict[str, int],
        qubit_indices: tuple[int, ...] | None = None,
    ) -> "MeasurementResult":
        """Initializes a ``MeasurementResult`` from a dictionary of counts.

        **Example**::

            MeasurementResult.from_counts({"00": 175, "11": 177})
        """
        counter = Counter(counts)
        return cls(list(counter.elements()), qubit_indices)

    def get_counts(self) -> dict[str, int]:
        """Returns a Python dictionary whose keys are the measured
        bitstrings and whose values are the counts.
        """
        strings = ["".join(map(str, bits)) for bits in self.result]
        return dict(Counter(strings))

    def prob_distribution(self) -> dict[str, float]:
        """Returns a Python dictionary whose keys are the measured
        bitstrings and whose values are their empirical frequencies.
        """
        return {k: v / self.shots for k, v in self.get_counts().items()}

    def to_dict(self) -> dict[str, Any]:
        """Exports data to a Python dictionary.

        Note: Information about the order measurements is not preserved.
        """

        return {
            "nqubits": self.nqubits,
            "qubit_indices": self.qubit_indices,
            "shots": self.shots,
            "counts": self.get_counts(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeasurementResult":
        """Loads a ``MeasurementResult`` from a Python dictionary.

        Note: Only ``data["counts"]`` and ``data["qubit_indices"]`` are used
        by this method. Total shots and number of qubits are deduced.
        """
        return cls.from_counts(data["counts"], data["qubit_indices"])

    def filter_qubits(self, qubit_indices: list[int]) -> npt.NDArray[np.int64]:
        """Returns the bitstrings associated to a subset of qubits."""
        return np.array([self._measurements[i] for i in qubit_indices]).T

    def __repr__(self) -> str:
        # We redefine __repr__ in this way to avoid very long output strings.
        return "MeasurementResult: " + str(self.to_dict())


# An `executor` function inputs a quantum program and outputs an object from
# which expectation values can be computed. Explicitly, this object can be one
# of the following types:
# - float: The expectation value itself.
# - MeasurementResult: Sampled bitstrings.
# - np.ndarray: Density matrix.
QuantumResult = float | MeasurementResult | np.ndarray

Numeric: TypeAlias = int | float | complex
