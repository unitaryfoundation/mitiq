# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from collections.abc import Sequence
from typing import Any, cast

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import QPROGRAM, MeasurementResult
from mitiq.interface import atomic_converter
from mitiq.typing import Numeric
from mitiq.utils import _cirq_pauli_to_string


class PauliString:
    """A ``PauliString`` is a (tensor) product of single-qubit Pauli gates
    :math:`I, X, Y`, and :math:`Z`, with a leading (real or complex)
    coefficient. ``PauliString`` objects can be measured in any
    ``mitiq.QPROGRAM``.

    Args:
        spec: String specifier of the PauliString. Should only contain
            characters 'I', 'X', 'Y', and 'Z'.
        coeff: Coefficient of the PauliString.
        support: Qubits the ``spec`` acts on, if provided.

    Examples:
        >>> PauliString(spec="IXY")  # X(1)*Y(2)
        >>> PauliString(spec="ZZ", coeff=-0.5)  # -0.5*Z(0)*Z(1)
        >>> PauliString(spec="XZ", support=(10, 17))  # X(10)*Z(17)
    """

    _string_to_gate_map = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

    def __init__(
        self,
        spec: str = "",
        coeff: complex = 1.0,
        support: Sequence[int] | None = None,
    ) -> None:
        if not set(spec).issubset(set(self._string_to_gate_map.keys())):
            raise ValueError(
                f"One or more invalid characters in spec {spec}. Valid "
                f"characters are 'I', 'X', 'Y', and 'Z', and the spec should "
                f"not contain any spaces."
            )
        if support is not None:
            if len(support) != len(spec):
                raise ValueError(
                    f"The spec has {len(spec)} Pauli's but the support has "
                    f"{len(support)} qubits. These numbers must be equal."
                )
        else:
            support = range(len(spec))

        self._pauli: cirq.PauliString[Any] = cirq.PauliString(
            coeff,
            (
                self._string_to_gate_map[s].on(cirq.LineQubit(i))
                for (i, s) in zip(support, spec)
            ),
        )

    @staticmethod
    def from_cirq_pauli_string(
        cirq_pauli_string: cirq.PauliString[Any],
    ) -> "PauliString":
        return PauliString(
            spec=_cirq_pauli_to_string(cirq_pauli_string),
            coeff=cirq_pauli_string.coefficient,  # type: ignore
            support=sorted(q.x for q in cirq_pauli_string.qubits),
        )

    @property
    def coeff(self) -> complex:
        return cast(complex, self._pauli.coefficient)

    def matrix(
        self,
        qubit_indices_to_include: list[int] | None = None,
    ) -> npt.NDArray[np.complex64]:
        """Returns the (potentially very large) matrix of the PauliString."""
        qubits = (
            [cirq.LineQubit(x) for x in qubit_indices_to_include]
            if qubit_indices_to_include
            else self._pauli.qubits
        )
        return self._pauli.matrix(qubits=qubits)

    def _basis_rotations(self) -> list[cirq.Operation]:
        """Returns the basis rotations needed to measure the PauliString."""
        return [
            op
            for op in self._pauli.to_z_basis_ops()
            if op.gate != cirq.SingleQubitCliffordGate.I
        ]

    def _qubits_to_measure(self) -> set[cirq.Qid]:
        return set(self._pauli.qubits)

    def measure_in(self, circuit: QPROGRAM) -> QPROGRAM:
        return PauliStringCollection(self).measure_in(circuit)

    def can_be_measured_with(self, other: "PauliString") -> bool:
        """Returns True if the expectation value of the PauliString can be
        simultaneously estimated with `other` via single-qubit measurements.

        Args:
            other: The PauliString to check simultaneous measurement with.
        """
        overlap = set(self._pauli.qubits).intersection(
            set(other._pauli.qubits)
        )
        for qubit in overlap:
            if cirq.I in (self._pauli.get(qubit), other._pauli.get(qubit)):
                continue
            if self._pauli.get(qubit) != other._pauli.get(qubit):
                return False
        return True

    def with_coeff(self, coeff: complex) -> "PauliString":
        return PauliString(
            spec=self.spec, coeff=coeff, support=sorted(self.support())
        )

    @property
    def spec(self) -> str:
        """Returns a string representation of the Pauli gates in
        the PauliString."""
        return _cirq_pauli_to_string(self._pauli)

    def support(self) -> set[int]:
        return {q.x for q in self._pauli.qubits}

    def weight(self) -> int:
        """Returns the weight of the PauliString, i.e., the number of
        non-identity terms in the PauliString.
        """
        return sum(gate != cirq.I for gate in self._pauli.values())

    def _expectation_from_measurements(
        self, measurements: MeasurementResult
    ) -> float:
        return PauliStringCollection(self)._expectation_from_measurements(
            measurements
        )

    def __mul__(self, other: "PauliString | Numeric") -> "PauliString":
        if isinstance(other, PauliString):
            return PauliString.from_cirq_pauli_string(
                self._pauli * other._pauli
            )
        elif isinstance(other, (int, float, complex)):
            return PauliString.from_cirq_pauli_string(self._pauli * other)
        return NotImplemented

    def __rmul__(self, other: Numeric) -> "PauliString":
        return self.__mul__(other)

    def __eq__(self, other: Any) -> bool:
        return self._pauli == other._pauli

    def __hash__(self) -> int:
        return self._pauli.__hash__()

    def __str__(self) -> str:
        return str(self._pauli)

    def __repr__(self) -> str:
        return repr(self._pauli)


class PauliStringCollection:
    """A collection of PauliStrings that qubit-wise commute and so can be
    measured with a single circuit.

    Args:
        paulis: PauliStrings to add to the collection.
        check_precondition: If True, raises an error if some of the
            ``PauliString`` objects do not qubit-wise commute.

    Example:
        >>> pcol = PauliStringCollection(
        >>>     PauliString(spec="X"),
        >>>     PauliString(spec="IZ", coeff=-2.2)
        >>> )
        >>> print(pcol)  # X(0) + (-2.2+0j)*Z(1)
        >>> print(pcol.support())  # {0, 1}
        >>>
        >>> # XZ qubit-wise commutes with X(0) and Z(1), so can be added.
        >>> print(pcol.can_add(PauliString(spec="XZ")))  # True.
        >>> pcol.add(PauliString(spec="XZ"))
        >>> print(pcol)  # X(0) + (-2.2+0j)*Z(1) + X(0)*Z(1)
        >>>
        >>> # Z(0) doesn't qubit-wise commute with X(0), so can't be added.
        >>> print(pcol.can_add(PauliString(spec="Z")))  # False.
    """

    def __init__(
        self, *paulis: PauliString, check_precondition: bool = True
    ) -> None:
        self._paulis_by_weight: dict[int, Counter[PauliString]] = dict()
        self.add(*paulis, check_precondition=check_precondition)

    def can_add(self, pauli: PauliString) -> bool:
        return all(pauli.can_be_measured_with(p) for p in self.elements)

    def add(
        self, *paulis: PauliString, check_precondition: bool = True
    ) -> None:
        for pauli in paulis:
            if check_precondition and not self.can_add(pauli):
                raise ValueError(
                    f"Cannot add PauliString {pauli} to PauliStringCollection."
                )
            weight = pauli.weight()
            if self._paulis_by_weight.get(weight) is None:
                self._paulis_by_weight[weight] = Counter({pauli})
            else:
                self._paulis_by_weight[weight].update({pauli})

    @property
    def elements(self) -> list[PauliString]:
        return [
            pauli
            for paulis in self._paulis_by_weight.values()
            for pauli in paulis.elements()
        ]

    @property
    def elements_by_weight(self) -> dict[int, Counter[PauliString]]:
        return self._paulis_by_weight

    def support(self) -> set[int]:
        return {cast(cirq.LineQubit, q).x for q in self._qubits_to_measure()}

    def max_weight(self) -> int:
        return max(self._paulis_by_weight.keys(), default=0)

    def min_weight(self) -> int:
        return min(self._paulis_by_weight.keys(), default=0)

    def _qubits_to_measure(self) -> set[cirq.Qid]:
        qubits: set[cirq.Qid] = set()
        for pauli in self.elements:
            qubits.update(pauli._pauli.qubits)
        return qubits

    def measure_in(self, circuit: QPROGRAM) -> QPROGRAM:
        return self._measure_in(circuit, self)

    @staticmethod
    @atomic_converter
    def _measure_in(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map[q])

        if not paulis._qubits_to_measure().issubset(set(circuit.all_qubits())):
            raise ValueError(
                f"Qubit mismatch. The PauliString(s) act on qubits "
                f"{paulis.support()} but the circuit has qubit indices "
                f"{sorted([q for q in circuit.all_qubits()])}."
            )

        basis_rotations = set()
        support = set()
        qubits_with_measurements = set[cirq.Qid]()

        # Find any existing measurement gates in the circuit
        for _, op, _ in circuit.findall_operations_with_gate_type(
            cirq.MeasurementGate
        ):
            qubits_with_measurements.update(op.qubits)

        for pauli in paulis.elements:
            basis_rotations.update(pauli._basis_rotations())
            support.update(pauli._qubits_to_measure())
        measured = circuit + basis_rotations + cirq.measure(*sorted(support))

        if support & qubits_with_measurements:
            raise ValueError(
                f"More than one measurement found for qubits: "
                f"{support & qubits_with_measurements}. Only a single "
                f"measurement is allowed per qubit."
            )

        # Transform circuit back to original qubits.
        reverse_qubit_map = dict(zip(qubit_map.values(), qubit_map.keys()))
        return measured.transform_qubits(lambda q: reverse_qubit_map[q])

    def _expectation_from_measurements(
        self, measurements: MeasurementResult
    ) -> float:
        total = 0.0
        for pauli in self.elements:
            bitstrings = measurements.filter_qubits(sorted(pauli.support()))
            value = (
                np.average([(-1) ** np.sum(bits) for bits in bitstrings])
                if len(bitstrings) > 0
                else 1.0
            )
            total += pauli.coeff * value
        return total

    def __eq__(self, other: Any) -> bool:
        return self._paulis_by_weight == other._paulis_by_weight

    def __len__(self) -> int:
        return len(self.elements)

    def __str__(self) -> str:
        return " + ".join(map(str, self.elements))
