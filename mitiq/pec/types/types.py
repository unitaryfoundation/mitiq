# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Types used in probabilistic error cancellation."""

import warnings
from copy import deepcopy
from typing import Any

import cirq
import numpy as np
import numpy.typing as npt
from cirq.value.linear_dict import _format_coefficient

from mitiq import QPROGRAM
from mitiq.interface import (
    CircuitConversionError,
    UnsupportedCircuitError,
    convert_to_mitiq,
)


class NoisyOperation:
    """An operation (or sequence of operations) which a noisy quantum computer
    can actually implement.

    Args:
        circuit: A short circuit which, when executed on a given noisy
            quantum computer, generates a noisy channel. It typically
            contains a single-gate or a short sequence of gates.
        channel_matrix: Superoperator representation of the noisy channel
            which is generated when executing the input ``circuit`` on the
            noisy quantum computer.

    Raises:
        TypeError: If ``ideal`` is not a ``QPROGRAM``.
    """

    def __init__(
        self,
        circuit: QPROGRAM,
        channel_matrix: npt.NDArray[np.complex64] | None = None,
    ) -> None:
        self._native_circuit = deepcopy(circuit)

        try:
            cirq_circuit, native_type = convert_to_mitiq(circuit)
        except (CircuitConversionError, UnsupportedCircuitError):
            raise TypeError(
                "Failed to convert to an internal Mitiq representation"
                f"the input circuit:\n{type(circuit)}\n"
            )

        self._circuit = cirq_circuit
        self._native_type = native_type

        dimension = 2**self.num_qubits

        if channel_matrix is None:
            self._channel_matrix = None

        elif channel_matrix.shape != (
            dimension**2,
            dimension**2,
        ):
            raise ValueError(
                f"Arg `channel_matrix` has shape {channel_matrix.shape}"
                " but the expected shape is"
                f" {dimension**2, dimension**2}."
            )
        self._channel_matrix = deepcopy(channel_matrix)

    @property
    def circuit(self) -> cirq.Circuit:
        """Returns the circuit of the NoisyOperation as a Cirq circuit."""
        return self._circuit

    @property
    def native_circuit(self) -> QPROGRAM:
        """Returns the circuit used to initialize the NoisyOperation."""
        return self._native_circuit

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return tuple(self._circuit.all_qubits())

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def channel_matrix(self) -> npt.NDArray[np.complex64]:
        if self._channel_matrix is None:
            raise ValueError("The channel matrix is unknown.")
        return deepcopy(self._channel_matrix)

    def __add__(self, other: Any) -> "NoisyOperation":
        if not isinstance(other, NoisyOperation):
            raise ValueError(
                f"Arg `other` must be a NoisyOperation but was {type(other)}."
            )

        if self.qubits != other.qubits:
            raise NotImplementedError

        if self._channel_matrix is None or other._channel_matrix is None:
            matrix = None
        else:
            matrix = other._channel_matrix @ self._channel_matrix

        return NoisyOperation(self._circuit + other._circuit, matrix)

    def __str__(self) -> str:
        return self._native_circuit.__str__()


class NoisyBasis:
    """A set of noisy operations which a quantum computer can actually
    implement, assumed to form a basis of n-qubit unitary matrices.

    This class has been removed since Mitiq v0.24.0.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The NoisyBasis class has been removed since Mitiq v0.24.0."
            " Please replace it with a list of NoisyOperation objects."
        )


class OperationRepresentation:
    """A decomposition (basis expansion) of an operation or sequence of
    operations in a basis of noisy, implementable operations.

    Args:
        ideal: The ideal operation desired to be implemented.
        basis_expansion: Representation of the ideal operation in a basis
            of ``NoisyOperation`` objects.
        is_qubit_dependent: If True, the representation
            corresponds to the operation on the specific qubits defined in
            ``ideal``. If False, the representation is valid for the same
            gate even if acting on different qubits from those specified in
            ``ideal``.

    Raises:
        TypeError: If all keys of ``basis_expansion`` are not instances of
            ``NoisyOperation`` objects.
    """

    def __init__(
        self,
        ideal: QPROGRAM,
        noisy_operations: list[NoisyOperation],
        coeffs: list[float],
        is_qubit_dependent: bool = True,
    ) -> None:
        if not all(isinstance(o, NoisyOperation) for o in noisy_operations):
            raise TypeError(
                "All elements of `noisy_operations` must be "
                "of type `NoisyOperation`."
            )

        if not all(isinstance(c, float) for c in coeffs):
            raise TypeError("All elements of `coeffs` must be floats.")

        self._native_ideal = deepcopy(ideal)
        self._ideal, self._native_type = convert_to_mitiq(ideal)
        self._noisy_operations = noisy_operations
        self._coeffs = coeffs
        self._norm = sum(abs(c) for c in coeffs)
        self._distribution = [abs(c) / self._norm for c in coeffs]
        self.is_qubit_dependent = is_qubit_dependent
        self._validate()

    def _validate(self) -> None:
        """Validates initialization arguments."""
        if len(self._noisy_operations) != len(self._coeffs):
            raise ValueError(
                "`noisy_operations` and `coeffs` must have equal length"
                f" but {len(self._noisy_operations)}!={len(self._coeffs)}."
            )
        if not np.isclose(sum(self._coeffs), 1.0, atol=10**-4):
            warnings.warn("The sum of the coefficients is different from 1.")
        for op in self._noisy_operations:
            if self._ideal.all_qubits() != op.circuit.all_qubits():
                raise ValueError(
                    "The operation to represent acts on"
                    f" {self._ideal.all_qubits()}. Noisy operations"
                    f" must act on the same qubits but {op} acts on:"
                    f" {op.circuit.all_qubits()}"
                )

    @property
    def ideal(self) -> cirq.Circuit:
        return self._ideal

    @property
    def basis_expansion(self) -> list[tuple[float, NoisyOperation]]:
        return [(c, o) for c, o in zip(self._coeffs, self._noisy_operations)]

    @property
    def noisy_operations(self) -> list[NoisyOperation]:
        return self._noisy_operations

    @property
    def coeffs(self) -> list[float]:
        """Returns the coefficients of the quasi-probability distribution."""
        return self._coeffs

    @property
    def norm(self) -> float:
        """Returns the 1-norm of the quasi-probability distribution."""
        return self._norm

    @property
    def distribution(self) -> list[float]:
        """Returns the probability distribution obtained from taking
        the absolute value and normalizing the quasi-probability distribution.
        """
        return self._distribution

    def sample(
        self, random_state: int | np.random.RandomState | None = None
    ) -> tuple[NoisyOperation, int, float]:
        """Returns a randomly sampled NoisyOperation from the basis expansion.

        Args:
            random_state: Defines the seed for sampling if provided.
        """
        if not random_state:
            rng = np.random
        elif isinstance(random_state, int):
            rng = np.random.RandomState(random_state)  # type: ignore
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state  # type: ignore
        else:
            raise TypeError(
                "Arg `random_state` should be of type `np.random.RandomState` "
                f"or `int`, but was {type(random_state)}."
            )

        idx = rng.choice(len(self.coeffs), p=self.distribution)
        coeff, noisy_op = self.basis_expansion[idx]
        return noisy_op, int(np.sign(coeff)), coeff

    def __str__(self) -> str:
        lhs = str(self._ideal) + " = "
        rhs = ""
        for c, circ in zip(self.coeffs, self.noisy_operations):
            c_str = _format_coefficient(".3f", c)
            if c_str:
                if c_str[0] not in ["+", "-"]:
                    c_str = "+" + c_str
                if len(self._ideal.all_qubits()) == 1:
                    # Print single-qubit circuits horizontally
                    rhs += f"{c_str}*({circ!s})"
                else:
                    # Print multi-qubit circuits vertically
                    rhs += "\n\n" + f"{c_str}\n{circ!s}"
        # Handle special cases as in cirq.value.linear_dict._format_terms()
        if not rhs:
            rhs = f"{0:.3f}"
        # Remove "+" in the first term of a single-qubit representation
        if rhs[0] == "+":
            rhs = rhs[1:]
        # Remove "+" in the first term of a multi-qubit representation
        if rhs[0:3] == "\n\n+":
            rhs = "\n\n" + rhs[3:]
        return lhs + rhs

    def __eq__(self, other: Any) -> bool:
        """Checks if two representations are equivalent. This function returns
        True if the representations have the same ideal operation, the same
        coefficients and equivalent NoisyOperation(s) (same gates but not
        necessarily same channel_matrix since channel_matrix is optional).
        """
        if self.is_qubit_dependent != other.is_qubit_dependent:
            return False

        if self._native_type != other._native_type:
            return False

        if not self._ideal == other._ideal:
            return False

        if len(self.basis_expansion) != len(other.basis_expansion):
            return False

        for c_a, op_a in self.basis_expansion:
            found = False
            for c_b, op_b in other.basis_expansion:
                if op_a._circuit == op_b._circuit:
                    found = True
                    break
            if not found:
                return False
            if not np.isclose(c_a, c_b):
                return False
        return True
