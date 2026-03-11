# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for finding optimal representations given a noisy basis."""

import functools
from typing import cast

import numpy as np
import numpy.typing as npt
from cirq import kraus
from scipy.optimize import LinearConstraint, minimize

from mitiq import QPROGRAM
from mitiq.interface import convert_to_mitiq
from mitiq.pec.channels import kraus_to_super
from mitiq.pec.types import NoisyOperation, OperationRepresentation
from mitiq.utils import matrix_to_vector


def minimize_one_norm(
    ideal_matrix: npt.NDArray[np.complex64],
    basis_matrices: list[npt.NDArray[np.complex64]],
    tol: float = 1.0e-8,
    initial_guess: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Returns the list of real coefficients :math:`[x_0, x_1, \dots]`,
    which minimizes :math:`\sum_j |x_j|` with the contraint that
    the following representation of the input ``ideal_matrix`` holds:

    .. math::
        \text{ideal_matrix} = x_0 A_0 + x_1 A_1 + \cdots

    where :math:`\{A_j\}` are the basis matrices, i.e., the elements of
    the input ``basis_matrices``.

    This function can be used to compute the optimal representation
    of an ideal superoperator (or Choi state) as a linear
    combination of real noisy superoperators (or Choi states).

    Args:
        ideal_matrix: The ideal matrix to represent.
        basis_matrices: The list of basis matrices.
        tol: The error tolerance for each matrix element
            of the represented matrix.
        initial_guess: Optional initial guess for the coefficients
            :math:`[x_0, x_1, \dots]`.

    Returns:
        The list of optimal coefficients :math:`[x_0, x_1, \dots]`.
    """

    # Map complex matrices to extended real matrices
    ideal_matrix_real = np.hstack(
        (np.real(ideal_matrix), np.imag(ideal_matrix))
    )
    basis_matrices_real = [
        np.hstack((np.real(mat), np.imag(mat))) for mat in basis_matrices
    ]

    # Express the representation constraint written in the docstring in the
    # form of a matrix multiplication applied to the x vector: A @ x == b.
    matrix_a = np.array(
        [matrix_to_vector(mat) for mat in basis_matrices_real]  # type: ignore[arg-type]
    ).T
    array_b = matrix_to_vector(ideal_matrix_real)  # type: ignore[arg-type]

    constraint = LinearConstraint(matrix_a, lb=array_b - tol, ub=array_b + tol)

    def one_norm(x: npt.NDArray[np.complex64]) -> float:
        return cast(float, np.linalg.norm(x, 1))

    if initial_guess is None:
        initial_guess = np.zeros(len(basis_matrices))

    result = minimize(one_norm, x0=initial_guess, constraints=constraint)

    if not result.success:
        raise RuntimeError("The search for an optimal representation failed.")

    return result.x


def find_optimal_representation(
    ideal_operation: QPROGRAM,
    noisy_operations: list[NoisyOperation],
    tol: float = 1.0e-8,
    initial_guess: npt.NDArray[np.float64] | None = None,
    is_qubit_dependent: bool = True,
) -> OperationRepresentation:
    r"""Returns the ``OperationRepresentation`` of the input ideal operation
    which minimizes the one-norm of the associated quasi-probability
    distribution.

    More precisely, it solve the following optimization problem:

    .. math::
        \min_{\eta_\alpha} \left\{\sum_\alpha |\eta_\alpha| \, : \,
        \mathcal G = \sum_\alpha \eta_\alpha \mathcal O_\alpha \right\}

    where :math:`\{\mathcal O_j\}` is the input basis of noisy operations,
    and :math:`\mathcal{G}` is the ideal operation to be represented.

    Args:
        ideal_operation: The ideal operation to represent.
        noisy_operations: The basis in which the ``ideal_operation``
            should be represented. Must be a list of ``NoisyOperation`` objects
            which are initialized with a numerical superoperator matrix.
        tol: The error tolerance for each matrix element
            of the represented operation.
        initial_guess: Optional initial guess for the coefficients
            :math:`\{ \eta_\alpha \}`.
        is_qubit_dependent: If True, the representation corresponds to the
            operation on the specific qubits defined in `ideal_operation`.
            If False, the representation is valid for the same gate even if
            acting on different qubits from those specified in
            `ideal_operation`.

    Returns: The optimal OperationRepresentation.
    """
    ideal_cirq_circuit, _ = convert_to_mitiq(ideal_operation)

    ops = list(ideal_cirq_circuit.all_operations())
    super_ops = [
        kraus_to_super(cast(list[npt.NDArray[np.complex64]], kraus(op)))
        for op in ops
    ]

    # Compute super operator of the full ideal_circuit
    ideal_matrix = functools.reduce(lambda a, b: a @ b, super_ops)

    try:
        basis_matrices = [
            noisy_op.channel_matrix for noisy_op in noisy_operations
        ]
    except ValueError as err:
        if str(err) == "The channel matrix is unknown.":
            raise ValueError(
                "The input noisy_basis should contain NoisyOperation objects"
                " which are initialized with a numerical superoperator matrix."
            )
        else:
            raise err  # pragma: no cover

    # Run numerical optimization problem
    quasi_prob_dist = minimize_one_norm(
        ideal_matrix,
        basis_matrices,
        tol=tol,
        initial_guess=initial_guess,
    )

    return OperationRepresentation(
        ideal_operation,
        noisy_operations,
        quasi_prob_dist.tolist(),
        is_qubit_dependent,
    )
