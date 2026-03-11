# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from cirq import (
    CCNOT,
    CNOT,
    CZ,
    ISWAP,
    SWAP,
    Circuit,
    Gate,
    H,
    I,
    LineQubit,
    X,
    Y,
    Z,
    ops,
    unitary,
)

from mitiq.pec.channels import _circuit_to_choi, _operation_to_choi
from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)


def single_qubit_biased_noise_overhead(epsilon: float, eta: float) -> float:
    """Overhead calculation similar to that presented in
    :cite:`Temme_2017_PRL` and modified according to combined (biased) noise
    channel in :cite:`Strikis_2021_PRXQuantum`.
    """
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))
    eta1 = (a**2 + a * b - 2 * c**2) / (
        a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2
    )
    eta2 = -c / (a**2 + 2 * a * b + b**2 - 4 * c**2)
    eta3 = (-a * b - b**2 + 2 * c**2) / (
        a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2
    )

    return abs(eta1) + 2 * abs(eta2) + abs(eta3)


def two_qubit_biased_noise_overhead(epsilon: float, eta: float) -> float:
    """Overhead calculation similar to that presented in
    :cite:`Temme_2017_PRL` and modified according to combined (biased) noise
    channel in :cite:`Strikis_2021_PRXQuantum`.
    """
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))
    alpha_sq = (a**2 + a * b - 2 * c**2) ** 2 / (
        a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2
    ) ** 2
    alpha_beta = (
        (a**2 + a * b - 2 * c**2)
        * (-a * b - b**2 + 2 * c**2)
        / (a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2)
        ** 2
    )
    alpha_gamma = (
        -c
        * (a**2 + a * b - 2 * c**2)
        / (
            (a**2 + 2 * a * b + b**2 - 4 * c**2)
            * (a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2)
        )
    )
    beta_sq = (-a * b - b**2 + 2 * c**2) ** 2 / (
        a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2
    ) ** 2
    beta_gamma = (
        -c
        * (-a * b - b**2 + 2 * c**2)
        / (
            (a**2 + 2 * a * b + b**2 - 4 * c**2)
            * (a**3 + a**2 * b - a * b**2 - 4 * a * c**2 - b**3 + 4 * b * c**2)
        )
    )
    gamma_sq = c**2 / (a**2 + 2 * a * b + b**2 - 4 * c**2) ** 2
    overhead = (
        alpha_sq
        + 4 * abs(alpha_gamma)
        + 2 * abs(alpha_beta)
        + 4 * (gamma_sq)
        + 4 * abs(beta_gamma)
        + beta_sq
    )

    return overhead


@pytest.mark.parametrize("epsilon", [0, 0.1, 0.7])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("gate", [X, Y, Z, H])
def test_single_qubit_representation_norm(
    gate: Gate, epsilon: float, eta: float
):
    q = LineQubit(0)
    optimal_norm = single_qubit_biased_noise_overhead(epsilon, eta)
    norm = represent_operation_with_local_biased_noise(
        Circuit(gate(q)), epsilon, eta
    ).norm
    assert np.isclose(optimal_norm, norm)


@pytest.mark.parametrize("epsilon", [0, 0.1, 0.7])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("gate", (CZ, CNOT, ISWAP, SWAP))
def test_two_qubit_representation_norm(gate: Gate, epsilon: float, eta: float):
    qreg = LineQubit.range(2)
    optimal_norm = two_qubit_biased_noise_overhead(epsilon, eta)
    norm = represent_operation_with_local_biased_noise(
        Circuit(gate(*qreg)), epsilon, eta
    ).norm
    assert np.isclose(optimal_norm, norm)


def test_three_qubit_biased_noise_representation_error():
    q0, q1, q2 = LineQubit.range(3)
    with pytest.raises(ValueError):
        represent_operation_with_local_biased_noise(
            Circuit(CCNOT(q0, q1, q2)), 0.05, 10
        )


@pytest.mark.parametrize("epsilon", [0, 0.1, 0.7])
@pytest.mark.parametrize("eta", [0, 1, 1000])
@pytest.mark.parametrize("gate", [X, Y, Z, H, CZ, CNOT, ISWAP, SWAP])
def test_biased_noise_representation_with_choi(
    gate: Gate, epsilon: float, eta: float
):
    """Tests the representation by comparing exact Choi matrices."""
    qreg = LineQubit.range(gate.num_qubits())
    ideal_choi = _operation_to_choi(gate.on(*qreg))
    op_rep = represent_operation_with_local_biased_noise(
        Circuit(gate.on(*qreg)), epsilon, eta
    )
    choi_components = []

    # Define biased noise channel
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))

    mix = [
        (a, unitary(I)),
        (b, unitary(Z)),
        (c, unitary(X)),
        (c, unitary(Y)),
    ]

    for coeff, noisy_op in op_rep.basis_expansion:
        implementable_circ = noisy_op.circuit
        # Apply noise after each sequence.
        # NOTE: noise is not applied after each operation.
        biased_op = ops.MixedUnitaryChannel(mix).on_each(*qreg)
        implementable_circ.append(biased_op)
        sequence_choi = _circuit_to_choi(implementable_circ)
        choi_components.append(coeff * sequence_choi)
    combination_choi = np.sum(choi_components, axis=0)
    assert np.allclose(ideal_choi, combination_choi, atol=10**-6)
