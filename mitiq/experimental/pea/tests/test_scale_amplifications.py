# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mitiq.experimental.pea.scale_amplifications functions."""

import pytest
from cirq import (
    CNOT,
    Circuit,
    H,
    LineQubit,
    X,
    Y,
)

from mitiq.experimental.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_op_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.experimental.pea.scale_amplifications import (
    scale_circuit_amplifications,
    scale_representation,
)
from mitiq.pec import NoisyOperation, OperationRepresentation

qreg = LineQubit.range(2)
circ = Circuit([CNOT(*qreg), H(qreg[0]), Y(qreg[1]), CNOT(*qreg)])


@pytest.mark.parametrize("epsilon", [0.01, 0.02])
@pytest.mark.parametrize(
    "noise_model, noise_function",
    [
        (
            "local_depolarizing",
            amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
        ),
        (
            "global_depolarizing",
            amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
        ),
    ],
)
def test_scale_circuit_amplifications(epsilon, noise_model, noise_function):
    scale_factors = [1, 3, 5, 7]
    amp_fn = noise_function
    for s in scale_factors:
        amps = amp_fn(circ, s * epsilon)
        scaled_amps = scale_circuit_amplifications(
            circ, s, noise_model, epsilon
        )
        assert amps == scaled_amps


def test_noise_model_not_implemented_error():
    noise_model = "amplitude_damping"
    with pytest.raises(ValueError, match="Noise model not supported"):
        scale_circuit_amplifications(circ, 1.0, noise_model, 0.01)


def test_scale_positive_representation_matches_amplified_depolarizing_path():
    epsilon = 0.08
    gate = Circuit(Y(qreg[0]))
    scaled_factor = 2.5
    representation = amplify_noisy_op_with_global_depolarizing_noise(
        gate, epsilon
    )
    scaled_representation = scale_representation(representation, scaled_factor)
    expected = amplify_noisy_op_with_global_depolarizing_noise(
        gate, scaled_factor * epsilon
    )

    assert scaled_representation.coeffs == pytest.approx(expected.coeffs)


def test_scale_circuit_amplifications_with_representations_path():
    epsilon = 0.05
    gate = Circuit(Y(qreg[0]))
    base_representations = amplify_noisy_op_with_global_depolarizing_noise(
        gate, epsilon
    )
    scaled_factor = 2
    expected = scale_representation(base_representations, scaled_factor)
    scaled = scale_circuit_amplifications(
        gate, scaled_factor, representations=[base_representations]
    )

    assert len(scaled) == 1
    assert scaled[0].coeffs == pytest.approx(expected.coeffs)


def test_scale_signed_representation_uses_canonical_scaling():
    # 1.2 * O_+ -0.2 * O_- should become:
    # a_+ (1 - s*γ^-/γ^+) and a_- (1-s), with γ^+=1.2, γ^-=0.2.
    q = qreg[0]
    representation = OperationRepresentation(
        Circuit(X(q)),
        [NoisyOperation(Circuit(X(q))), NoisyOperation(Circuit(Y(q)))],
        [1.2, -0.2],
    )
    scale_factor = 2.0
    scaled = scale_representation(representation, scale_factor)

    assert scaled.coeffs == pytest.approx([0.8, 0.2])


def test_scale_signed_representation_enforces_canonical_limit():
    q = qreg[0]
    representation = OperationRepresentation(
        Circuit(X(q)),
        [NoisyOperation(Circuit(X(q))), NoisyOperation(Circuit(Y(q)))],
        [1.2, -0.2],
    )
    with pytest.raises(ValueError, match="canonical limit"):
        scale_representation(representation, 7.0)


def test_scale_circuit_amplifications_requires_mutually_exclusive_inputs():
    epsilon = 0.05
    local_noisy_representations = (
        amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
            circ, epsilon
        )
    )
    with pytest.raises(ValueError, match="Either ``noise_model``"):
        scale_circuit_amplifications(circ, 2, None, epsilon)

    with pytest.raises(ValueError, match="Either ``noise_model``"):
        scale_circuit_amplifications(
            circ,
            2,
            "local_depolarizing",
            epsilon,
            local_noisy_representations,
        )
