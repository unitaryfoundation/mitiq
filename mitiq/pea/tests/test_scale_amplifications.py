# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mitiq.pea.scale_amplifications functions."""

import pytest
from cirq import (
    CNOT,
    Circuit,
    H,
    LineQubit,
    Y,
)

from mitiq.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pea.scale_amplifications import (
    scale_circuit_amplifications,
)

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
