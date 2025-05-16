import numpy as np
import pytest
from cirq import (
    CNOT,
    Circuit,
    H,
    LineQubit,
    Y,
)

from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
)
from mitiq.pea.sample_amplifications import (
    sample_circuit_amplifications,
    scale_circuit_amplifications,
)

qreg = LineQubit.range(2)
circ = Circuit([CNOT(*qreg), H(qreg[0]), Y(qreg[1]), CNOT(*qreg)])


@pytest.mark.parametrize("epsilon", [0.01, 0.02])
def test_scale_circuit_amplifications(epsilon):
    amp_norms = []
    scaled_amp_norms = []
    scale_factors = [1, 3, 5, 7]
    for s in scale_factors:
        amps = amplify_noisy_ops_in_circuit_with_global_depolarizing_noise(
            circ, s * epsilon
        )
        amp_norms.append([amp.norm for amp in amps])
        scaled_amps = scale_circuit_amplifications(
            circ, s, "global_depolarizing", epsilon
        )
        scaled_amp_norms.append(
            [scaled_amp.norm for scaled_amp in scaled_amps]
        )

    assert np.allclose(amp_norms, scaled_amp_norms)


def sample_executor(circuit):
    return compute_density_matrix(circuit)[0, 0].real


@pytest.mark.parametrize("epsilon", [0.01, 0.02])
def test_sample_circuit_amplifications(epsilon):
    scale_factors = [1, 3, 5, 7]

    amp_values = sample_circuit_amplifications(
        circ,
        sample_executor,
        scale_factors,
        "local_depolarizing",
        epsilon,
    )

    ideal_exp_val = sample_executor(circ)
    errors = abs(np.array(amp_values) - ideal_exp_val)

    assert all(np.diff(errors) > 0)
