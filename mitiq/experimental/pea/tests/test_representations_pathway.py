# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Comprehensive tests for the ``representations`` pathway of PEA (#2936).

These tests scrutinize the convention-aware ``scale_representation`` primitive
across every quasi-probability representation mitiq can build (depolarizing
local/global, biased noise, amplitude damping, optimal/tomographic) plus a
hand-built sparse Pauli-Lindblad representation, and verify the canonical
noise-scaling math of Section D of arXiv:2108.02237 end to end.

All tests are cirq-only (no ``SUPPORTED_PROGRAM_TYPES`` parametrization) so
they are deterministic and do not depend on optional frontend extras.
"""

import cirq
import numpy as np
import pytest
from cirq import (
    CNOT,
    Circuit,
    DepolarizingChannel,
    H,
    LineQubit,
    X,
    Y,
    Z,
)

from mitiq.experimental.pea import construct_circuits, execute_with_pea
from mitiq.experimental.pea.amplifications.amplify_depolarizing import (
    amplify_noisy_op_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_global_depolarizing_noise,
    amplify_noisy_ops_in_circuit_with_local_depolarizing_noise,
)
from mitiq.experimental.pea.scale_amplifications import (
    scale_circuit_amplifications,
    scale_representation,
)
from mitiq.interface import convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.pec import (
    NoisyOperation,
    OperationRepresentation,
    represent_operation_with_global_depolarizing_noise,
    represent_operation_with_local_depolarizing_noise,
    represent_operations_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pec.channels import _circuit_to_choi, choi_to_super
from mitiq.pec.representations import (
    _represent_operation_with_amplitude_damping_noise,
    find_optimal_representation,
    represent_operation_with_local_biased_noise,
)
from mitiq.zne.inference import LinearFactory, RichardsonFactory

BASE_NOISE = 0.02
q0, q1 = LineQubit.range(2)
oneq_circ = Circuit(Z.on(q0), Z.on(q0))  # ideal expectation value 1.0
twoq_circ = Circuit(Y.on(q1), CNOT.on(q0, q1), Y.on(q1))


def depolarizing_executor(circuit, noise=BASE_NOISE):
    """Ground-state-projector expectation under depolarizing noise."""
    circuit, _ = convert_to_mitiq(circuit)
    return compute_density_matrix(
        circuit, noise_model_function=cirq.depolarize, noise_level=(noise,)
    )[0, 0].real


# --------------------------------------------------------------------------- #
# Hand-built sparse Pauli-Lindblad reps (mitiq has no SPL builder)            #
# --------------------------------------------------------------------------- #
def spl_single_generator_rep(ideal_circuit, qubit, generator, rate):
    r"""Inverse of a single-generator Pauli-Lindblad channel as a (signed)
    quasi-probability representation:  N^{-1} = c_I I + c_P P,
    c_I = (1 + e^{2r})/2,  c_P = (1 - e^{2r})/2 (< 0),  gamma = e^{2r}.
    """
    c_I = float((1 + np.exp(2 * rate)) / 2)
    c_P = float((1 - np.exp(2 * rate)) / 2)
    noisy_ops = [
        NoisyOperation(ideal_circuit),
        NoisyOperation(ideal_circuit + Circuit(generator(qubit))),
    ]
    return OperationRepresentation(ideal_circuit, noisy_ops, [c_I, c_P])


def spl_two_generator_rep(ideal_circuit, qubit, gen_a, gen_b, rate_a, rate_b):
    """Product of two single-generator inverses (4 signed terms)."""
    cIa = (1 + np.exp(2 * rate_a)) / 2
    cPa = (1 - np.exp(2 * rate_a)) / 2
    cIb = (1 + np.exp(2 * rate_b)) / 2
    cPb = (1 - np.exp(2 * rate_b)) / 2
    noisy_ops = [
        NoisyOperation(ideal_circuit),
        NoisyOperation(ideal_circuit + Circuit(gen_b(qubit))),
        NoisyOperation(ideal_circuit + Circuit(gen_a(qubit))),
        NoisyOperation(ideal_circuit + Circuit(gen_a(qubit), gen_b(qubit))),
    ]
    coeffs = [
        float(cIa * cIb),
        float(cIa * cPb),
        float(cPa * cIb),
        float(cPa * cPb),
    ]
    return OperationRepresentation(ideal_circuit, noisy_ops, coeffs)


def optimal_single_qubit_rep(noise_level=0.02):
    """A tomography-derived (numerically optimized) single-qubit rep."""
    q = LineQubit(0)
    ideal_op = Circuit(H(q))
    implementable = [Circuit(ideal_op)] + [
        Circuit([ideal_op, g(q)]) for g in (X, Y, Z)
    ]
    noisy_circuits = [
        c + Circuit(DepolarizingChannel(noise_level).on_each(q))
        for c in implementable
    ]
    super_ops = [choi_to_super(_circuit_to_choi(c)) for c in noisy_circuits]
    noisy_ops = [
        NoisyOperation(ideal, real)
        for ideal, real in zip(implementable, super_ops)
    ]
    return find_optimal_representation(ideal_op, noisy_ops, tol=1e-8)


# =========================================================================== #
# Group A — scale_representation math correctness                             #
# =========================================================================== #
@pytest.mark.parametrize("p", [0.01, 0.05])
@pytest.mark.parametrize("lam", [1.0, 2.0, 3.0])
def test_canonical_scaling_reproduces_paper_per_etas(p, lam):
    """Headline: canonical scaling of a signed depolarizing PEC rep equals the
    paper's closed-form per_etas eta_j(lambda) (Eq. per_etas / Section D)."""
    q = LineQubit(0)
    rep = represent_operation_with_global_depolarizing_noise(Circuit(X(q)), p)
    scaled = scale_representation(rep, lam)

    eps = 4 / 3 * p
    f = eps * (1 - lam) / (1 - eps)
    expected = [1 + 0.75 * f] + 3 * [-0.25 * f]
    assert scaled.coeffs == pytest.approx(expected)


@pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 3.0])
def test_scaling_preserves_unit_sum(lam):
    q = LineQubit(0)
    reps = [
        represent_operation_with_global_depolarizing_noise(
            Circuit(X(q)), 0.05
        ),
        represent_operation_with_local_depolarizing_noise(
            Circuit(CNOT(q0, q1)), 0.05
        ),
        represent_operation_with_local_biased_noise(Circuit(X(q)), 0.05, 1.0),
        _represent_operation_with_amplitude_damping_noise(Circuit(X(q)), 0.05),
        spl_single_generator_rep(Circuit(H(q)), q, Z, 0.05),
        spl_two_generator_rep(Circuit(H(q)), q, Z, X, 0.05, 0.03),
    ]
    for rep in reps:
        scaled = scale_representation(rep, lam)
        assert sum(scaled.coeffs) == pytest.approx(1.0)


def test_scale_one_gives_unit_norm_for_signed_rep():
    """At lambda=1 a signed rep collapses to the base channel Phi+ (norm 1)."""
    q = LineQubit(0)
    rep = represent_operation_with_global_depolarizing_noise(
        Circuit(X(q)), 0.05
    )
    assert scale_representation(rep, 1.0).norm == pytest.approx(1.0)


@pytest.mark.parametrize("lam", [0.5, 1.0, 1.5, 3.0])
def test_one_norm_law(lam):
    """gamma(lam) = gamma - lam(gamma-1) on [0,1]; = 1 on [1, lam_max]."""
    q = LineQubit(0)
    rep = represent_operation_with_global_depolarizing_noise(
        Circuit(X(q)), 0.05
    )
    gamma = rep.norm
    scaled = scale_representation(rep, lam)
    expected = gamma - lam * (gamma - 1) if lam <= 1 else 1.0
    assert scaled.norm == pytest.approx(expected)


def test_canonical_limit_raises_beyond_lambda_max():
    q = LineQubit(0)
    rep = represent_operation_with_global_depolarizing_noise(
        Circuit(X(q)), 0.05
    )
    with pytest.raises(ValueError, match="canonical limit"):
        scale_representation(rep, 100.0)


def test_positive_path_warns_past_upper_bound():
    """Deviation scaling of an all-positive rep can drive the identity coeff
    negative; we warn past s > 1/(1-a0)."""
    q = LineQubit(0)
    rep = amplify_noisy_op_with_global_depolarizing_noise(Circuit(X(q)), 0.05)
    # a0 = 1 - 0.05 = 0.95  =>  limit = 1/0.05 = 20
    with pytest.warns(UserWarning, match="no longer a valid probability"):
        scale_representation(rep, 50.0)


def test_sign_dispatch_tolerates_floating_point_dust():
    """A dust-magnitude negative coefficient (numerical noise for a term that
    is really zero) must not misroute an otherwise all-positive representation
    onto the canonical (signed) path; it should scale identically to the clean
    all-positive representation (deviation-from-identity)."""
    q = LineQubit(0)
    ideal = Circuit(X(q))
    ops = [
        NoisyOperation(ideal),
        NoisyOperation(Circuit([X(q), Y(q)])),
        NoisyOperation(Circuit([X(q), Z(q)])),
    ]
    clean = OperationRepresentation(ideal, ops, [0.98, 0.02, 0.0])
    dusted = OperationRepresentation(ideal, ops, [0.98, 0.02 + 1e-16, -1e-16])
    s = 2.0
    assert scale_representation(dusted, s).coeffs == pytest.approx(
        scale_representation(clean, s).coeffs, abs=1e-9
    )


def test_two_qubit_local_canonical_differs_from_factorized():
    """Canonical (single-parameter) scaling cannot reproduce factorized local
    (tensor) scaling: documented limitation of Section D for local noise."""
    p, lam = 0.05, 2.0
    rep2 = represent_operation_with_local_depolarizing_noise(
        Circuit(CNOT(q0, q1)), p
    )
    rep1 = represent_operation_with_global_depolarizing_noise(
        Circuit(X(LineQubit(0))), p
    )
    canon2 = np.array(scale_representation(rep2, lam).coeffs)
    s1 = np.array(scale_representation(rep1, lam).coeffs)
    factorized = np.kron(s1, s1)

    assert canon2.sum() == pytest.approx(1.0)
    assert len(canon2) == len(factorized) == 16
    assert not np.allclose(np.sort(canon2), np.sort(factorized))


# =========================================================================== #
# Group B — coverage across every representation type                         #
# =========================================================================== #
def _rep_catalog():
    q = LineQubit(0)
    return {
        "global_depol_1q": represent_operation_with_global_depolarizing_noise(
            Circuit(X(q)), BASE_NOISE
        ),
        "local_depol_1q": represent_operation_with_local_depolarizing_noise(
            Circuit(X(q)), BASE_NOISE
        ),
        "global_depol_2q": represent_operation_with_global_depolarizing_noise(
            Circuit(CNOT(q0, q1)), BASE_NOISE
        ),
        "local_depol_2q": represent_operation_with_local_depolarizing_noise(
            Circuit(CNOT(q0, q1)), BASE_NOISE
        ),
        "biased_1q": represent_operation_with_local_biased_noise(
            Circuit(X(q)), BASE_NOISE, 1.0
        ),
        "biased_2q": represent_operation_with_local_biased_noise(
            Circuit(CNOT(q0, q1)), BASE_NOISE, 1.0
        ),
        "amp_damping_1q": _represent_operation_with_amplitude_damping_noise(
            Circuit(X(q)), BASE_NOISE
        ),
        "optimal_1q": optimal_single_qubit_rep(BASE_NOISE),
        "amplify_positive_1q": amplify_noisy_op_with_global_depolarizing_noise(
            Circuit(X(q)), BASE_NOISE
        ),
        "spl_1gen": spl_single_generator_rep(Circuit(H(q)), q, Z, 0.05),
        "spl_2gen": spl_two_generator_rep(Circuit(H(q)), q, Z, X, 0.05, 0.03),
    }


@pytest.mark.parametrize("name", list(_rep_catalog()))
@pytest.mark.parametrize("scale_factor", [1.0, 2.0])
def test_scale_representation_invariants_all_types(name, scale_factor):
    rep = _rep_catalog()[name]
    is_signed = any(c < 0 for c in rep.coeffs)
    scaled = scale_representation(rep, scale_factor)

    assert isinstance(scaled, OperationRepresentation)
    assert len(scaled.coeffs) == len(rep.coeffs)
    assert len(scaled.noisy_operations) == len(rep.noisy_operations)
    assert sum(scaled.coeffs) == pytest.approx(1.0)
    assert np.isfinite(scaled.norm)
    if is_signed and scale_factor == 1.0:
        # signed -> canonical: lambda=1 is the base noisy channel (norm 1)
        assert scaled.norm == pytest.approx(1.0)


@pytest.mark.parametrize(
    "name", ["global_depol_1q", "local_depol_2q", "amp_damping_1q", "spl_1gen"]
)
def test_construct_circuits_accepts_each_rep_type(name):
    rep = _rep_catalog()[name]
    scaled = construct_circuits(
        rep.ideal,
        scale_factors=[1.0, 2.0],
        representations=[rep],
        num_samples=8,
        random_state=0,
    )
    sampled_circuits, signs, norms = scaled
    assert len(sampled_circuits) == 2  # one per scale factor
    assert all(len(c) == 8 for c in sampled_circuits)


# =========================================================================== #
# Group C — end-to-end mitigation with signed PEC representations             #
# =========================================================================== #
@pytest.mark.parametrize("circuit", [oneq_circ, twoq_circ])
def test_execute_with_pea_signed_reps_mitigate_noise(circuit):
    """End-to-end mitigation with signed PEC representations fed directly to
    PEA, against a depolarizing simulator."""
    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, BASE_NOISE
    )
    ideal = depolarizing_executor(circuit, noise=0.0)
    unmitigated = depolarizing_executor(circuit)

    mitigated = execute_with_pea(
        circuit,
        depolarizing_executor,
        scale_factors=[1.0, 1.5, 2.0],
        representations=reps,
        extrapolation_method=RichardsonFactory.extrapolate,
        random_state=7,
        num_samples=4000,
    )
    assert abs(mitigated - ideal) < abs(unmitigated - ideal)
    assert np.isclose(mitigated, ideal, atol=0.1)


def test_signed_and_noise_model_paths_both_accurate():
    """The signed-rep interface and the legacy noise_model interface both run
    end to end and produce accurate estimates (they use different scaling
    conventions, so need not be numerically identical). Uses the config the
    existing PEA mitigation test is known to converge with."""
    circuit = oneq_circ
    ideal = depolarizing_executor(circuit, noise=0.0)

    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, BASE_NOISE
    )
    via_reps = execute_with_pea(
        circuit,
        depolarizing_executor,
        scale_factors=[1.0, 1.2, 1.6],
        representations=reps,
        extrapolation_method=LinearFactory.extrapolate,
        random_state=101,
    )
    via_model = execute_with_pea(
        circuit,
        depolarizing_executor,
        scale_factors=[1.0, 1.2, 1.6],
        noise_model="local_depolarizing",
        epsilon=BASE_NOISE,
        extrapolation_method=LinearFactory.extrapolate,
        random_state=101,
    )
    assert np.isclose(via_reps, ideal, atol=0.1)
    assert np.isclose(via_model, ideal, atol=0.1)


def test_pea_with_learned_depolarizing_parameter_mitigates():
    """Hardware-agnostic workflow: build reps from a (learned-style) epsilon
    and mitigate, simulating a learned epsilon near the true noise level."""
    circuit = oneq_circ
    learned_epsilon = 0.02  # stand-in for learn_depolarizing_noise_parameter
    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, learned_epsilon
    )
    ideal = depolarizing_executor(circuit, noise=0.0)
    unmitigated = depolarizing_executor(circuit)
    mitigated = execute_with_pea(
        circuit,
        depolarizing_executor,
        scale_factors=[1.0, 1.5, 2.0],
        representations=reps,
        extrapolation_method=LinearFactory.extrapolate,
        random_state=11,
        num_samples=4000,
    )
    assert abs(mitigated - ideal) < abs(unmitigated - ideal)


# =========================================================================== #
# Group D — validation / robustness                                           #
# =========================================================================== #
def test_construct_circuits_empty_representations_raises():
    with pytest.raises(ValueError, match="non-empty"):
        construct_circuits(oneq_circ, scale_factors=[1, 2], representations=[])


def test_scale_circuit_amplifications_empty_representations_raises():
    with pytest.raises(ValueError, match="non-empty"):
        scale_circuit_amplifications(oneq_circ, 2.0, representations=[])


def test_scale_representation_rejects_nonpositive_scale():
    q = LineQubit(0)
    rep = represent_operation_with_global_depolarizing_noise(
        Circuit(X(q)), 0.05
    )
    with pytest.raises(ValueError, match="positive"):
        scale_representation(rep, 0.0)
    with pytest.raises(ValueError, match="positive"):
        scale_representation(rep, -1.0)


def test_scale_representation_rejects_nonfinite_scale():
    q = LineQubit(0)
    rep = represent_operation_with_global_depolarizing_noise(
        Circuit(X(q)), 0.05
    )
    with pytest.raises(ValueError, match="finite"):
        scale_representation(rep, float("inf"))


def test_all_positive_without_identity_term_raises():
    q = LineQubit(0)
    rep = OperationRepresentation(
        Circuit(X(q)),
        [
            NoisyOperation(Circuit([X(q), Y(q)])),
            NoisyOperation(Circuit([X(q), Z(q)])),
        ],
        [0.5, 0.5],
    )
    with pytest.raises(ValueError, match="identity"):
        scale_representation(rep, 2.0)


def test_construct_circuits_rejects_both_inputs():
    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        oneq_circ, BASE_NOISE
    )
    with pytest.raises(ValueError, match="Either ``noise_model``"):
        construct_circuits(
            oneq_circ,
            scale_factors=[1, 2],
            noise_model="local_depolarizing",
            epsilon=BASE_NOISE,
            representations=reps,
        )


def test_construct_circuits_requires_some_input():
    with pytest.raises(ValueError, match="Either ``noise_model``"):
        construct_circuits(oneq_circ, scale_factors=[1, 2])


def test_construct_circuits_rejects_empty_scale_factors():
    q = LineQubit(0)
    rep = represent_operation_with_global_depolarizing_noise(
        Circuit(X(q)), 0.05
    )
    with pytest.raises(ValueError, match="scale_factors"):
        construct_circuits(
            Circuit(X(q)), scale_factors=[], representations=[rep]
        )


def test_noise_model_path_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="noise_model is deprecated"):
        construct_circuits(
            oneq_circ,
            scale_factors=[1, 2],
            noise_model="local_depolarizing",
            epsilon=BASE_NOISE,
            num_samples=4,
        )


# =========================================================================== #
# Group E — multi-qubit noise_model-equivalence scope (criterion 2)
# =========================================================================== #
def test_twoqubit_global_depolarizing_representations_equal_noise_model():
    """2-qubit GLOBAL depolarizing is a single-parameter channel, so the
    representations path equals the legacy noise_model path EXACTLY at every
    scale factor."""
    circ = Circuit(CNOT(q0, q1))
    eps = 0.05
    base = amplify_noisy_ops_in_circuit_with_global_depolarizing_noise(
        circ, eps
    )
    for s in (1.0, 2.0, 3.0):
        via_reps = scale_circuit_amplifications(circ, s, representations=base)
        via_model = scale_circuit_amplifications(
            circ, s, noise_model="global_depolarizing", epsilon=eps
        )
        assert len(via_reps) == len(via_model) == 1
        assert via_reps[0].coeffs == pytest.approx(via_model[0].coeffs)


def test_twoqubit_local_depolarizing_representations_differ_from_noise_model():
    """2-qubit LOCAL depolarizing is a tensor product (multi-parameter), so the
    single-parameter representations path canNOT reproduce the factorized
    noise_model rebuild. This regression-locks the documented limitation
    (acceptance criterion 2 holds only for global + single-qubit local)."""
    circ = Circuit(CNOT(q0, q1))
    eps, s = 0.05, 2.0
    base = amplify_noisy_ops_in_circuit_with_local_depolarizing_noise(
        circ, eps
    )
    via_reps = scale_circuit_amplifications(circ, s, representations=base)[
        0
    ].coeffs
    via_model = scale_circuit_amplifications(
        circ, s, noise_model="local_depolarizing", epsilon=eps
    )[0].coeffs
    max_diff = max(abs(a - b) for a, b in zip(via_reps, via_model))
    assert max_diff == pytest.approx(0.005, abs=5e-4)


def test_num_samples_uses_max_scaled_norm():
    """num_samples is deduced from the largest scaled one-norm across the
    requested scale factors; a sub-unit (noise-reduction) factor with
    one-norm > 1 raises the budget above the scale-1.0-only value."""
    q = LineQubit(0)
    ideal = Circuit(X(q))
    rep = represent_operation_with_global_depolarizing_noise(ideal, 0.05)
    precision = 0.2
    scale_factors = [0.5, 1.0, 2.0]

    max_norm = max(scale_representation(rep, s).norm for s in scale_factors)
    expected = int((max_norm / precision) ** 2)

    sampled, _, _ = construct_circuits(
        ideal,
        scale_factors=scale_factors,
        representations=[rep],
        precision=precision,
        random_state=0,
    )
    assert len(sampled[0]) == expected
    # strictly more than the (incorrect) scale-1.0-only deduction (norm == 1)
    assert expected > int((1.0 / precision) ** 2)


def test_scale_circuit_amplifications_rejects_nonfinite_scale():
    q = LineQubit(0)
    ideal = Circuit(X(q))
    rep = represent_operation_with_global_depolarizing_noise(ideal, 0.05)

    with pytest.raises(ValueError, match="finite"):
        scale_circuit_amplifications(
            ideal,
            float("inf"),
            representations=[rep],
        )
    with pytest.raises(ValueError, match="finite"):
        scale_circuit_amplifications(
            ideal,
            float("inf"),
            noise_model="global_depolarizing",
            epsilon=0.05,
        )
