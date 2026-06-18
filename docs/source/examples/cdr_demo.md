---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{tags} cdr, clifft, cirq, intermediate
```

# Clifford data regression with the Clifft near-Clifford simulator

Clifford data regression (CDR) is an error mitigation technique that learns a
correction from noisy circuit outputs to their ideal values by training on
classically simulable near-Clifford circuits. In this tutorial, we run CDR
end-to-end on a layered variational circuit, using the
[Clifft](https://github.com/unitaryfoundation/clifft) near-Clifford simulator to
provide the ideal training data needed for the regression.

CDR occupies a different niche from other error mitigation methods. Zero-noise
extrapolation (ZNE) estimates the zero-noise answer by executing the same circuit
at several amplified noise levels and extrapolating back to the noiseless limit,
while probabilistic error cancellation (PEC) reconstructs ideal expectation
values through quasi-probability sampling. CDR instead learns a noisy-to-ideal
map from a collection of classically tractable training circuits. See the
[CDR guide](../guide/cdr.md) for a detailed introduction and the original work by
Czarnik et al. {cite}`Czarnik_2021_Quantum` for the underlying method. The key
observation we will exploit is that these training circuits are near-Clifford by
construction, making a specialized near-Clifford simulator such as Clifft a
natural fit for generating the reference data that CDR requires.

## Setup

In addition to Mitiq and Cirq, this tutorial uses Clifft to simulate the
near-Clifford training circuits used by CDR.

```{code-cell}
from functools import reduce

import cirq
import clifft
import matplotlib.pyplot as plt
import numpy as np

from mitiq import cdr
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.zne.scaling import fold_global

N_QUBITS = 4
N_LAYERS = 4
NOISE = 0.02
```

## The circuit we want to run

Our target circuit is a hardware-efficient ansatz: alternating layers of
single-qubit rotations and a nearest-neighbor CNOT ladder. We build each rotation
as $H \cdot R_z(\theta) \cdot H$ — equivalent to an $R_x(\theta)$ rotation, but
written so the only non-Clifford gate is the $R_z$, as CDR requires. (A bare
$R_x(\theta)$ is itself a non-Clifford gate that is not a $Z$-axis rotation, so it
would not satisfy that constraint.) This rotates the qubit so that measurements of
$Z$-type observables such as $\langle Z_0 Z_3 \rangle$ depend on the variational
angle $\theta$. The CNOT ladder then spreads those local rotations into
multi-qubit correlations across the register.

CDR requires the circuit to be compiled into a gateset whose only non-Clifford
gates are single-qubit $R_z$ rotations (see the [CDR guide](../guide/cdr.md)). We
build the ansatz directly in such a gateset: the Hadamards and CNOTs are Clifford,
so the only non-Clifford content lives in the continuous `rz` rotation angles.
That separation is the key idea behind this tutorial — it is what lets CDR
generate near-Clifford training circuits by adjusting a small number of
non-Clifford rotations, and it is precisely the regime where a near-Clifford
simulator such as Clifft provides an advantage over general-purpose simulation.

```{code-cell}
def build_ansatz(qubits, n_layers, angles):
    circuit = cirq.Circuit()
    for layer in range(n_layers):
        for i, q in enumerate(qubits):
            circuit.append([cirq.H(q), cirq.rz(angles[layer][i]).on(q), cirq.H(q)])
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


qubits = cirq.LineQubit.range(N_QUBITS)
rng = np.random.default_rng(42)
angles = rng.uniform(-np.pi / 2, np.pi / 2, size=(N_LAYERS, N_QUBITS))
cirq_circuit = build_ansatz(qubits, N_LAYERS, angles)

n_total = len(list(cirq_circuit.all_operations()))
n_rz = sum(
    1 for op in cirq_circuit.all_operations() if isinstance(op.gate, cirq.ZPowGate)
)
print(f"{n_total} gates total; {n_rz} are non-Clifford Rz rotations")
print("(the H and CNOT gates are Clifford)")
```

## The observable, and the problem that noise causes

Our observable is the endpoint correlation $\langle Z_0 Z_3 \rangle$, which
measures how strongly the first and last qubits in the chain remain correlated
after the variational circuit. Because the ansatz contains multiple layers of
rotations and entangling gates, this quantity is sensitive to both the circuit
parameters and the propagation of quantum correlations across the device.

To illustrate the challenge that motivates CDR, we evaluate the same observable
two ways: a noisy executor with depolarizing noise, and an ideal executor with
noise disabled. Each returns the expectation value $\langle Z_0 Z_3 \rangle$
directly (hence the `-> float` type hints) — the form Mitiq's CDR uses when the
executors compute the observable themselves rather than being handed one
separately. The output shows the effect of noise clearly: the ideal correlation
is substantially suppressed, with the noisy value retaining only part of the
original signal. Recovering that lost correlation is exactly the task that CDR
will take on in the next section.

```{code-cell}
# Z0 Z3 (Z on qubits 0 and 3) is diagonal, so its expectation value is a
# parity-weighted sum of the basis-state populations — no matrix multiply needed.
z03_diag = reduce(
    np.kron, [np.array([1.0, -1.0]), np.ones(2), np.ones(2), np.array([1.0, -1.0])]
)


def noisy(c: cirq.Circuit) -> float:
    rho = compute_density_matrix(c, noise_level=(NOISE,))
    return float(np.sum(np.real(np.diagonal(rho)) * z03_diag))


def ideal(c: cirq.Circuit) -> float:
    rho = compute_density_matrix(c, noise_level=(0.0,))
    return float(np.sum(np.real(np.diagonal(rho)) * z03_diag))


ideal_val = ideal(cirq_circuit)
noisy_val = noisy(cirq_circuit)
print(f"ideal <Z0 Z3> = {ideal_val:+.4f}")
print(f"noisy <Z0 Z3> = {noisy_val:+.4f}")
print(f"error         = {abs(noisy_val - ideal_val):.4f}")
```

## A near-Clifford simulator: Clifft

CDR relies on a collection of training circuits that are near-Clifford by
construction: most of their operations are Clifford gates, with only a small
amount of non-Clifford content remaining. That structure makes a near-Clifford
simulator a natural choice for generating the reference values required during
training. Rather than using a general-purpose statevector or density-matrix
simulator for every training circuit, we can exploit the fact that these circuits
live in a regime where specialized algorithms are particularly effective. Clifft
is a simulator designed for exactly this setting {cite}`Chase_2026_Clifft`
([arXiv:2604.27058](https://arxiv.org/abs/2604.27058)).

To use Clifft, we translate the Cirq circuit into Clifft's text-based format — our
gateset is just `H`, `Rz`, and `CNOT`, so the conversion is direct — and append an
`EXP_VAL` probe that asks Clifft for $\langle Z_0 Z_3 \rangle$ directly. This
matters: materializing the full statevector or density matrix would cost $2^n$ and
defeat the purpose of a near-Clifford simulator, whereas Clifft evaluates the
expectation value natively, so the same code generalizes cleanly to far larger
circuits. One conversion detail: Clifft expresses $R_z$ rotations in half-turns,
which Cirq already stores in `gate.exponent`.

The check below confirms Clifft's `EXP_VAL` result matches the exact value from a
density-matrix simulation.

```{code-cell}
def to_clifft_text(circuit, qubits):
    idx = {q: i for i, q in enumerate(qubits)}
    lines = []
    for op in circuit.all_operations():
        g, qs = op.gate, op.qubits
        if isinstance(g, cirq.HPowGate) and abs(g.exponent - 1) < 1e-9:
            lines.append(f"H {idx[qs[0]]}")
        elif isinstance(g, cirq.ZPowGate):
            lines.append(f"R_Z({g.exponent}) {idx[qs[0]]}")
        elif g == cirq.CNOT:
            lines.append(f"CX {idx[qs[0]]} {idx[qs[1]]}")
        else:
            raise ValueError(f"unhandled gate: {g!r}")
    return "\n".join(lines)


def clifft_sim(c: cirq.Circuit) -> float:
    qubits = sorted(c.all_qubits())
    text = to_clifft_text(c, qubits) + "\nEXP_VAL Z0*Z3"
    result = clifft.sample(clifft.compile(text), 1)
    return float(result.exp_vals[0, 0])  # exact expectation, no statevector


print(f"clifft EXP_VAL <Z0 Z3> = {clifft_sim(cirq_circuit):+.6f}")
print(f"exact (density matrix) = {ideal(cirq_circuit):+.6f}")
```

## Running CDR

With all the pieces in place, we can run CDR in a single call. The arguments
mirror the conceptual ingredients of the method: the target circuit whose
expectation value we want to estimate, a noisy executor representing the device,
and a simulator capable of evaluating the near-Clifford training circuits.
(Because our executors return $\langle Z_0 Z_3 \rangle$ directly, we don't pass a
separate observable.) We also choose how many training circuits to generate, what
fraction of the original non-Clifford structure each retains, and a random seed
for reproducibility.

The important difference from earlier CDR examples is the simulator. Instead of
relying on a full density-matrix simulation to provide ideal training data, we use
Clifft, which is specialized for the near-Clifford circuits generated during CDR's
training phase. The output shows the learned correction in action: the mitigated
estimate moves substantially closer to the ideal value, reducing the error by
several times. The exact numbers will vary with circuit parameters, but the
qualitative result is the same—the noisy observable is systematically biased, and
CDR learns a correction from classically simulable cousins of the target circuit.

```{code-cell}
mitigated = cdr.execute_with_cdr(
    cirq_circuit,
    noisy,
    simulator=clifft_sim,
    num_training_circuits=100,
    fraction_non_clifford=0.2,
    random_state=0,
).real
print(f"ideal     = {ideal_val:+.4f}")
print(f"noisy     = {noisy_val:+.4f}")
print(f"CDR       = {mitigated:+.4f}")
print(f"error: {abs(noisy_val - ideal_val):.4f} -> {abs(mitigated - ideal_val):.4f}")
```

## How well does CDR hold up as noise grows?

A single noise level only tells part of the story. To understand when CDR
succeeds and where it begins to struggle, we repeat the experiment while sweeping
the depolarizing noise strength and keeping the circuit fixed. For each noise
level we record the raw noisy expectation value and the corresponding
CDR-mitigated estimate.

The resulting plot highlights both the strengths and limitations of the method.
The ideal expectation value remains constant because the underlying circuit does
not change, while the noisy estimate is steadily driven toward zero as
decoherence destroys the correlation between the endpoint qubits. CDR tracks the
ideal value remarkably well at low and moderate noise strengths, recovering much
of the lost signal. As the noise becomes more severe, the quality of the learned
correction gradually degrades. This behavior is expected: CDR is not removing
noise from the quantum device, but rather learning a model of how noise distorts
nearby circuits. When that distortion becomes too large or too nonlinear, the
learned correction becomes less predictive. The graceful breakdown visible in the
plot is therefore an important part of understanding the method, not a failure of
it.

```{code-cell}
noise_levels = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08]
noisy_curve, cdr_curve = [], []

for p in noise_levels:
    def noisy_p(c: cirq.Circuit) -> float:
        rho = compute_density_matrix(c, noise_level=(p,))
        return float(np.sum(np.real(np.diagonal(rho)) * z03_diag))

    noisy_curve.append(noisy_p(cirq_circuit))
    cdr_curve.append(
        cdr.execute_with_cdr(
            cirq_circuit,
            noisy_p,
            simulator=clifft_sim,
            num_training_circuits=100,
            fraction_non_clifford=0.2,
            random_state=0,
        ).real
    )

plt.figure(figsize=(7, 5))
plt.axhline(ideal_val, ls="--", color="green", label="ideal (noiseless)")
plt.plot(noise_levels, noisy_curve, "o-", color="crimson", label="noisy")
plt.plot(noise_levels, cdr_curve, "s-", color="steelblue",
         label="CDR-mitigated (Clifft simulator)")
plt.xlabel("depolarizing noise level")
plt.ylabel(r"$\langle Z_0 Z_3 \rangle$")
plt.legend()
plt.show()
```

CDR assumes the relationship between noisy and ideal expectation values is simple
enough for the regression to capture — here, a linear fit. That assumption holds
well in the regime used above, but it is not guaranteed. For deeper circuits,
strong noise can compress the noisy values into a narrow range, making the linear
fit steep enough to *overshoot* the true value even when the target lies inside
the training data. Choosing a circuit and noise range where the linear
approximation is valid is part of applying CDR well.

## Variable-noise CDR (vnCDR)

Variable-noise CDR (vnCDR) {cite}`Lowe_2021_PRR` extends the basic idea by
providing the regression with additional training data collected at multiple
noise strengths. Instead of observing each training circuit only at the native
device noise level, vnCDR evaluates it at several amplified noise levels and
learns from the resulting family of noisy outputs. In Mitiq, we generate these
amplified noise levels by folding gates with `fold_global`, which
deterministically increases the circuit depth while preserving the ideal unitary
action.

One subtle but important detail is that only the noisy executor sees the folded
circuits. The simulator still evaluates the original, unscaled training circuits,
since its role is to provide the ideal reference values. In this example, vnCDR
typically performs comparably to standard CDR, with both methods dramatically
outperforming the raw noisy estimate. The additional training data often becomes
more valuable for deeper circuits or stronger noise regimes, where a single
linear correction is no longer flexible enough to capture the relationship
between noisy and ideal observables.

```{code-cell}
vncdr = cdr.execute_with_cdr(
    cirq_circuit,
    noisy,
    simulator=clifft_sim,
    num_training_circuits=100,
    fraction_non_clifford=0.2,
    random_state=0,
    scale_factors=(1, 2, 3),
    scale_noise=fold_global,
).real
print(f"plain CDR error: {abs(mitigated - ideal_val):.4f}")
print(f"vnCDR error:     {abs(vncdr - ideal_val):.4f}")
```

## Conclusion

In this tutorial, we ran Clifford data regression end-to-end on a layered
variational circuit and used Clifft as the simulator responsible for evaluating
the near-Clifford training circuits. Because we built the circuit so that its only
non-Clifford gates are `rz` rotations, CDR could learn a correction from noisy
expectation values to ideal ones and substantially reduce the observable error.
Reading the expectation value out of Clifft with `EXP_VAL` — rather than
materializing a statevector — is what lets this approach scale to circuits far
larger than this demo. We then examined how the correction behaves as noise
increases and compared standard CDR with its variable-noise extension.

There are several natural directions to explore from here. Increasing the circuit
depth or qubit count makes the near-Clifford structure increasingly important and
further motivates the use of specialized simulators such as Clifft. You can also
replace the synthetic depolarizing-noise executor with a real backend, experiment
with different observables, or vary the fraction of non-Clifford gates retained in
the training circuits. In each case, the central idea remains the same: learn from
circuits that are classically tractable, then use that knowledge to improve
estimates for circuits that are not.
