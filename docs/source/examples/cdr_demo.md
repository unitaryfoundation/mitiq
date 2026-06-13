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

```{tags} cdr, clifft, qiskit, cirq, intermediate
```

# Clifford data regression with the Clifft near-Clifford simulator

Clifford data regression (CDR) is an error mitigation technique that learns a
correction from noisy circuit outputs to their ideal values by training on
classically simulable near-Clifford circuits. In this tutorial, we run CDR
end-to-end on a hardware-native variational circuit, using the Clifft
near-Clifford simulator to provide the ideal training data needed for the
regression.

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

In addition to Mitiq and Cirq, this tutorial uses Qiskit (to build and transpile
a hardware-native circuit) and Clifft (to simulate the near-Clifford training
circuits used by CDR).

```{code-cell}
import cirq
import clifft
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile

from mitiq import Observable, PauliString, cdr
from mitiq.interface import convert_to_mitiq
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.zne.scaling import fold_global

N_QUBITS = 4
N_LAYERS = 6
NOISE = 0.02
```

## The circuit we want to run

Our target circuit is a hardware-efficient ansatz consisting of alternating
layers of single-qubit $R_y(\theta)$ rotations and a nearest-neighbor CNOT
ladder. We use $R_y$ rotations because they directly affect measurements of
$Z$-type observables: rotating a qubit around the $y$-axis changes its projection
onto the $z$-axis, making observables such as $\langle Z_0 Z_3 \rangle$ sensitive
to the variational parameters. The CNOT ladder then spreads those local rotations
into multi-qubit correlations across the device.

To connect this abstract circuit to real hardware, we transpile it into IBM's
native gate set `{rz, sx, cx}`. This decomposition exposes an important
structural feature: after transpilation, the arbitrary $R_y$ rotations have been
rewritten into sequences of `rz` and `sx` gates, while entanglement remains in
the `cx` gates. Since `sx` (the $\sqrt{X}$ gate) and `cx` are Clifford
operations, the only generic non-Clifford content left in the circuit resides in
the continuous `rz` rotation angles. That separation is the key idea behind this
tutorial: it is what allows CDR to generate near-Clifford training circuits by
modifying a small number of non-Clifford rotations, and it is precisely the
regime where a near-Clifford simulator such as Clifft can provide a substantial
advantage over general-purpose simulation methods.

```{code-cell}
def build_ansatz(n_qubits, n_layers, angles):
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(angles[layer][q], q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


rng = np.random.default_rng(42)
angles = rng.uniform(-np.pi / 2, np.pi / 2, size=(N_LAYERS, N_QUBITS))
qc = build_ansatz(N_QUBITS, N_LAYERS, angles)
native = transpile(qc, basis_gates=["rz", "sx", "cx"], optimization_level=1)
cirq_circuit = convert_to_mitiq(native)[0]

print("logical gate counts:", dict(qc.count_ops()))
print("native gate counts: ", dict(native.count_ops()))
```

## The observable, and the problem that noise causes

Our observable is the endpoint correlation $\langle Z_0 Z_3 \rangle$, which
measures how strongly the first and last qubits in the chain remain correlated
after the variational circuit. Because the ansatz contains multiple layers of
rotations and entangling gates, this quantity is sensitive to both the circuit
parameters and the propagation of quantum correlations across the device.

To illustrate the challenge that motivates CDR, we evaluate the same observable
using two executors: a noisy executor with depolarizing noise and an ideal
executor with noise disabled. Both return density matrices (hence the
`-> np.ndarray` type hints), allowing Mitiq to compute expectation values
directly. The output below shows the effect of noise clearly: the ideal
correlation is substantially suppressed, with the noisy value retaining only a
small fraction of the original signal. Recovering that lost correlation is
exactly the task that CDR will take on in the next section.

```{code-cell}
obs = Observable(PauliString("ZIIZ"))


def noisy(c: cirq.Circuit) -> np.ndarray:
    return compute_density_matrix(c, noise_level=(NOISE,))


def ideal(c: cirq.Circuit) -> np.ndarray:
    return compute_density_matrix(c, noise_level=(0.0,))


ideal_val = obs.expectation(cirq_circuit, ideal).real
noisy_val = obs.expectation(cirq_circuit, noisy).real
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
is a simulator designed for exactly this setting.

To use Clifft, we first translate the transpiled Cirq circuit into Clifft's
text-based circuit format. Since our native gate set consists only of `rz`, `sx`,
and `cx`, the conversion is straightforward. Two implementation details are worth
noting. First, Clifft expresses $R_Z$ rotations in half-turn units rather than
radians; fortunately, Cirq already stores this quantity as `gate.exponent` for
`ZPowGate`s. Second, Clifft and Cirq use opposite qubit-ordering conventions when
constructing statevectors, so after simulation we reverse the tensor axes to
match Cirq's ordering. These are purely representational differences—the
underlying quantum state is unchanged.

The fidelity check below confirms that the conversion is correct. A fidelity
essentially equal to one means that Clifft and Cirq produce the same noiseless
quantum state, giving us confidence that Clifft can serve as the simulator inside
the CDR workflow.

```{code-cell}
def to_clifft_text(circuit, qubits):
    idx = {q: i for i, q in enumerate(qubits)}
    lines = []
    for op in circuit.all_operations():
        g, qs = op.gate, op.qubits
        if isinstance(g, cirq.ZPowGate):
            lines.append(f"R_Z({g.exponent}) {idx[qs[0]]}")
        elif isinstance(g, cirq.XPowGate) and abs(g.exponent - 0.5) < 1e-9:
            lines.append(f"SQRT_X {idx[qs[0]]}")
        elif g == cirq.CNOT:
            lines.append(f"CX {idx[qs[0]]} {idx[qs[1]]}")
        else:
            raise ValueError(f"unhandled gate: {g!r}")
    return "\n".join(lines)


def clifft_statevector(circuit, qubits):
    prog = clifft.compile(to_clifft_text(circuit, qubits))
    state = clifft.State(
        peak_rank=prog.peak_rank,
        num_measurements=prog.num_measurements,
        num_detectors=prog.num_detectors,
        num_observables=prog.num_observables,
        num_exp_vals=prog.num_exp_vals,
    )
    clifft.execute(prog, state)  # required before reading when peak_rank > 0
    psi = clifft.get_statevector(prog, state)
    n = len(qubits)
    psi = psi.reshape([2] * n).transpose(range(n - 1, -1, -1)).reshape(-1)
    return psi


def clifft_sim(c: cirq.Circuit) -> np.ndarray:
    qubits = sorted(c.all_qubits())
    psi = clifft_statevector(c, qubits)
    return np.outer(psi, psi.conj())  # density matrix of a pure state


# sanity check: Clifft and cirq agree on the (noiseless) state
_qubits = sorted(cirq_circuit.all_qubits())
_psi_cirq = cirq.final_state_vector(
    cirq_circuit, qubit_order=_qubits, dtype=np.complex128
)
_fidelity = abs(np.vdot(clifft_statevector(cirq_circuit, _qubits), _psi_cirq)) ** 2
print(f"clifft vs cirq state fidelity: {_fidelity:.7f}")
```

## Running CDR

With all the pieces in place, we can run CDR in a single call. The arguments
mirror the conceptual ingredients of the method: the target circuit whose
expectation value we want to estimate, a noisy executor representing the device,
the observable of interest, and a simulator capable of evaluating the
near-Clifford training circuits. We also choose how many training circuits to
generate, what fraction of the original non-Clifford structure each training
circuit retains, and a random seed for reproducibility.

The important difference from earlier CDR examples is the simulator. Instead of
relying on a full density-matrix simulation to provide ideal training data, we
use Clifft, which is specialized for the near-Clifford circuits generated during
CDR's training phase. The output shows the learned correction in action: the
mitigated estimate moves substantially closer to the ideal value, reducing the
error by roughly an order of magnitude. The exact numbers will vary slightly with
circuit parameters, but the qualitative result is the same—the noisy observable
is systematically biased, and CDR learns a correction from classically simulable
cousins of the target circuit.

```{code-cell}
mitigated = cdr.execute_with_cdr(
    cirq_circuit,
    noisy,
    observable=obs,
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
noise_levels = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08]
noisy_curve, cdr_curve = [], []

for p in noise_levels:
    def noisy_p(c: cirq.Circuit) -> np.ndarray:
        return compute_density_matrix(c, noise_level=(p,))

    noisy_curve.append(obs.expectation(cirq_circuit, noisy_p).real)
    cdr_curve.append(
        cdr.execute_with_cdr(
            cirq_circuit,
            noisy_p,
            observable=obs,
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
    observable=obs,
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

In this tutorial, we ran Clifford data regression end-to-end on a hardware-native
variational circuit and used Clifft as the simulator responsible for evaluating
the near-Clifford training circuits. After transpilation exposed the circuit's
non-Clifford content as a collection of continuous `rz` rotations, CDR learned a
correction from noisy expectation values to ideal ones and substantially reduced
the observable error. We then examined how that correction behaves as noise
increases and compared standard CDR with its variable-noise extension.

There are several natural directions to explore from here. Increasing the circuit
depth or qubit count makes the near-Clifford structure increasingly important and
further motivates the use of specialized simulators such as Clifft. You can also
replace the synthetic depolarizing-noise executor with a real backend, experiment
with different observables, or vary the fraction of non-Clifford gates retained in
the training circuits. In each case, the central idea remains the same: learn from
circuits that are classically tractable, then use that knowledge to improve
estimates for circuits that are not.
