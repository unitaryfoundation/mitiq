---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{tags} pea, pec, cirq, intermediate
```

# Probabilistic error amplification (PEA) with randomized benchmarking circuits

This notebook is a worked end-to-end example of probabilistic error amplification (PEA), the technique used in *Kim et al. Nature (2023)* {cite}`Kim_2023_Nature` to compute expectation values on a 127-qubit device. It complements the [PEA section of the user guide](../guide/pea.md), which documents the available options in detail; here we focus on running the full workflow on a small example and understanding each step.

PEA is a hybrid of two other techniques covered in this documentation:

- [Zero-noise extrapolation (ZNE)](../guide/zne.md) runs a circuit at several *increased* noise levels and extrapolates the measured expectation values back to the zero-noise limit. It needs no knowledge of the noise, but the amplification (e.g. unitary folding) is only a proxy for how the real noise acts.
- [Probabilistic error cancellation (PEC)](../guide/pec.md) uses a characterized noise model to represent each ideal operation $\mathcal{G}_i$ as a quasi-probability combination of implementable noisy operations $\mathcal{O}_\alpha$,

  $$
  \mathcal{G}_i = \sum_{\alpha} \eta_{i, \alpha} \mathcal{O}_{\alpha},
  $$

  and samples circuits from this decomposition (with signs) to *cancel* the noise in expectation. The price is a sampling overhead $\gamma^2$, where $\gamma = \prod_i \sum_\alpha |\eta_{i,\alpha}|$ collects one factor for every operation of the circuit and therefore grows exponentially with circuit size.

PEA borrows the quasi-probability machinery from PEC and the extrapolation endgame from ZNE: instead of using the representations to cancel the noise, it *rescales* them so that sampled circuits realize the actual noise channel amplified by a gain $s$, evaluates the expectation value at several $s \geq 1$, and extrapolates to $s \to 0$. The result is a physically faithful amplification of the characterized noise, at a sampling cost far milder than PEC's, as we will verify below.

The rescaling of the representations follows the canonical noise scaling prescription of *Mari et al. PRA (2021)* {cite}`Mari_2021_PRA` (Sec. VI D): the positive and negative coefficients of each representation, with volumes $\gamma^+$ and $\gamma^-$ (satisfying $\gamma^+ - \gamma^- = 1$), are rescaled separately,

$$
\eta^+_{\alpha} \rightarrow \eta^+_{\alpha}\, \frac{\gamma^+ - s\, \gamma^-}{\gamma^+},
\qquad
\eta^-_{\alpha} \rightarrow \eta^-_{\alpha}\, (1 - s).
$$

This calibrates the $s$ axis so that $s = 0$ leaves the representation unchanged — ordinary PEC, which cancels the noise — while at $s = 1$ the negative coefficients vanish and the representation becomes a genuine probability distribution over implementable circuits with one-norm $1$. For $s > 1$ real extra noise is injected. Extrapolating the measured curve to $s = 0$ therefore genuinely targets the ideal (noiseless) expectation value.

```{note}
PEA lives in `mitiq.experimental.pea`. Experimental modules are not covered by Mitiq's semantic versioning guarantees and their API may change without notice; importing the module emits a `FutureWarning` (visible in the output of the setup cell below) to make this explicit.
```

## Setup

```{code-cell} ipython3
import cirq
import matplotlib.pyplot as plt
import numpy as np

from mitiq import Executor, benchmarks
from mitiq.experimental import pea
from mitiq.experimental.pea.scale_amplifications import scale_representations
from mitiq.pec.representations import (
    represent_operations_in_circuit_with_local_depolarizing_noise,
)
from mitiq.zne.inference import LinearFactory, RichardsonFactory

plt.rcParams.update({"font.family": "serif", "font.size": 15})
```

We fix the parameters of the experiment: the baseline noise strength `epsilon`, the noise scale factors at which the expectation value is measured, and the number of circuits sampled from the quasi-probability decomposition at each scale factor.

```{code-cell} ipython3
epsilon = 0.005
scale_factors = [1.0, 1.6, 2.2]
num_samples = 2_000
random_state = 1
```

## Problem setup

We use a two-qubit randomized benchmarking (RB) circuit as the benchmark. RB circuits compile to the identity, so the ideal probability of measuring the all-zeros state — the expectation value of the observable $|00\rangle\langle 00|$ — is exactly $1$, which makes the error of any estimate unambiguous. Two qubits keep the docs build fast while still involving entangling gates.

```{code-cell} ipython3
circuit = benchmarks.generate_rb_circuits(
    n_qubits=2, num_cliffords=1, return_type="cirq", seed=1
)[0]
print(circuit)
```

The circuit generation is seeded for reproducibility; the concrete Clifford sequence (and hence the exact numbers below) can still vary across library versions.

As the executor we use Cirq's density matrix simulator, adding depolarizing noise of strength `epsilon` with `circuit.with_noise`, and return the population of the all-zeros state. This is a *controlled demonstration*: the executor's noise closely matches the depolarizing model that the representations below assume, so the experiment isolates the method itself. The match is close but not exact — `with_noise` applies a channel to every qubit after every moment, including idle qubits, and the Pauli-correction gates sampled from the representations acquire depolarizing noise of their own; the final section catalogues these residual differences. On real hardware the noise model would instead have to be learned from the device (see the final section).

```{code-cell} ipython3
def execute(circ: cirq.Circuit) -> float:
    """Simulates the circuit with depolarizing noise and returns the
    probability of the all-zeros state."""
    noisy_circ = circ.with_noise(cirq.depolarize(p=epsilon))
    rho = cirq.DensityMatrixSimulator().simulate(noisy_circ).final_density_matrix
    return rho[0, 0].real


ideal_value = cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix[0, 0].real
noisy_value = execute(circuit)

print(f"Ideal expectation value:       {ideal_value:.4f}")
print(f"Unmitigated expectation value: {noisy_value:.4f}")
print(f"Unmitigated error:             {abs(ideal_value - noisy_value):.4f}")
```

## Building the noise representations

PEA starts from the same object as PEC: one {class}`.OperationRepresentation` per unique operation of the circuit, expressing the ideal operation as a signed combination of noisy implementable operations. Since our executor applies local depolarizing noise, we can build these representations analytically.

```{code-cell} ipython3
representations = represent_operations_in_circuit_with_local_depolarizing_noise(
    circuit, epsilon
)
print(f"Number of unique operations represented: {len(representations)}")
```

The structure of a representation depends on the number of qubits the operation acts on. A single-qubit representation has one positive coefficient on the bare noisy operation and three small negative coefficients on Pauli-corrected operations. The two-qubit representation — this circuit contains exactly one unique two-qubit operation — is the tensor product of two such single-qubit inverse channels, so it additionally carries nine *positive* double-Pauli terms (products of two negative coefficients): 16 terms in total, ten positive and six negative. The double-Pauli coefficients are of order $\epsilon^2$, too small to appear in the three-decimal printout below, but they matter for understanding the noise scaling later on.

```{code-cell} ipython3
two_qubit_rep = next(
    rep for rep in representations if len(rep.ideal.all_qubits()) == 2
)
print(two_qubit_rep)

coeffs = np.array(two_qubit_rep.coeffs)
print(
    f"Terms: {len(coeffs)} "
    f"(positive: {np.sum(coeffs > 0)}, negative: {np.sum(coeffs < 0)})"
)
```

The one-norm $\gamma_i = \sum_\alpha |\eta_{i,\alpha}|$ of each representation determines PEC's sampling overhead: the variance of the PEC estimator is amplified by $\gamma^2$, where $\gamma$ multiplies one factor $\gamma_i$ for every operation *occurrence* in the circuit — a repeated operation contributes its factor each time it appears, which is what makes the overhead exponential in circuit size.

```{code-cell} ipython3
gamma = 1.0
for op in circuit.all_operations():
    rep = next(r for r in representations if r.ideal == cirq.Circuit(op))
    gamma *= rep.norm

arities = [len(op.qubits) for op in circuit.all_operations()]
print(
    f"Operation occurrences: {len(arities)} "
    f"({arities.count(1)} single-qubit, {arities.count(2)} two-qubit)"
)
for num_qubits in (1, 2):
    norms = sorted(
        rep.norm
        for rep in representations
        if len(rep.ideal.all_qubits()) == num_qubits
    )
    print(f"{num_qubits}-qubit representations: {len(norms)}, one-norms:", np.round(norms, 4))
print(f"Circuit one-norm gamma:        {gamma:.4f}")
print(f"PEC sampling overhead gamma^2: {gamma**2:.4f}")
```

Even for this small circuit — 23 single-qubit and 4 two-qubit operations — PEC would pay a variance penalty of about $1.86$, and the penalty grows exponentially with circuit size. PEA avoids it entirely in the amplification band, as we verify next.

## PEA in a single call

The high-level entry point is {func}`mitiq.experimental.pea.pea.execute_with_pea`. We pass the representations built above together with the scale factors and an extrapolation method from {mod}`mitiq.zne.inference`:

```{code-cell} ipython3
pea_value = pea.execute_with_pea(
    circuit,
    execute,
    scale_factors=scale_factors,
    extrapolation_method=LinearFactory.extrapolate,
    representations=representations,
    num_samples=num_samples,
    random_state=random_state,
)
print(f"PEA-mitigated expectation value: {pea_value:.4f}")
print(f"PEA error:                       {abs(ideal_value - pea_value):.4f}")
```

The error is several times smaller than the unmitigated error computed above. To see where this number comes from, we now run the same workflow step by step.

## The two-stage workflow

The first stage, {func}`mitiq.experimental.pea.pea.construct_circuits`, rescales the representations at each scale factor and Monte-Carlo samples `num_samples` implementable circuits from each rescaled decomposition. It returns the sampled circuits together with their signs and the one-norm of the rescaled representation at each scale factor.

```{code-cell} ipython3
scaled_circuits, scaled_signs, scaled_norms = pea.construct_circuits(
    circuit,
    scale_factors=scale_factors,
    representations=representations,
    num_samples=num_samples,
    random_state=random_state,
)

print("Circuits sampled per scale factor:", [len(circs) for circs in scaled_circuits])
print("One-norms of the rescaled representations:", np.round(scaled_norms, 8))
```

This is the sampling-overhead payoff: for every $s \geq 1$ the rescaled representation has one-norm exactly $1$ — it is a genuine probability distribution with no negative coefficients, so there is no sign problem and no PEC-style variance amplification. The $\gamma^2 \approx 1.86$ overhead computed above applies only at $s = 0$, the PEC limit that PEA never has to sample.

Canonical noise scaling is only defined up to the per-operation limit $s \leq \gamma^+/\gamma^-$, at which the positive volume is exhausted; beyond it, scaling raises a `ValueError`. For our representations that limit is far above the scale factors used here:

```{code-cell} ipython3
canonical_limits = []
for rep in representations:
    gamma_plus = sum(c for c in rep.coeffs if c > 0)
    gamma_minus = -sum(c for c in rep.coeffs if c < 0)
    canonical_limits.append(gamma_plus / gamma_minus)

print(f"Smallest per-operation canonical limit: s <= {min(canonical_limits):.1f}")
```

The second stage is circuit execution, which in a real experiment would be a batched submission to hardware. Many of the sampled circuits are duplicates — at $s = 1$ nearly every sample is the bare circuit — so we wrap the executor in {class}`.Executor` and let it run only the unique circuits, exactly as `execute_with_pea` does internally:

```{code-cell} ipython3
executor = Executor(execute)
scaled_results = [
    executor.evaluate(circs, force_run_all=False) for circs in scaled_circuits
]
print(f"Unique circuits actually executed: {len(executor.executed_circuits)}")
```

Before extrapolating, it is instructive to look at the noise-scaled expectation values themselves. At each scale factor the unbiased estimator averages the executed values weighted by their signs and the one-norm:

```{code-cell} ipython3
expectation_values = [
    np.mean([norm * sign * value for sign, value in zip(signs, values)])
    for values, norm, signs in zip(scaled_results, scaled_norms, scaled_signs)
]

for s, ev in zip(scale_factors, expectation_values):
    print(f"E(s={s}) = {ev:.4f}")
print(f"Unmitigated value:  {noisy_value:.4f}")
```

Note that $E(s=1)$ matches the unmitigated expectation value. Canonical scaling anchors the amplification axis at the device's physical noise level in the sense that $s = 1$ injects no *additional* noise: the rescaled representation is a probability distribution over implementable circuits, each carrying only the noise the device already has. For these near-identity depolarizing representations that distribution happens to put its weight overwhelmingly — though not entirely — on the bare noisy operation: the single-qubit representations collapse to the single bare term, while the two-qubit representation retains its nine tiny double-Pauli terms:

```{code-cell} ipython3
rep_at_1 = next(
    rep
    for rep in scale_representations(representations, 1.0)
    if len(rep.ideal.all_qubits()) == 2
)
coeffs_at_1 = np.array(rep_at_1.coeffs)
bare_weight = sum(
    coeff
    for coeff, op in zip(rep_at_1.coeffs, rep_at_1.noisy_operations)
    if op.circuit == rep_at_1.ideal
)
num_two_qubit_ops = sum(
    1 for op in circuit.all_operations() if len(op.qubits) == 2
)

print(f"Two-qubit operation instances in the circuit: {num_two_qubit_ops}")
print(f"Two-qubit representation at s=1: {np.sum(np.abs(coeffs_at_1) > 0)} nonzero terms")
print(f"Probability of the bare noisy operation: {bare_weight:.10f}")
print(f"Total probability of double-Pauli terms: {np.sum(coeffs_at_1) - bare_weight:.3e}")
print(
    "Expected number of sampled circuits containing one: "
    f"{num_samples * num_two_qubit_ops * (np.sum(coeffs_at_1) - bare_weight):.2f}"
)
print(
    "Probability a sampled circuit is not the bare circuit: "
    f"{1 - bare_weight**num_two_qubit_ops:.3e}"
)
```

The $s = 1$ mixture is therefore a subtly different quantity from the bare-device expectation value: the double-Pauli terms shift the *expected* value of the estimator by a systematic offset that would persist with infinite sampling. Here the effect is tiny — the probability that a sampled circuit is not the bare circuit is about $10^{-4}$ — and with an expected count of only $0.2$ such circuits among the 2,000 samples, none was drawn in this run, which is why the two printed values coincide to all digits. That coincidence is a property of this draw, not an identity. The anchoring point stands in its accurate form: moving from the bare circuit to the $s = 1$ mixture adds no noise beyond what the device already has.

The final stage, {func}`mitiq.experimental.pea.pea.combine_results`, performs this recombination and extrapolates the resulting curve to $s = 0$:

```{code-cell} ipython3
pea_linear = pea.combine_results(
    scale_factors, scaled_results, scaled_norms, scaled_signs, LinearFactory.extrapolate
)
pea_richardson = pea.combine_results(
    scale_factors, scaled_results, scaled_norms, scaled_signs, RichardsonFactory.extrapolate
)

print(f"Ideal value:  {ideal_value:.4f}")
print(f"Unmitigated:  {noisy_value:.4f}   error {abs(ideal_value - noisy_value):.4f}")
print(f"PEA (linear):     {pea_linear:.4f}   error {abs(ideal_value - pea_linear):.4f}")
print(f"PEA (Richardson): {pea_richardson:.4f}   error {abs(ideal_value - pea_richardson):.4f}")
```

## Visualizing the extrapolation

The plot below is the whole method in one picture: the measured expectation values decay as the noise is amplified, and the fitted curves carry that trend back to the zero-noise limit $s = 0$, far closer to the ideal value than the unmitigated result at $s = 1$.

```{code-cell} ipython3
s_grid = np.linspace(0.0, max(scale_factors) + 0.1, 100)
linear_fit = np.poly1d(np.polyfit(scale_factors, expectation_values, deg=1))
richardson_fit = np.poly1d(np.polyfit(scale_factors, expectation_values, deg=2))

plt.figure(figsize=(9, 5))
plt.axhline(ideal_value, color="tab:green", linestyle="--", label="Ideal")
plt.axhline(noisy_value, color="tab:red", linestyle=":", label="Unmitigated")
plt.plot(s_grid, linear_fit(s_grid), color="tab:blue", lw=2, label="Linear fit")
plt.plot(
    s_grid, richardson_fit(s_grid), color="tab:orange", lw=2, label="Richardson fit"
)
plt.plot(
    scale_factors, expectation_values, "o", color="black", ms=8, zorder=5,
    label="Measured $E(s)$",
)
plt.plot(0.0, linear_fit(0.0), "s", color="tab:blue", ms=9, zorder=5)
plt.plot(0.0, richardson_fit(0.0), "s", color="tab:orange", ms=9, zorder=5)
plt.xlabel("Noise scale factor $s$")
plt.ylabel("Expectation value")
plt.title("PEA: extrapolating noise-amplified expectation values")
plt.legend(fontsize=12)
plt.tight_layout()
```

## Choosing the extrapolation

The extrapolation model matters. Under depolarizing noise the fidelity decays approximately — but not exactly — exponentially in the amplification gain, so over a short span of scale factors $E(s)$ is only approximately linear. A linear fit therefore carries a residual bias that grows with the noise strength and with the range of scale factors; in the run above it undershoots the ideal value. The Richardson fit (a degree-2 polynomial through the three points) removes more of that curvature bias, at the cost of amplifying the statistical fluctuations of the measured points — with the density matrix simulator used here those fluctuations come only from the Monte-Carlo sampling of the Pauli insertions. This bias–variance trade-off is the same one faced in [ZNE](../guide/zne.md), and all the extrapolation factories of {mod}`mitiq.zne.inference` can be passed as the `extrapolation_method`.

## Limitations and when to use PEA

- **PEA needs a characterized noise model.** Here the representations were built analytically from the known depolarizing channel; on hardware they must be learned from the device, e.g. via sparse Pauli–Lindblad noise tomography as in *Kim et al.* {cite}`Kim_2023_Nature`. The `representations` argument accepts such learned representations directly.
- **The noise match is close but not exact.** Cirq's `with_noise` transform applies a depolarizing channel to every qubit after every moment — including qubits that are idle in that moment — while the representations model one channel per operation qubit; the two counts are compared below. In addition, the Pauli corrections sampled from the representations are appended as real gates, so the executor gives them depolarizing noise of their own, which the analytic representations do not model. Because of both effects, even the $s = 0$ (PEC) limit of these representations would not recover the ideal value exactly. On a real device, imperfect noise characterization plays the same role, typically much more strongly, and adds a systematic error that mitigation cannot remove.
- **Finite sampling.** The measured $E(s)$ carry statistical error from the finite `num_samples` circuits per scale factor (and, on hardware, from finite shots), which the extrapolation propagates — and can amplify — in the zero-noise estimate.
- **Scale factors are bounded.** Canonical noise scaling is defined only up to the per-operation limit $s \leq \gamma^+/\gamma^-$ computed above; larger scale factors raise a `ValueError`.

The channel-count mismatch of the second point can be made concrete: on this circuit the executor applies one more depolarizing channel than the per-operation representations account for, because the final moment leaves one qubit idle.

```{code-cell} ipython3
channels_applied = sum(
    1
    for op in circuit.with_noise(cirq.depolarize(p=epsilon)).all_operations()
    if isinstance(op.gate, cirq.DepolarizingChannel)
)
channels_in_representations = sum(
    len(op.qubits) for op in circuit.all_operations()
)
print(f"Depolarizing channels applied by the executor: {channels_applied}")
print(f"Channels accounted for by the representations: {channels_in_representations}")
```

For guidance on when PEA is preferable to PEC or ZNE, see [the PEA use-case section of the user guide](../guide/pea-2-use-case.md).
