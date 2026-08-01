---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# What additional options are available in QSE?
In addition to the necessary ingredients already discussed in [How do I use QSE?](qse-1-intro.md), there are a few additional options included in the implementation.

## Caching Pauli Strings to Expectation Values

Specifically, in order to save runtime, the QSE implementation supports the use of a cache that maps pauli strings to their expectation values. This is taken as an additional parameter in the [`execute_with_qse`](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.qse.qse.execute_with_qse) function.

```{warning}
The cache object is modified in place when passing it to `execute_with_qse`.
```

The inclusion of the cache significantly speeds up the runtime and avoids the need for re-computation of already computed values.
Furthermore, since the cache is modified in place, it can be reused as long as the noise model remains the same.

## Finding and creating check operators

Check operators are the main way QSE encodes prior knowledge about the *ideal* output state.
Formally, if $| \Psi \rangle$ is the state prepared by the noiseless circuit, every check operator $M_i$ should leave that state invariant:

$$
M_i | \Psi \rangle = | \Psi \rangle.
$$

While many check operators naturally arise from applying a quantum error-correcting (QEC)
code, {func}`.execute_with_qse` works with any set of operators that satisfy the invariance
condition above — including stabilizers of a code, physical symmetries of a problem,
and other known invariants of the prepared state {cite}`McClean_2020_NatComm`.
In the API they are passed as a sequence of {class}`.PauliString` objects. Prefer
operators for which the ideal state is a $+1$ eigenstate. If it is a $-1$ eigenstate of a
Pauli $P$, use $-P$ via the {class}`.PauliString` coefficient. Multi-term symmetries should
be split into Pauli terms that individually stabilize the state.

```{note}
If you do not know any operators that leave the ideal state invariant, QSE is not a good fit
for that problem. The method needs this a priori structure; it cannot discover check operators
from noisy samples alone.
```

### Sources of check operators

**1. Stabilizer (QEC) codes.**
When the circuit prepares a logical state of a stabilizer code, take generators of the stabilizer
group (or a subset of group elements) as check operators. Expanding the product of
$(I + G_k)$ over generators $G_k$ yields the full set of group elements, which is what the
[[5,1,3]] example in [How do I use QSE?](qse-1-intro.md) does. The identity is optional,
although it is included when the full stabilizer group is used. Any nonempty subset is valid:
more operators can improve mitigation, while fewer operators reduce circuit executions and the
$O(n^2)$ overlap-matrix measurement cost (before caching and Pauli aggregation).

**2. Problem symmetries and conserved quantities.**
Many algorithms prepare states with known symmetries of the underlying Hamiltonian or
observable — for example:

- *Parity* — an even-parity computational-basis state is a $+1$ eigenstate of $Z^{\otimes n}$
  (or of a chosen subset of $Z$ operators).
- *Particle-number / excitation parity* — when the ideal state has definite fermion parity
  (or a related $\mathbb{Z}_2$ symmetry), the corresponding Pauli representation of that
  parity can be used as a check operator.
- *Spin or other discrete symmetries* — if the prepared state is an eigenstate of a Pauli
  (or a small set of commuting Paulis) with known eigenvalue $+1$, those Paulis are valid checks.

Even a single nontrivial symmetry can define a useful one-dimensional expansion beyond
the identity.

**3. Mixed sets.**
Stabilizers and symmetries can be combined. Including a symmetry that is *not* already in the
stabilizer group helps suppress error components that violate that symmetry
(see [What is the theory behind QSE?](qse-5-theory.md)). For example, when running a circuit
that prepares a logical codeword of the [[5,1,3]] code, the four stabilizer generators are one
natural check-operator set. If the logical state is additionally known to be an eigenstate of
the logical $\bar{Z}$ operator — as it is for a computational-basis logical state — $\bar{Z}$
commutes with every stabilizer but is not itself a product of them, so it can be added to the
check list as an extra, non-stabilizer check operator.

### Constructing the code Hamiltonian

The code Hamiltonian $H_c$ defines which state in the expanded subspace is treated as
“least erroneous.” The usual choice, matching the theory page, is to penalize violations of
the checks:

$$
H_c = -\sum_i M_i,
$$

so that the simultaneous $+1$ eigenspace of the $M_i$ is the ground space of $H_c$.
In Mitiq this is an {class}`.Observable` built from the same Pauli terms (with negative
coefficients) that appear in the check list — as in the [[5,1,3]] helper in the intro guide.

You may use a different $H_c$ if you have a better energy-like proxy for the ideal state, but
it should still rank the codespace / symmetry sector you care about below states that break
the checks.

### Example: check operators from a parity symmetry (no QEC code)

The following two-qubit circuit prepares the even-parity Bell state
$| \Phi^+ \rangle = (|00\rangle + |11\rangle)/\sqrt{2}$. That state is a $+1$ eigenstate of
both $XX$ and $ZZ$, which we use as check operators.

```{code-cell} ipython3
import cirq
from mitiq import Observable, PauliString

qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
)

# Symmetries of |Φ+⟩: XX |Φ+⟩ = |Φ+⟩ and ZZ |Φ+⟩ = |Φ+⟩
check_operators = [
    PauliString("II"),
    PauliString("XX"),
    PauliString("ZZ"),
]
code_hamiltonian = Observable(
    PauliString("II", coeff=-1),
    PauliString("XX", coeff=-1),
    PauliString("ZZ", coeff=-1),
)
```

These objects are then passed to {func}`.execute_with_qse` together with an executor and the
observable of interest, exactly as in [How do I use QSE?](qse-1-intro.md).

## Requirements for Check Operators

When specifying the check operators, it is **not** necessary to specify the full exponential number of operators.
As many or as few operators can be specified.
The tradeoff is the fidelity of the projected state.
See [Finding and creating check operators](#finding-and-creating-check-operators) above for how to choose them.
