# Composing Techniques

Real quantum hardware suffers from multiple, simultaneous sources of noise — gate errors, readout
errors, time-correlated noise, coherent over-rotations, and more. No single error mitigation
technique addresses all of these at once. Mitiq makes it straightforward to **compose** (stack)
multiple techniques so each one targets the noise it handles best.

This page explains:

- which techniques can be composed and how their interfaces interact,
- when composing can backfire,
- a noise × technique compatibility reference.

For worked examples of specific combinations, see the tutorials linked throughout this page and
in the {doc}`../examples/examples` page.

---

## Technique interfaces

Most Mitiq techniques expose a `mitigate_executor` helper. It accepts a raw executor (a callable
`circuit → QuantumResult`) and returns a new executor that applies that technique automatically.
Because the output is itself an executor, it can be passed into another technique's
`mitigate_executor`, creating a chain.

The key question when composing is: **what does each technique's executor need as input, and
what does it return?**

| Technique | `mitigate_executor`? | Executor input | Returns |
|-----------|:--------------------:|----------------|---------|
| ZNE, LRE, DDD, PEC, CDR, QSE | yes | `Circuit` | `float` |
| REM | yes | `Circuit` | `MeasurementResult` (bitstrings) |
| PT | no | `Circuit` | `float` (after manual `np.mean`) |
| VD | no | Multi-copy `Circuit` | `list[float]` |
| PEA | no | `Circuit` + noise model | `float` |
| Shadows | no | `Circuit` | `float` (via classical post-processing) |

Most techniques treat the quantum computer as a black box that returns a single expectation
value — a `float`. They operate at that level, running the circuit one or more times and
producing a mitigated number.

**REM** is the exception. Rather than working on expectation values, REM operates on the full
probability distribution. The data flow when REM is composed with another technique looks like
this:

1. **Backend** executes the circuit and returns a `MeasurementResult` (raw bitstrings / counts)
2. **REM** (innermost) converts counts → probability vector → applies inverse confusion matrix → returns corrected `MeasurementResult`
3. **Observable** converts corrected counts → `float` expectation value
4. **Outer technique** (e.g. any `mitigate_executor`-based technique) takes that `float` → returns mitigated `float`

This is why REM must always be the innermost technique when combined with others — it needs
to run before any expectation value is computed. See {doc}`rem-5-theory` for the full details.

**PT** has no `mitigate_executor` for composability reasons. PT is designed to be used as a
circuit modifier inside another technique's loop — twirling each noise-scaled circuit
individually before execution. Wrapping it in its own executor would make that nesting
impossible. See {doc}`../examples/pt_zne` for how the manual loop works in practice.

**VD, PEA, and Shadows** have no `mitigate_executor` because their execution patterns don't
fit the simple `executor → executor` wrapper model. VD requires a specially constructed
multi-copy circuit; PEA requires an explicit noise model and probabilistic sampling across
scale factors; Shadows uses a two-step classical post-processing API. Each is accessed via
its own `execute_with_*` or dedicated functions.

---

## Composing two techniques

When chaining two techniques via `mitigate_executor`, the inner technique wraps the raw
executor and the outer technique wraps the result:

```python
inner_executor = technique_A.mitigate_executor(raw_executor, ...)
outer_executor = technique_B.mitigate_executor(inner_executor, ...)
result = outer_executor(circuit)
```

The output type of the inner technique must be compatible with what the outer technique expects.
Since REM returns `MeasurementResult` and all other techniques return `float`, REM must always
be inner. Every other combination of techniques that both use `mitigate_executor` is
straightforward to chain.

---

## When composing can backfire

Stacking techniques is not always better. More techniques means more circuit executions, deeper
circuits, and more opportunities for things to interact badly.

**Applying ZNE or LRE to coherent noise without PT first.** ZNE and LRE work by amplifying
noise and extrapolating back to zero — an assumption that only holds for incoherent, Markovian
noise that scales predictably. For coherent noise, gate folding amplifies errors unfavourably
and results can be worse than doing nothing. The {doc}`../examples/pt_zne` tutorial documents
this directly. Applying PT first converts coherent noise to stochastic noise, making
extrapolation reliable.

**DDD applied to Markovian noise.** DDD is designed for time-correlated, non-Markovian noise.
For purely Markovian noise, the {doc}`ddd-5-theory` page notes it can make the channel more
symmetric but cannot decouple the system from the environment. Additionally, gate-level DDD is
an approximation of the ideal pulse-level technique — some backends reschedule gates internally
in ways that undermine the sequences. As the theory page states, for some sequences the final
error can actually increase.

**PT with too few twirled variants.** PT works by averaging over many random circuit variants
to cancel coherent noise. With too few variants, the averaging is incomplete. In edge cases,
PT can transform the noise into a fully depolarizing channel, which means a total loss of
quantum information.

**Execution overhead.** Every technique adds circuit executions. ZNE needs circuits at multiple
noise scale factors. PT needs many twirled variants. DDD runs multiple trials. On a real device
with a limited shot budget, stacking too many techniques can hurt statistics more than the
mitigation helps. See {doc}`resource-requirements` for how to measure overhead. The advanced
pipeline tutorial notes this directly: the full pipeline does not always perform best.

**ZNE error amplification.** Any statistical uncertainty in intermediate results propagates to
the ZNE extrapolation step and can be amplified. If earlier techniques introduce bias or
variance, ZNE extrapolates over that too.

The short version: composing works best when your device has clearly distinct noise sources
that map to different techniques. If you are not sure what noise dominates, start with one
technique, characterize the improvement, and add a second only if there is a clear remaining
gap.

---

## Technique × noise model compatibility

The table below shows which techniques address which noise types. **✅** means the technique
directly addresses that noise type, **partial** means it has an indirect or limited benefit,
and **❌** means it is not designed for it.

| Noise type | ZNE | LRE | REM | DDD | PT | PEC | CDR | QSE | VD | PEA | Shadows |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Incoherent gate noise (depolarizing, amplitude damping) | ✅ | ✅ | ❌ | partial | partial | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Coherent gate noise (over-rotations) | partial | partial | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | partial | ✅ | partial |
| Time-correlated (non-Markovian) noise | partial | partial | ❌ | ✅ | partial | ❌ | ❌ | ❌ | ❌ | partial | ❌ |
| Readout / measurement errors | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Crosstalk (idle-qubit errors) | ❌ | ❌ | ❌ | ✅ | partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Use this table to identify complementary techniques — ones that each address a different column.
Stacking techniques that target the same noise type adds overhead without meaningful benefit.

For worked examples of specific combinations, see:
- {doc}`../examples/combine_rem_zne` — REM + ZNE
- {doc}`../examples/combine_ddd_zne` — DDD + ZNE
- {doc}`../examples/pt_zne` — PT + ZNE
- {doc}`../examples/advanced_error_mitigation_pipeline` — PT + DDD + REM + ZNE