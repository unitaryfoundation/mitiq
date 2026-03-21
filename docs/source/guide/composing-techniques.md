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

# Composing Techniques

Real quantum hardware suffers from multiple, simultaneous sources of noise — gate errors, readout
errors, time-correlated noise, coherent over-rotations, and more. No single error mitigation
technique addresses all of these at once. Mitiq makes it straightforward to **compose** (stack)
multiple techniques so each one targets the noise it handles best.

This page explains:

- how ZNE acts as the outer framework in most compositions,
- why different techniques have different interfaces and what that means for composing them,
- worked examples for REM + ZNE, DDD + ZNE, PT + ZNE, and a three-technique combination.

---

## ZNE as the outer framework

A useful mental model for composing techniques in Mitiq is that **ZNE acts as a container**.
It defines the noise levels, generates the scaled circuits, and does the final extrapolation.
Every other technique — DDD, PT, REM — operates *inside* that container, acting on each
individual scaled circuit before ZNE collects the results and draws the extrapolation line.

```
ZNE: define noise levels [λ=1, λ=3, λ=5]
  │
  ├── scaled circuit at λ=1 → [DDD / PT / REM act here] → expectation value
  ├── scaled circuit at λ=3 → [DDD / PT / REM act here] → expectation value
  └── scaled circuit at λ=5 → [DDD / PT / REM act here] → expectation value
                                                                    │
                                                          ZNE: extrapolate to λ=0
```

This means the "order" of techniques has two perspectives that can seem contradictory at first:

- **In terms of circuit preparation**, ZNE goes first — it creates the scaled circuits that
  everything else operates on.
- **In terms of noise handling**, PT/DDD/REM go first — they clean up or tailor the noise on
  each scaled circuit before ZNE extrapolates.

Both are true at the same time. When this page says "apply PT before ZNE", it means PT acts on
each ZNE-scaled circuit, not that you call PT before calling ZNE in your code.

---

## Why techniques have different interfaces

Not all techniques plug together the same way. The difference comes down to what each one is
actually doing to the circuit and its results.

**ZNE and DDD** both produce a single mitigated `float` as output. ZNE runs the circuit multiple
times at different noise levels and extrapolates. DDD transforms the circuit once by inserting
pulse sequences, then runs it. Because both return `float`, either can wrap the other's executor
directly via `mitigate_executor`.

**REM is different.** REM specifically targets errors that happen during *measurement* — the
readout stage. To correct those errors, it needs the raw measurement data (bitstrings) before
they are turned into an average. This is why an REM executor must return `MeasurementResult`,
not `float`. The mitigation math (applying the inverse confusion matrix) runs directly on those
raw counts. Only after that correction does REM hand a usable result upstream.

This has a direct consequence for composition: **REM must always be the innermost technique**
when combined with ZNE or DDD, because it is the only one that operates on raw measurement data.
ZNE wraps REM, not the other way around.

| Technique | `mitigate_executor`? | Executor must return | Returns |
|-----------|:--------------------:|----------------------|---------|
| ZNE | yes | `float` or `QuantumResult` + observable | `float` |
| REM | yes | `MeasurementResult` | `MeasurementResult` |
| DDD | yes | `float` or `QuantumResult` + observable | `float` |
| PT | no — manual loop | any | averaged `float` |

**PT has no `mitigate_executor` at all.** This is structural, not an oversight. Most techniques
follow a 1-to-1 or 1-to-N circuit flow that still ends in a single expectation value —
`mitigate_executor` handles that pattern automatically. PT is N-to-1: it requires generating
many random circuit variants, running all of them, and averaging the results to cancel out
coherent noise. That averaging loop cannot be hidden behind a single executor call, so PT always
requires a manual loop using `zne.construct_circuits` and `pt.generate_pauli_twirl_variants`.

---

## Example 1: REM + ZNE

REM corrects bitflip errors at the measurement stage. ZNE extrapolates away gate noise.
They target completely different parts of the problem, which is what makes them a natural pair.

The composition here follows directly from their interfaces. The raw executor returns
`MeasurementResult` — REM needs that to apply the inverse confusion matrix to the raw bitstrings.
After correction, REM hands results upstream. ZNE then wraps the REM executor, calling it
multiple times across noise scale factors and extrapolating to zero noise. In code this is a
single chain: `zne.mitigate_executor(rem_executor, ...)`.

```{code-cell} ipython3
import cirq
import numpy as np
from functools import partial

from mitiq import MeasurementResult, Observable, PauliString, raw
from mitiq.benchmarks import generate_rb_circuits
from mitiq import rem, zne

# --- Circuit and observable --------------------------------------------------
circuit = generate_rb_circuits(2, 10)[0]
obs = Observable(PauliString("ZI"), PauliString("IZ"))  # ideal value = 2

# --- Raw executor: depolarizing gate noise + readout bit-flip ---------------
def execute(
    circuit: cirq.Circuit,
    noise_level: float = 0.002,
    p0: float = 0.05,
) -> MeasurementResult:
    measurements = circuit[-1]
    body = circuit[:-1].with_noise(cirq.depolarize(noise_level))
    body.append(cirq.bit_flip(p0).on_each(body.all_qubits()))
    body.append(measurements)
    result = cirq.DensityMatrixSimulator().run(body, repetitions=8192)
    return MeasurementResult(np.column_stack(list(result.measurements.values())))

# --- Unmitigated baseline ---------------------------------------------------
noisy_value = raw.execute(circuit, execute, obs)

# --- REM alone --------------------------------------------------------------
p_flip = 0.05
icm = rem.generate_inverse_confusion_matrix(2, p_flip, p_flip)
rem_executor = rem.mitigate_executor(execute, inverse_confusion_matrix=icm)
rem_result = obs.expectation(circuit, rem_executor)

# --- ZNE alone --------------------------------------------------------------
zne_executor = zne.mitigate_executor(execute, observable=obs)
zne_result = zne_executor(circuit)

# --- REM + ZNE: wrap the REM executor with ZNE ------------------------------
# rem_executor returns MeasurementResult; ZNE wraps it and handles the
# observable internally, returning float.
combined_executor = zne.mitigate_executor(rem_executor, observable=obs)
combined_result = combined_executor(circuit)

print(f"Ideal value:            2.00000")
print(f"Unmitigated:            {noisy_value.real:.5f}")
print(f"REM only:               {rem_result.real:.5f}")
print(f"ZNE only:               {zne_result.real:.5f}")
print(f"REM + ZNE:              {combined_result.real:.5f}")
```

```{note}
The full tutorial for this combination is available at
{doc}`../examples/combine_rem_zne`.
```

---

## Example 2: DDD + ZNE

DDD targets time-correlated (non-Markovian) noise by inserting decoupling pulse sequences into
idle windows of a circuit. ZNE then extrapolates the remaining gate noise to zero.

The composition works like this: ZNE generates noise-scaled copies of the circuit. For each
scaled copy, DDD inserts its pulse sequences before the circuit is executed. ZNE then collects
the results from all noise levels and extrapolates. In other words, ZNE is the outer container
and DDD acts inside it — on every single scaled circuit independently.

In practice the simplest way to implement this is to insert the DDD sequences into the circuit
first with `ddd.insert_ddd_sequences`, then hand the modified circuit to the ZNE executor.
ZNE will fold the DDD-modified circuit at each noise level, meaning every scaled variant
automatically has DDD sequences in it.

```{code-cell} ipython3
import cirq
from mitiq import MeasurementResult, Observable, PauliString
from mitiq import ddd, zne
from functools import partial

# --- Circuit: 6-qubit GHZ with idle steps to amplify correlated noise -------
def ghz(num_qubits, idle_steps=0):
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    for i in range(num_qubits):
        if i == 0:
            circuit.append(cirq.H(qubits[0]))
        else:
            circuit.append(cirq.CNOT(qubits[0], qubits[i]))
            others = qubits[1:i] + qubits[i + 1:]
            for _ in range(idle_steps):
                circuit.append(cirq.I(q) for q in others)
    return circuit

num_qubits = 6
circuit = ghz(num_qubits, idle_steps=3)

obs = Observable(PauliString("X" * num_qubits))  # ideal value = 1

# --- Executor: systematic Rz dephasing + depolarizing noise -----------------
def execute(
    circuit: cirq.Circuit,
    rz_noise: float = 0.02,
    depolar_noise: float = 0.005,
) -> MeasurementResult:
    circuit = circuit.with_noise(cirq.rz(rz_noise))
    circuit = circuit.with_noise(cirq.bit_flip(depolar_noise))
    circuit += cirq.measure(*sorted(circuit.all_qubits()), key="m")
    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=1000)
    return MeasurementResult(result.measurements["m"])

noisy_exec = partial(execute, rz_noise=0.02, depolar_noise=0.005)
ideal_exec = partial(execute, rz_noise=0.0, depolar_noise=0.0)

ideal_value = obs.expectation(circuit, ideal_exec).real
noisy_value = obs.expectation(circuit, noisy_exec).real

# --- DDD alone --------------------------------------------------------------
ddd_executor = ddd.mitigate_executor(noisy_exec, observable=obs, rule=ddd.rules.yy)
ddd_result = ddd_executor(circuit)

# --- ZNE alone --------------------------------------------------------------
zne_executor = zne.mitigate_executor(noisy_exec, observable=obs)
zne_result = zne_executor(circuit)

# --- DDD + ZNE: pass the DDD circuit (already modified) to the ZNE executor.
# ZNE scales the DDD-modified circuit; each ZNE variant is then evaluated via
# the noisy executor.  The simplest approach is to build the DDD circuit first
# and then apply ZNE on top.
ddd_circuit = ddd.insert_ddd_sequences(circuit, ddd.rules.yy)
combined_result = zne_executor(ddd_circuit)

print(f"Ideal value:            {ideal_value:.5f}")
print(f"Unmitigated:            {noisy_value:.5f}")
print(f"DDD only:               {ddd_result:.5f}")
print(f"ZNE only:               {float(np.real(zne_result)):.5f}")
print(f"DDD + ZNE:              {float(np.real(combined_result)):.5f}")
```

```{note}
The full tutorial for this combination is available at
{doc}`../examples/combine_ddd_zne`.

Because the noise model here is time-correlated, ZNE's gate folding can interact with it
unpredictably — meaning ZNE alone may occasionally appear to outperform DDD + ZNE on a single
run. This is expected. The combination is more consistent across runs because DDD first converts
the time-correlated noise into something ZNE can extrapolate over reliably. The existing
tutorial notes this explicitly: "ZNE by itself actually makes things worse than the unmitigated
expectation value" for this noise model in many cases.
```

---

## Example 3: PT + ZNE

PT is a noise *tailoring* technique — it does not reduce noise on its own. What it does is
convert coherent noise (structured over-rotations) into stochastic Pauli noise, which ZNE can
then extrapolate over reliably. Without PT, ZNE applied to coherent noise can make things worse
because gate folding amplifies coherent errors unfavourably.

PT has no `mitigate_executor` because of how it works. Most techniques follow a pattern that
`mitigate_executor` can automate: take one circuit, return one result. PT is different — it is
N-to-1. To cancel out coherent noise through averaging, you need to generate many random
circuit variants (the twirled circuits), run all of them, and average the results. That loop
cannot be hidden behind a single executor call, so PT always requires manual handling.

The composition is done in steps:

1. **ZNE** generates noise-scaled circuit variants at each scale factor.
2. For each scaled circuit, **PT** generates many twirled copies and averages the results —
   this is the inner loop.
3. **ZNE** takes the averaged expectation values across noise levels and extrapolates to zero.

Note that PT twirls the already-scaled circuits, not the original. If you twirled first and
then let ZNE fold the twirled circuit, the folding would corrupt the twirling gates.

```{code-cell} ipython3
import cirq
import numpy as np
from functools import partial

from mitiq import Executor
from mitiq.benchmarks import generate_ghz_circuit
from mitiq import zne, pt
from cirq import DensityMatrixSimulator, CircuitOperation, CXPowGate, Circuit, Ry

# --- Circuit -----------------------------------------------------------------
circuit = generate_ghz_circuit(n_qubits=7)

# --- Coherent noise model: Ry over-rotation on each CNOT output -------------
from cirq.devices.noise_model import GateSubstitutionNoiseModel

def coherent_cnot(over_rotation: float) -> GateSubstitutionNoiseModel:
    rads = over_rotation * np.pi / 2
    def cnot_ry(op):
        if isinstance(op.gate, CXPowGate):
            return CircuitOperation(
                Circuit(op.gate.on(*op.qubits), Ry(rads=rads).on_each(op.qubits)).freeze()
            )
        return op
    return GateSubstitutionNoiseModel(cnot_ry)

def execute(circuit: Circuit, noise_level: float) -> float:
    return (
        DensityMatrixSimulator(noise=coherent_cnot(noise_level))
        .simulate(circuit)
        .final_density_matrix[0, 0]
        .real
    )

NOISE_LEVEL = 0.2
ideal_value = execute(circuit, noise_level=0.0)
noisy_value = execute(circuit, noise_level=NOISE_LEVEL)
noisy_exec  = partial(execute, noise_level=NOISE_LEVEL)

# --- ZNE alone --------------------------------------------------------------
zne_result = zne.execute_with_zne(circuit, noisy_exec)

# --- PT + ZNE ---------------------------------------------------------------
scale_factors      = [1, 3, 5]
NUM_TWIRLED        = 100

noise_scaled_circuits  = zne.construct_circuits(circuit, scale_factors)
executor_obj           = Executor(noisy_exec)
noise_scaled_expvals   = []

for scaled_circuit in noise_scaled_circuits:
    twirled = pt.generate_pauli_twirl_variants(scaled_circuit, num_circuits=NUM_TWIRLED)
    avg     = np.average(executor_obj.evaluate(twirled))
    noise_scaled_expvals.append(avg)

extrapolate = zne.inference.RichardsonFactory(scale_factors=scale_factors).extrapolate
pt_zne_result = zne.combine_results(scale_factors, noise_scaled_expvals, extrapolate)

print(f"Ideal value:            {ideal_value:.5f}")
print(f"Unmitigated:            {noisy_value:.5f}")
print(f"ZNE only:               {zne_result:.5f}")
print(f"PT + ZNE:               {pt_zne_result:.5f}")
```

```{note}
The full tutorial for this combination, including a comparison of errors over a range of
noise strengths, is available at {doc}`../examples/pt_zne`.
```

---

## Example 4: PT + DDD + ZNE (three-technique combination)

When a circuit has both coherent noise and time-correlated noise on top of generic gate noise,
stacking all three techniques can help — each one addressing a different layer of the problem.

ZNE is still the outer container. Inside it, for each scaled circuit: DDD inserts pulse sequences
to suppress time-correlated noise, and PT twirls the result to convert any remaining coherent
noise into stochastic noise. ZNE then extrapolates across the noise levels.

The step-by-step order in code:

1. **ZNE** generates noise-scaled circuits.
2. **DDD** inserts decoupling sequences into each scaled circuit.
3. **PT** generates twirled variants of each DDD-modified, scaled circuit and averages them.
4. **ZNE** extrapolates across the averaged expectation values at each noise level.

```{code-cell} ipython3
import cirq
import numpy as np
from functools import partial

from mitiq import Executor, MeasurementResult, Observable, PauliString
from mitiq import pt, ddd, zne
from mitiq.benchmarks import generate_ghz_circuit
from cirq import DensityMatrixSimulator, CircuitOperation, CXPowGate, Circuit, Ry
from cirq.devices.noise_model import GateSubstitutionNoiseModel

# --- Circuit and observable --------------------------------------------------
circuit = generate_ghz_circuit(n_qubits=5)
obs = Observable(*[PauliString("X" * 5)])  # ideal value = 1

# --- Combined noise: coherent CNOT over-rotation + time-correlated dephasing
def coherent_cnot(over_rotation: float) -> GateSubstitutionNoiseModel:
    rads = over_rotation * np.pi / 2
    def cnot_ry(op):
        if isinstance(op.gate, CXPowGate):
            return CircuitOperation(
                Circuit(op.gate.on(*op.qubits), Ry(rads=rads).on_each(op.qubits)).freeze()
            )
        return op
    return GateSubstitutionNoiseModel(cnot_ry)

def execute(circuit: Circuit, noise_level: float = 0.15, rz_noise: float = 0.01) -> float:
    """Simulate with coherent CNOT over-rotation and Rz dephasing."""
    circuit = circuit.with_noise(cirq.rz(rz_noise))
    return (
        DensityMatrixSimulator(noise=coherent_cnot(noise_level))
        .simulate(circuit)
        .final_density_matrix[0, 0]
        .real
    )

NOISE_LEVEL = 0.15
ideal_value = execute(circuit, noise_level=0.0, rz_noise=0.0)
noisy_value = execute(circuit, noise_level=NOISE_LEVEL)
noisy_exec  = partial(execute, noise_level=NOISE_LEVEL)

# --- PT + DDD + ZNE ---------------------------------------------------------
# Step 1: ZNE — generate noise-scaled circuit variants
scale_factors = [1, 3, 5]
noise_scaled_circuits = zne.construct_circuits(circuit, scale_factors)

executor_obj       = Executor(noisy_exec)
NUM_TWIRLED        = 50
noise_scaled_expvals = []

for scaled_circuit in noise_scaled_circuits:
    # Step 2: DDD — insert decoupling sequences into this noise-scaled circuit
    ddd_circuit = ddd.insert_ddd_sequences(scaled_circuit, ddd.rules.yy)

    # Step 3: PT — twirl the DDD-modified, noise-scaled circuit
    twirled = pt.generate_pauli_twirl_variants(ddd_circuit, num_circuits=NUM_TWIRLED)
    avg     = np.average(executor_obj.evaluate(twirled))
    noise_scaled_expvals.append(avg)

# Step 4: ZNE — extrapolate to zero noise
extrapolate = zne.inference.RichardsonFactory(scale_factors=scale_factors).extrapolate
combined_result = zne.combine_results(scale_factors, noise_scaled_expvals, extrapolate)

# Individual technique baselines for comparison
zne_result = zne.execute_with_zne(circuit, noisy_exec)
# noisy_exec already returns float, so no observable argument needed for DDD
ddd_executor = ddd.mitigate_executor(noisy_exec, rule=ddd.rules.yy)
ddd_result = ddd_executor(circuit)

print(f"Ideal value:            {ideal_value:.5f}")
print(f"Unmitigated:            {noisy_value:.5f}")
print(f"ZNE only:               {zne_result:.5f}")
print(f"DDD only:               {ddd_result:.5f}")
print(f"PT + DDD + ZNE:         {combined_result:.5f}")
```

```{note}
This example uses three techniques. If your device also has significant readout errors, the
full four-technique combination (PT + DDD + REM + ZNE) is covered in
{doc}`../examples/advanced_error_mitigation_pipeline`.
```

---

## When composing can backfire

Stacking techniques is not always better. More techniques means more circuit executions, deeper
circuits, and more opportunities for things to interact badly. Before composing, it is worth
knowing the failure modes — all of which are documented in the individual technique guides.

**Applying ZNE to the wrong noise type.** ZNE works by intentionally amplifying noise and
extrapolating back to zero. That assumption only holds cleanly for incoherent, Markovian noise
that scales predictably with circuit depth. For coherent noise (systematic over-rotations), gate
folding amplifies the coherent errors unfavourably, and ZNE can make results worse than doing
nothing at all. The `pt_zne` tutorial documents this directly. The fix is to apply PT first to
convert the coherent noise into stochastic noise before ZNE runs — which is exactly why that
combination exists.

**DDD applied to the wrong noise type.** DDD is specifically designed for time-correlated,
non-Markovian noise. For purely Markovian noise, the DDD theory page notes it can make the
channel more symmetric but cannot actually decouple the system from the environment — meaning
the benefit is limited at best. On top of that, gate-level DDD is an approximation of the ideal
pulse-level technique, and some backends internally reschedule gates in ways that can undermine
the sequences. The theory page is explicit: "it may happen that, for some sequences, the final
error of the quantum computation is actually increased."

**PT transforming noise into something worse.** PT generally simplifies a noise channel, but
the PT theory page flags a specific failure mode: in some circumstances it can transform the
noise into a completely depolarizing channel, which means a total loss of quantum information.
This is more likely when the noise is already severe or the number of twirled circuits used for
averaging is too small.

**Each technique adds execution overhead.** Every technique you add requires more circuit
executions. ZNE needs circuits at multiple noise scale factors. PT needs many twirled variants
to average over. DDD runs multiple trials. The `resource-requirements` guide covers how to
measure this overhead — but the practical implication is that on a real device with a limited
shot budget, stacking too many techniques can hurt your statistics more than the mitigation
helps. The advanced pipeline tutorial makes this point directly: "the full pipeline does not
always perform best."

**ZNE error amplification compounds across techniques.** When you compose techniques, any
statistical uncertainty in the intermediate results gets passed downstream. The
`error-mitigation` guide notes that initial errors in measured expectation values propagate to
the extrapolated ZNE value and can significantly amplify statistical uncertainty. If earlier
techniques in the pipeline (PT averaging, REM correction) introduce any bias or variance, ZNE
will extrapolate over that too.

The short version: composing is most valuable when your device has clearly distinct, identifiable
noise sources that map to different techniques. If you are not sure what noise dominates, start
with a single technique, characterize the improvement, and add a second only if there is a clear
remaining gap to address.

---

## Technique × noise model compatibility

Not all techniques are created equal — and not all noise is the same. Before stacking techniques,
it helps to know what each one actually does and what kind of noise it is going after. Combining
two techniques that both target the same noise source adds overhead without adding much benefit.
The goal is to pick techniques that each tackle a *different* piece of the problem.

The table below shows how each technique maps to common noise types. **Y** means the technique
directly addresses that noise type, **partial** means it has an indirect or limited benefit, and
**N** means it simply is not designed for it.

| Noise type | ZNE | LRE | REM | DDD | PT | PEC | CDR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Incoherent gate noise (depolarizing, amplitude damping) | Y | Y | N | partial | partial | Y | Y |
| Coherent gate noise (over-rotations) | partial | partial | N | N | Y | Y | Y |
| Time-correlated (non-Markovian) noise | partial | partial | N | Y | partial | N | N |
| Readout / measurement errors | N | N | Y | N | N | N | N |
| Crosstalk (idle-qubit errors) | N | N | N | Y | partial | N | N |

To understand the table, it helps to know a bit about each technique's properties:

**ZNE and LRE** are extrapolation-based techniques. They intentionally run circuits at amplified
noise levels and infer what the result would be at zero noise. They work well on incoherent,
Markovian gate noise — the kind that scales predictably. The catch is that coherent or
time-correlated noise does not scale cleanly, so applying ZNE or LRE to those noise types without
first tailoring the noise can actually make things worse. That is where PT and DDD come in.

**REM** is the only technique in this list that handles readout errors — the mistakes that happen
at the measurement stage, not during the computation itself. If your device has significant
measurement error, REM is not optional; it is the only tool for the job. It wraps the raw
executor and corrects the bitstring distributions before anything else processes them.

**DDD** works by inserting carefully chosen pulse sequences into the idle windows of a circuit.
It is specifically designed for time-correlated (non-Markovian) noise — the kind where errors
build up over time rather than occurring independently at each gate. It also helps with
crosstalk between qubits that are sitting idle. On purely random, Markovian noise, DDD has
limited impact.

**PT (Pauli Twirling)** is a noise *tailoring* technique, not a noise *reduction* technique. On
its own, it does not lower your error rate. What it does is convert coherent noise — structured,
systematic over-rotations — into stochastic Pauli noise. That conversion matters because ZNE
and LRE assume stochastic noise. Applying PT first makes those extrapolation techniques far
more reliable in the presence of coherent errors.

**PEC and CDR** are noise-aware techniques that can handle both incoherent and coherent noise,
but they come with a higher sampling cost. PEC requires a characterised noise model; CDR
requires a set of training circuits with known ideal values. They are powerful, but the overhead
means they are best used when simpler combinations are not sufficient.

The guiding principle is straightforward: look at the noise sources in your device, identify
which techniques address different ones, and compose those. Stacking ZNE on top of more ZNE
does not help. Stacking REM (readout) with DDD (time-correlated) with ZNE (gate noise) does —
because each one is solving a different part of the problem.

For additional worked examples covering more technique combinations, see the
{doc}`../examples/examples` page.