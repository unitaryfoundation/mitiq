---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{tags} braket, zne, intermediate
```

(label-zne-braket-program-sets)=
# Zero-noise extrapolation with Braket program sets

This tutorial shows how to combine zero-noise extrapolation ([ZNE](../guide/zne.md)) with the
[program sets](https://github.com/amazon-braket/amazon-braket-sdk-python) feature of the
[Braket](https://github.com/amazon-braket/amazon-braket-sdk-python) SDK.
A `ProgramSet` bundles many (circuit, observable) pairs into a *single* task, which is a natural
fit for ZNE: the technique requires executing several noise-scaled versions of the same circuit,
and a program set submits all of them at once instead of one task per circuit.

We use the two-step ZNE workflow of {func}`.zne.construct_circuits` and
{func}`.zne.combine_results`, which keeps Mitiq's noise scaling and extrapolation steps separate
from Braket's batched execution.
Everything below runs on a local noisy simulator, so no AWS account or credentials are needed.

## Setup

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from braket.circuits import Circuit, Noise, Observable
from braket.devices import LocalSimulator
from braket.program_sets import ProgramSet

from mitiq import zne
```

## Defining the circuit and observables

We use a small two-qubit circuit made of `rx`, `ry`, and `cnot` gates.
It prepares an entangled state whose expectation values are neither $0$ nor $1$, so there is a
nontrivial signal for noise to degrade and for ZNE to recover.

```{code-cell} ipython3
circuit = Circuit().rx(0, np.pi / 4).ry(1, np.pi / 4).cnot(0, 1)
print(circuit)
```

We estimate two observables, $Z \otimes Z$ and $I \otimes Z$, defined with **explicit qubit
targets**.

```{code-cell} ipython3
observables = [
    Observable.Z(0) @ Observable.Z(1),
    Observable.I(0) @ Observable.Z(1),
]
```

```{warning}
The qubit targets are essential.
If untargeted observables such as `Observable.Z() @ Observable.Z()` are passed to
`ProgramSet.product`, the observables act on an *empty* set of qubits and every measured
expectation value silently evaluates to exactly $1.0$ --- no error is raised, and any
extrapolation fitted to those values is meaningless.
We demonstrate this pitfall explicitly in the section
{ref}`pitfall-untargeted-observables`.
```

Since the circuit is small, its exact expectation values can be computed classically.
Running with `shots=0` on the local state-vector simulator returns exact (infinite-shot)
expectation values, which we use later to quantify the error of each estimate.

```{code-cell} ipython3
exact_circuit = circuit.copy()
for observable in observables:
    exact_circuit.expectation(observable=observable)

exact_zz, exact_iz = LocalSimulator("braket_sv").run(exact_circuit, shots=0).result().values
print(f"Exact <ZZ>: {exact_zz:.4f}")
print(f"Exact <IZ>: {exact_iz:.4f}")
```

## What is a program set?

A Braket
[`ProgramSet`](https://docs.aws.amazon.com/braket/latest/developerguide/braket-program-sets.html)
is a collection of *executables* --- (circuit, observable) pairs --- submitted to a device as one
task.
Two constructors cover the common cases:

- `ProgramSet.product(circuits, observables=...)` forms the Cartesian product: every circuit is
  paired with every observable, giving `len(circuits) * len(observables)` executables.
- `ProgramSet.zip(circuits, observables=...)` pairs the $i$-th circuit with the $i$-th
  observable, giving `len(circuits)` executables.

The `shots` argument passed to `run` is the *total* over the whole program set, divided evenly
among the executables.
It must therefore be divisible by `total_executables`, otherwise the SDK raises
`ValueError: Total shots must be divisible by number of executables.`

For ZNE this batching is convenient: with $k$ noise-scale factors and $m$ observables, a single
`product` program set replaces $k \times m$ separate submissions.
On managed hardware this means one queued task instead of many round trips, and all noise-scaled
circuits run close together in time, under more similar device conditions --- a practical
motivation, though in this tutorial we only run locally and make no measured claims about
hardware behavior.

## Constructing the noise-scaled circuits

ZNE estimates the zero-noise limit of an expectation value by executing the circuit at several
*noise scale factors* and extrapolating the results back to zero noise.
Here noise is scaled digitally by *global unitary folding*: {func}`.fold_global` maps
a circuit $U$ to $U (U^\dagger U)^n$, which is logically equivalent but has $(2n+1)$ times as
many gates, so it accumulates correspondingly more noise when executed on a noisy device.

{func}`.zne.construct_circuits` applies the scaling method at each scale factor.
It accepts a native Braket `Circuit` and returns native Braket circuits.

```{code-cell} ipython3
scale_factors = [1, 3, 5]

folded_circuits = zne.construct_circuits(
    circuit,
    scale_factors=scale_factors,
    scale_method=zne.scaling.fold_global,
)

for scale_factor, folded in zip(scale_factors, folded_circuits):
    print(f"Scale factor {scale_factor}: {len(folded.instructions)} instructions")
```

The instruction counts grow in proportion to the scale factors, as expected from folding.

## Adding noise after folding

The default `LocalSimulator` is noiseless, so ZNE would have nothing to mitigate: every scaled
circuit would return the same expectation value.
To make the demonstration meaningful we simulate depolarizing noise after every gate, using the
density-matrix backend `LocalSimulator("braket_dm")`.

The order of operations matters: noise channels must be applied to the **already-folded**
circuits.
Unitary folding is defined only for unitary gates --- it inserts $G^\dagger G$ pairs, and a noise
channel has no inverse to insert.
Accordingly, Mitiq's folding functions reject circuits that contain noise instructions:

```{code-cell} ipython3
:tags: [raises-exception]

noisy_before_folding = circuit.copy().apply_gate_noise(Noise.Depolarizing(0.02))
zne.scaling.fold_global(noisy_before_folding, 3)
```

So we fold first (as done above) and then attach the noise model to each scaled circuit.

```{code-cell} ipython3
NOISE_LEVEL = 0.02

noisy_circuits = [
    folded.copy().apply_gate_noise(Noise.Depolarizing(NOISE_LEVEL))
    for folded in folded_circuits
]
```

## Executing the program set

We now build the Cartesian product of the three noise-scaled circuits and the two observables,
giving six executables, and run them all as one task.

```{code-cell} ipython3
program_set = ProgramSet.product(noisy_circuits, observables=observables)
print(f"Circuits in program set:  {len(program_set)}")
print(f"Total executables:        {program_set.total_executables}")
```

```{code-cell} ipython3
TOTAL_SHOTS = 300_000  # divisible by the number of executables

device = LocalSimulator("braket_dm")
result = device.run(program_set, shots=TOTAL_SHOTS).result()

shots_used = sum(result.entries[0][0].counts.values())
print(f"Shots per executable: {shots_used}")
```

The result is indexed like the product that produced it: `result.entries[i][j]` holds the
measured data of circuit `i` under observable `j`, and its `expectation` property gives the
estimated expectation value.

```{code-cell} ipython3
zz_values = [float(result.entries[i][0].expectation) for i in range(len(scale_factors))]
iz_values = [float(result.entries[i][1].expectation) for i in range(len(scale_factors))]

for scale_factor, zz, iz in zip(scale_factors, zz_values, iz_values):
    print(f"Scale factor {scale_factor}: <ZZ> = {zz:.4f}, <IZ> = {iz:.4f}")
```

Both expectation values decay toward zero as the scale factor grows --- exactly the trend ZNE
exploits.

```{note}
When the observables are passed as a list, per-observable values are only available through
`result.entries[i][j].expectation`.
The composite `result.entries[i].expectation(...)` accessor applies just to observables
submitted as a single Braket `Sum` Hamiltonian; with a plain list it raises
`ValueError: No Sum Hamiltonian was measured`.
```

(pitfall-untargeted-observables)=
### Pitfall: untargeted observables

Here is the failure mode from the warning above: with untargeted observables the program set
runs without complaint, but every expectation value is exactly $1.0$.

```{code-cell} ipython3
untargeted = [Observable.Z() @ Observable.Z(), Observable.I() @ Observable.Z()]
bad_program_set = ProgramSet.product(noisy_circuits, observables=untargeted)
bad_result = device.run(bad_program_set, shots=600).result()

print([
    [float(bad_result.entries[i][j].expectation) for j in range(len(untargeted))]
    for i in range(len(noisy_circuits))
])
```

Noisy circuits at three different noise levels all reporting a value of exactly $1.0$ is not
physics; it is the observable acting on no qubits at all.
Always give observables explicit qubit targets when building program sets.

## Extrapolating to the zero-noise limit

{func}`.zne.combine_results` fits the measured expectation values against the scale factors and
returns the extrapolated zero-noise estimate.
Here we use a linear fit, i.e. the extrapolation of {class}`.LinearFactory`, applied to each
observable separately.

```{code-cell} ipython3
zne_zz = zne.combine_results(scale_factors, zz_values, zne.inference.LinearFactory.extrapolate)
zne_iz = zne.combine_results(scale_factors, iz_values, zne.inference.LinearFactory.extrapolate)

print(f"<ZZ>: unmitigated = {zz_values[0]:.4f} (error {abs(zz_values[0] - exact_zz):.4f}), "
      f"ZNE = {zne_zz:.4f} (error {abs(zne_zz - exact_zz):.4f}), exact = {exact_zz:.4f}")
print(f"<IZ>: unmitigated = {iz_values[0]:.4f} (error {abs(iz_values[0] - exact_iz):.4f}), "
      f"ZNE = {zne_iz:.4f} (error {abs(zne_iz - exact_iz):.4f}), exact = {exact_iz:.4f}")
```

In this run, the extrapolated estimates are substantially closer to the exact values than the
unmitigated (scale factor $1$) results.
Other extrapolation methods only require changing the function handle --- for example Richardson
extrapolation, which fits a degree-two polynomial through the three points:

```{code-cell} ipython3
richardson_zz = zne.combine_results(
    scale_factors, zz_values, zne.inference.RichardsonFactory.extrapolate
)
print(f"Richardson <ZZ> = {richardson_zz:.4f} (error {abs(richardson_zz - exact_zz):.4f})")
```

Finally, we visualize the measured $\langle Z \otimes Z \rangle$ values and the linear fit used
for the extrapolation.

```{code-cell} ipython3
slope, intercept = np.polyfit(scale_factors, zz_values, 1)
fit_range = np.linspace(0, max(scale_factors), 100)

plt.plot(fit_range, slope * fit_range + intercept, "--", label="Linear fit")
plt.plot(scale_factors, zz_values, "o", label="Measured")
plt.plot(0, zne_zz, "s", label="ZNE estimate")
plt.axhline(exact_zz, color="gray", linestyle=":", label="Exact")
plt.xlabel("Noise scale factor")
plt.ylabel(r"$\langle Z \otimes Z \rangle$")
plt.legend()
plt.show()
```

## Why the two-step workflow instead of `execute_with_zne`?

Mitiq's high-level {func}`.zne.execute_with_zne` drives ZNE through an *executor* --- a function
from circuits to results.
A [batched executor](../guide/executors.md), annotated to return `list[float]`, receives all
noise-scaled circuits in a single call, so it *can* wrap a program set internally:

```{code-cell} ipython3
def batched_executor(circuits: list) -> list[float]:
    """Executes all circuits as one program set, measuring only <ZZ>."""
    noisy = [c.copy().apply_gate_noise(Noise.Depolarizing(NOISE_LEVEL)) for c in circuits]
    ps = ProgramSet.product(noisy, observables=[observables[0]])
    res = device.run(ps, shots=50_000 * len(circuits)).result()
    return [float(res.entries[i][0].expectation) for i in range(len(circuits))]


zne_zz_executor = zne.execute_with_zne(
    circuit,
    batched_executor,
    factory=zne.inference.LinearFactory(scale_factors=scale_factors),
    scale_noise=zne.scaling.fold_global,
)
print(f"<ZZ> via execute_with_zne with a batched executor: {zne_zz_executor:.4f}")
```

This works, and it still submits only one task per ZNE estimate.
To see why the two-step workflow fits program sets better, it helps to separate two contracts.
First, the raw executor contract: an executor returns one `QuantumResult` per circuit, where a
`QuantumResult` may be a float, a `MeasurementResult` (raw bitstrings), or a density matrix;
when the executor does not return floats directly, a Mitiq {class}`.Observable` passed to
`execute_with_zne` supplies the quantity to estimate, and Mitiq converts each circuit's result
into one scalar expectation value.
Second, the output contract: a single `execute_with_zne` call returns exactly one mitigated
scalar expectation value.

A native Braket program-set result is not itself one of the `QuantumResult` types, so a batched
executor must collapse it before Mitiq sees it --- above, by extracting one observable's
expectation values --- and the per-observable structure of the program-set result is discarded
at the executor boundary.
Independently reporting $m$ observables this way therefore takes $m$ separate
`execute_with_zne` calls, each building and running its own program set, even though a single
`product` program set already contains all the (circuit, observable) measurements.

The two-step workflow avoids this mismatch: {func}`.zne.construct_circuits` produces the scaled
circuits once, one program set executes every (circuit, observable) combination in a single
task, and {func}`.zne.combine_results` extrapolates each observable from the same result object.
Batched execution and multi-observable estimation stay decoupled from Mitiq's noise scaling and
inference.

```{warning}
This tutorial only uses the local simulator.
Program sets can also be submitted to AWS-managed simulators and QPUs by replacing
`LocalSimulator` with an `AwsDevice`, but doing so requires valid AWS credentials and incurs
monetary cost.
```

## Assumptions and limitations

A few caveats to keep in mind when interpreting the numbers above:

- **Shot noise.** All expectation values are estimated from a finite number of shots, and the
  local simulator does not accept a random seed, so repeated runs of this notebook give slightly
  different numbers.
  The shot count is chosen large enough that the qualitative conclusions are stable.
- **Idealized noise model.** Uniform depolarizing noise after every gate is a convenient
  abstraction.
  Real hardware exhibits coherent errors, crosstalk, readout error, and drift, none of which are
  captured here, so the improvement observed on hardware can differ substantially.
- **Extrapolation bias.** ZNE assumes the expectation value varies smoothly (here, linearly)
  with the scale factor.
  The true dependence under depolarizing noise is closer to exponential decay, so even with
  infinitely many shots the linear fit retains a small systematic bias.
- **Single run.** This notebook is one stochastic experiment, not a statistical study.
  ZNE reduces the error here, but it is not guaranteed to improve every estimate on every run,
  circuit, or device.
