# Debiasing

Debiasing, also called symmetrization, is a noise-tailoring technique for
reducing the effect of qubit-dependent errors. The circuit of interest is run
as several randomized variants, each one relabeled onto a different, randomly
permuted set of qubits ($C \rightarrow \pi C$ for a permutation $\pi$). The
permutation is undone on the measured bitstrings before the results are
combined, so on a noiseless device every variant reproduces the same ideal
distribution. On hardware, each variant assigns the logical qubits to a
different set of physical qubits, so errors that depend on which qubit is used
are sampled differently and average out when the unscrambled distributions are
combined. See {cite}`Maksymov_2023_arxiv` for the original description.

Mitiq also implements the optional *sharpening* step. Instead of averaging the
variant distributions, sharpening performs a shot-wise plurality vote and keeps
the bitstring that occurs most often across the variants. This is useful when
the answer is concentrated on a few bitstrings, as in optimization or
eigenvalue problems.

```{warning}
Debiasing lives in `mitiq.experimental`. Its API may change without notice and
is not covered by Mitiq's semantic versioning guarantees.
```

## Example

Debiasing works on measurement distributions rather than expectation values, so
the executor returns a {class}`.MeasurementResult` rather than a float.

```python
import cirq

from mitiq import MeasurementResult
from mitiq.experimental.deb import execute_with_debiasing

qreg = cirq.LineQubit.range(2)
circuit = cirq.Circuit([cirq.H(qreg[0]), cirq.CNOT(*qreg)])


def executor(circuit: cirq.Circuit) -> MeasurementResult:
    """Sample the circuit on a noisy simulator."""
    simulator = cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.02))
    measured = circuit + cirq.measure(*sorted(circuit.all_qubits()), key="m")
    samples = simulator.run(measured, repetitions=2000)
    return MeasurementResult(samples.measurements["m"].tolist())


distribution = execute_with_debiasing(circuit, executor, num_variants=10)
print({bits: round(prob, 3) for bits, prob in distribution.items()})
```

The returned dictionary maps each measured bitstring to its averaged
probability. To apply the sharpening step instead, pass
`method="sharpening"` to {func}`.execute_with_debiasing`.

Bitstrings are ordered to match `sorted(circuit.all_qubits())`: the first
character corresponds to the first qubit in that sorted order, and so on. The
same ordering is used by {func}`~mitiq.experimental.deb.symmetrization.construct_circuits` and
{func}`~mitiq.experimental.deb.deb.combine_results`, which is why the permutation returned by
`construct_circuits` must be passed to `combine_results` so it can be undone on
the measured bitstrings.
