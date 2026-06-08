# Debiasing

Debiasing, also called symmetrization, is a noise-tailoring technique for
reducing the effect of coherent errors. The circuit of interest is run as
several randomized variants, each one conjugated by a layer of random
single-qubit Pauli gates: a Pauli is applied to a qubit before the circuit and
the same Pauli is applied again afterwards. Because Pauli gates are their own
inverse the layer cancels on a noiseless device, so every variant computes the
same ideal result. Coherent errors, on the other hand, are dressed differently
in each variant and partially average out when the measurement distributions
are combined. The technique adds no extra qubits or two-qubit gates beyond the
pre- and post-circuit Pauli layer. See {cite}`Maksymov_2023_arxiv` for the
original description.

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
import numpy as np

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
print(np.round(distribution, 3))
```

The returned array is the averaged probability over the computational basis. To
apply the sharpening step instead, use
{func}`.execute_with_debiasing_and_sharpening` with the same signature.
