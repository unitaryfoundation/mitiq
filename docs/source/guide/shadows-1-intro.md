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

```{admonition} Note:
The documentation for Classical Shadows in Mitiq is still under construction. This users guide will change in the future.
```

# How Do I Use Classical Shadows Estimation?

The `mitiq.experimental.shadows` module facilitates the application of the classical shadows protocol on quantum circuits, designed for tasks like quantum state tomography or expectation value estimation. In addition this module integrates a robust shadow estimation protocol that's tailored to counteract noise. The primary objective of the classical shadow protocol is to extract information from a quantum state using repeated measurements.

The procedure can be broken down as follows:

1. `shadow_quantum_processing`:

   - Purpose: Execute quantum processing on the provided quantum circuit.
   - Outcome: Measurement results from the processed circuit.

2. `classical_post_processing`:
   - Purpose: Handle classical processing of the measurement results.
   - Outcome: Estimation based on user-defined inputs.

For users aiming to employ the robust shadow estimation protocol, an initial step is needed which entails characterizing the noisy quantum channel. This is done by:

0. `pauli_twirling_calibration`

   - Purpose: Characterize the noisy quantum channel.
   - Outcome: A dictionary of `calibration_results`.

1. `shadow_quantum_processing`: same as above.

2. `classical_post_processing`
   - Args: `calibration_results` = output of `pauli_twirling_calibration`
   - Outcome: Error mitigated estimation based on user-defined inputs.

Notes:

- The calibration process is specifically designed to mitigate noise encountered during the classical shadow protocol, such as rotation and computational basis measurements. It does not address noise that occurs during state preparation.
- Do not need to redo the calibration stage (0. `pauli_twirling_calibration`) if:
  1.  The input circuit has a consistent number of qubits.
  2.  The estimated observables have the same or fewer qubit support.

## Protocol Overview

The classical shadow protocol aims to create an approximate classical representation of a quantum state using minimal measurements. This approach not only characterizes and mitigates noise effectively but also retains sample efficiency and demonstrates noise resilience. For more details, see the section ([What is the theory behind Classical Shadow Estimation?](shadows-5-theory.md)).

One can use the `mitiq.experimental.shadows' module as follows.

### User-defined inputs

Define a quantum circuit, e.g., a circuit which prepares a GHZ state with $n$ = `3` qubits,

```{code-cell} ipython3
import numpy as np
#fix random seed
np.random.seed(1)
```

```{code-cell} ipython3
import cirq

qubits = cirq.LineQubit.range(3)
num_qubits = len(qubits)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
)
print(circuit)
```

Define an executor to run the circuit on a quantum computer or a noisy simulator. Note that the _robust shadow estimation_ technique can only calibrate and mitigate the noise acting on the operations associated to the classical shadow protocol. So, in order to test the technique, we assume that the state preparation part of the circuit is noiseless. In particular, we define an executor in which:

1. A noise channel is added to circuit right before the measurements. I.e. $U_{\Lambda_U}(M_z)_{\Lambda_{\mathcal{M}_Z}}\equiv U\Lambda\mathcal{M}_Z$.

2. A single measurement shot is taken for each circuit, as required by classical shadow protocol.

```{code-cell} ipython3
from mitiq import MeasurementResult


def cirq_executor(
    circuit: cirq.Circuit,
    noise_model_function=cirq.depolarize,
    noise_level=(0.2,),
    sampler=cirq.Simulator(),
) -> MeasurementResult:
    """
    This function returns the measurement outcomes of a circuit with noisy channel added before measurements.
    Args:
        circuit: The circuit to execute.
    Returns:
        A one shot MeasurementResult object containing the measurement outcomes.
    """
    circuit = circuit.copy()
    qubits = sorted(list(circuit.all_qubits()))
    if noise_level[0] > 0:
        noisy_circuit = cirq.Circuit()
        operations = list(circuit)
        n_ops = len(operations)
        for i, op in enumerate(operations):
            if i == n_ops - 1:
                noisy_circuit.append(
                    cirq.Moment(
                        *noise_model_function(*noise_level).on_each(*qubits)
                    )
                )
            noisy_circuit.append(op)
        circuit = noisy_circuit
    executor = cirq_sample_bitstrings(
        circuit,
        noise_model_function=None,
        noise_level=(0,),
        shots=1,
        sampler=sampler,
    )
    return executor
```

Given the above general executor, we define a specific example of a noisy executor, assuming a bit flip channel with a probability of `0.1'

```{code-cell} ipython3
from functools import partial

noisy_executor = partial(
    cirq_executor,
    noise_level=(0.1,),
    noise_model_function=cirq.bit_flip,
)
```

### 0. Calibration Stage

One can simply skip this stage if one just wants to perform the classical shadow protocol (without calibration). This step can also be skipped if calibration data is already available from previous runs.

By setting the total calibration rounds $R$ = `num_total_measurements_calibration` and the number of groups for the "median of means" estimation used for calibration $K$ = `k_calibration`, we can characterize the noisy quantum channel (see [this tutorial](../examples/rshadows_tutorial.md) for more details) by running the following code:

```{code-cell} ipython3
import sys
sys.modules["tqdm"] = None # disable tqdm for cleaner notebook rendering
from mitiq.experimental import shadows
from mitiq.experimental.shadows.shadows import (
    pauli_twirling_calibrate,
    shadow_quantum_processing,
    classical_post_processing,
)
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)

f_est = pauli_twirling_calibrate(
    k_calibration=1,
    locality=2,
    qubits=qubits,
    executor=noisy_executor,
    num_total_measurements_calibration=5000,
)
f_est
```

the varible `locality` is the maximum number of qubits on which our operators of interest are acting on. E.g. if our operator is a sequence of two point correlation terms $\{\langle Z_iZ_{i+1}\rangle\}_{0\leq i\leq n-1}$, then `locality` = 2. We note that one could also split the calibration process into two stages:

1.  `shadow_quantum_processing`

- Outcome: Get quantum measurement result of the calibration circuit $|0\rangle^{\otimes n}$ `zero_state_shadow_outcomes`.

2.  `pauli_twirling_calibration`

- Outcome: A dictionary of `calibration_results`.
  For more details, please refer to [this tutorial](../examples/rshadows_tutorial.md)

### 1. Quantum Processing

In this step, we obtain classical shadow snapshots from the input state (before applying the invert channel).

#### 1.1 Add Rotation Gate and Meausure the Rotated State in Computational Basis

At present, the implementation supports random Pauli measurement. This is equivalent to randomly sampling $U$ from the local Clifford group $Cl_2^n$, followed by a $Z$-basis measurement (see [this tutorial](../examples/shadows_tutorial.md) for a clear explanation).

#### 1.2 Get the Classical Shadows

One can obtain the list of measurement results of local Pauli measurements in terms of bitstrings, and the related Pauli-basis measured in terms of strings as follows.

You have two choices: run the quantum measurement or directly use the results from the previous run.

- If **True**, the measurement will be run again.
- If **False**, the results from the previous run will be used.

```{code-cell} ipython3
import zipfile, pickle, io, requests

run_quantum_processing = False
run_pauli_twirling_calibration = False

file_directory = "../examples/resources"

if not run_quantum_processing:
    saved_data_name = "shadows-1-intro-output1"
    with open(f"{file_directory}/{saved_data_name}.pkl", "rb") as file:
        shadow_measurement_output = pickle.load(file)

else:
    shadow_measurement_output = shadow_quantum_processing(
        circuit,
        noisy_executor,
        num_total_measurements_shadow=5000,
    )
```

As an example, we print out one of those measurement outcomes and the associated measured operator:

```{code-cell} ipython3
print("one snapshot measurement result = ", shadow_measurement_output[0][0])
print("one snapshot measurement basis = ", shadow_measurement_output[1][0])
```

### 2. Classical Post-Processing

In this step, we estimate our object of interest (expectation value or density matrix) by post-processing the (previously obtained) measurement outcomes.

#### 2.1 Example: Operator Expectation Value Esitimation

For example, if we want to estimate the two point correlation function $\{\langle Z_iZ_{i+1}\rangle\}_{0\leq i\leq n-1}$, we will define the corresponding Puali strings:

```{code-cell} ipython3
from mitiq import PauliString

two_pt_correlations = [
    PauliString("ZZ", support=(i, i + 1), coeff=1)
    for i in range(0, num_qubits - 1)
]
for i in range(0, num_qubits - 1):
    print(two_pt_correlations[i]._pauli)
```

The corresponding expectation values can be estimated (with and without calibration) as shown in the next code cell.

```{code-cell} ipython3
est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    observables=two_pt_correlations,
    k_shadows=1,
)
cal_est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    calibration_results=f_est,
    observables=two_pt_correlations,
    k_shadows=1,
)
```

Let's compare the results with the exact theoretical values:

```{code-cell} ipython3
expval_exact = []
state_vector = circuit.final_state_vector()
for i, pauli_string in enumerate(two_pt_correlations):
    exp = pauli_string._pauli.expectation_from_state_vector(
        state_vector, qubit_map={q: i for i, q in enumerate(qubits)}
    )
    expval_exact.append(exp.real)
```

```{code-cell} ipython3
print("Classical shadow estimation:", est_corrs)
print("Robust shadow estimation   :", cal_est_corrs)
print(
    "Exact expectation values:",
    "'Z(q(0))*Z(q(1))':",
    expval_exact[0],
    "'Z(q(1))*Z(q(2))':",
    expval_exact[1],
)
```

#### 2.2 Example: GHZ State Reconstruction

In addition to the estimation of expectation values, the `mitiq.shadow` module can also be used to reconstruct an approximated version of the density matrix.
As an example, we use the 3-qubit GHZ circuit, previously defined. As a first step, we calculate the Pauli fidelities $f_b$ characterizing the noisy quantum channel $\mathcal{M}=\sum_{b\in\{0,1\}^n}f_b\Pi_b$:

```{code-cell} ipython3
noisy_executor = partial(
    cirq_executor,
    noise_level=(0.2,),
    noise_model_function=cirq.bit_flip,
)

if not run_pauli_twirling_calibration:
    saved_data_name = "shadows-1-intro-PTC-50000"
    with open(f"{file_directory}/{saved_data_name}.pkl", "rb") as file:
        f_est = pickle.load(file)

else:
    f_est = pauli_twirling_calibrate(
        k_calibration=1,
        qubits=qubits,
        executor=noisy_executor,
        num_total_measurements_calibration=50000
    )

f_est
```

Similar to the previous case (estimation of expectation values), the quantum processing for estimating the density matrix is done as follows.

```{code-cell} ipython3
if not run_quantum_processing:
    saved_data_name = "shadows-1-intro-output2"
    with open(f"{file_directory}/{saved_data_name}.pkl", "rb") as file:
        shadow_measurement_output = pickle.load(file)

else:
    shadow_measurement_output = shadow_quantum_processing(
        circuit,
        noisy_executor,
        num_total_measurements_shadow=50000,
    )
```

```{code-cell} ipython3
est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    state_reconstruction=True,
)
cal_est_corrs = classical_post_processing(
    shadow_outcomes=shadow_measurement_output,
    calibration_results=f_est,
    state_reconstruction=True,
)
```

Let's compare the fidelity between the reconstructed state and the ideal state.

```{code-cell} ipython3
from mitiq.utils import operator_ptm_vector_rep

ghz_state = circuit.final_state_vector().reshape(-1, 1)
ghz_true = ghz_state @ ghz_state.conj().T
ptm_ghz_state = operator_ptm_vector_rep(ghz_true)
```

```{code-cell} ipython3
from mitiq.experimental.shadows.shadows_utils import fidelity

fidelity_shadow = fidelity(ghz_true, est_corrs["reconstructed_state"])
fidelity_shadow_calibrated = fidelity(
    ptm_ghz_state, cal_est_corrs["reconstructed_state"]
)
print(
    f"fidelity between true state and shadow reconstruced state {fidelity_shadow}"
)
print(
    f"fidelity between true state and rshadow reconstruced state {fidelity_shadow_calibrated}"
)
```
