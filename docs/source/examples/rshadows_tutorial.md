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

```{tags} shadows, cirq, intermediate
```

# Robust Shadow Estimation with Mitiq

**Corresponding to:** Min Li (minl2@illinois.edu)

This notebook demonstrates how to perform the robust shadow estimation protocol with Mitiq.


```{code-cell} ipython3
:tags: [remove-cell]
from functools import partialmethod
import tqdm
tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)
```

```{code-cell} ipython3
import cirq
import numpy as np
from mitiq import MeasurementResult
from mitiq.shadows import (
    classical_post_processing,
    pauli_twirling_calibrate,
    shadow_quantum_processing,
)
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)

np.random.seed(666)  # set random seed for reproducibility
```

The following flags control whether to re-run quantum measurements or load pre-saved results.

- If **True**, the measurements will be re-run.
- If **False**, the pre-saved results will be used.


```{code-cell} ipython3
import os
import pickle
import zipfile

run_quantum_processing = False
run_pauli_twirling_calibration = False

file_directory = "./resources"

if not run_quantum_processing:
    saved_data_name = "saved_data-rshadows"

    with zipfile.ZipFile(
        f"{file_directory}/rshadows-tutorial-{saved_data_name}.zip"
    ) as zf:
        saved_data = pickle.load(zf.open(f"{saved_data_name}.pkl"))
```

The *robust shadow estimation* protocol {cite}`chen2021robust`, building on {cite}`huang2020predicting`, exhibits noise resilience. The inherent randomization in the protocol simplifies the noise, transforming it into a Pauli noise channel that can be characterized relatively straightforwardly. Once the noisy channel $\widehat{\mathcal{M}}$ is characterized, it is incorporated into the channel inversion $\widehat{\mathcal{M}}^{-1}$, resulting in an unbiased state estimator. The sampling error in the determination of the Pauli channel contributes to the variance of this estimator.

## Define Quantum Circuit and Executor

In this notebook, we use the ground state of the Ising model with periodic boundary conditions to study energy and two-point correlation function estimation. We compare the performance of robust shadow estimation with the standard shadow protocol under bit-flip or depolarizing noise.

The Hamiltonian of the Ising model is given by
\begin{equation}
H = -J\sum_{i=0}^{n-1} Z_i Z_{i+1} -  g\sum_{i=1}^N X_i.
\end{equation}
We focus on the case $J = g = 1$ with 8 spins. The ground state circuit is loaded from:


```{code-cell} ipython3
# import ground state of the 1D Ising model with periodic boundary conditions
download_ising_circuits = True
num_qubits = 8
qubits: list[cirq.Qid] = cirq.LineQubit.range(num_qubits)

if download_ising_circuits:
    with open(f"{file_directory}/rshadows-tutorial-1D_Ising_g=1_{num_qubits}qubits.json", "rb") as file:
        circuit = cirq.read_json(json_text=file.read())
    g = 1

# or user can import from tensorflow_quantum
else:
    from tensorflow_quantum.datasets import tfi_chain

    qbs = cirq.GridQubit.rect(num_qubits, 1)
    circuits, labels, pauli_sums, addinfo = tfi_chain(qbs, "closed")
    lattice_idx = 40  # Critical point where g == 1
    g = addinfo[lattice_idx].g

    circuit = circuits[lattice_idx]
    qubit_map = {
        cirq.GridQubit(i, 0): cirq.LineQubit(i) for i in range(num_qubits)
    }

    circuit = circuit.transform_qubits(qubit_map=qubit_map)
```

As in the classical shadow protocol, we define an executor that performs circuit measurements. We add single-qubit depolarizing noise after the rotation gates but before the $Z$-basis measurement. Since the noise is assumed to be gate-independent, time-invariant, and Markovian, the noisy gate satisfies $U_{\Lambda_U}(M_z)_{\Lambda_{\mathcal{M}_Z}}\equiv U\Lambda\mathcal{M}_Z$:


```{code-cell} ipython3
def cirq_executor(
    circuit: cirq.Circuit,
    noise_model_function=cirq.depolarize,
    noise_level=(0.2,),
    sampler=cirq.Simulator(),
) -> MeasurementResult:
    """Return the measurement outcomes of a circuit with noise added before measurement.

    Args:
        circuit: The circuit to execute.

    Returns:
        A single-shot MeasurementResult containing the measurement outcomes.
    """
    tmp_circuit = circuit.copy()
    qubits = sorted(list(tmp_circuit.all_qubits()))
    if noise_level[0] > 0:
        noisy_circuit = cirq.Circuit()
        operations = list(tmp_circuit)
        n_ops = len(operations)
        for i, op in enumerate(operations):
            if i == n_ops - 1:
                noisy_circuit.append(
                    cirq.Moment(
                        *noise_model_function(*noise_level).on_each(*qubits)
                    )
                )
            noisy_circuit.append(op)
        tmp_circuit = noisy_circuit
    executor = cirq_sample_bitstrings(
        tmp_circuit,
        noise_model_function=None,
        noise_level=(0,),
        shots=1,
        sampler=sampler,
    )

    return executor
```

## Pauli Twirling Calibration

### PTM Representation

The PTM (Pauli Transfer Matrix) or Liouville representation provides a vector representation for all linear operators $\mathcal{L}(\mathcal{H}_d)$ on an $n$-qubit Hilbert space $\mathcal{H}_d$ (where $d = 2^n$). This representation uses the normalized Pauli operator basis $\sigma_a=P_a/\sqrt{d}$, with $P_a$ being the standard Pauli matrices.


```{code-cell} ipython3
from mitiq.utils import operator_ptm_vector_rep

operator_ptm_vector_rep(cirq.I._unitary_() / np.sqrt(2))
```

### Pauli Twirling of the Quantum Channel and Pauli Fidelity

The classical shadow estimation involves Pauli twirling of a quantum channel represented by $\mathcal{G} \subset U(d)$, with PTM representation $\mathcal{U}$. This twirling allows direct computation of $\widehat{\mathcal{M}}$ for the noisy channel $\Lambda$:
\begin{equation}
\widehat{\mathcal{M}} = \mathbb{E}_{\mathcal{G}}[\mathcal{U}^\dagger\mathcal{M}_z\Lambda\mathcal{U}]
\end{equation}
Local Clifford group projections are given by:
\begin{equation}
\Pi_{b_i}=\left\{
\begin{array}{ll}
|\sigma_0\rangle\!\rangle\langle\!\langle\sigma_0|& b_i=0 \\
\mathbb{I}- |\sigma_0\rangle\!\rangle\langle\!\langle\sigma_0|& b_i = 1
\end{array}\right.
\end{equation}
The Pauli fidelity for the local Clifford group is:
\begin{equation}
\hat{f}^{(r)}_b = \prod_{i=1}^n \langle\!\langle b_i|\mathcal{U}_i|P_z^{b_i}\rangle\!\rangle
\end{equation}
The final estimation uses the median-of-means estimator. See `get_single_shot_pauli_fidelity` and `get_pauli_fidelities` for implementation details.

### Noiseless Pauli Fidelity

In the ideal noise-free scenario, the Pauli fidelity is:
\begin{equation}
\hat{f}_{b}^{\mathrm{ideal}} = 3^{-|{b}|}
\end{equation}
For noisy channels, the inverse channel $\widehat{\mathcal{M}}^{-1}$ can be derived and used for robust shadow calibration, with deviations from the ideal values quantifying the noise.


```{code-cell} ipython3
from functools import partial

n_total_measurements_calibration = 20000
if run_quantum_processing:
    noisy_executor = partial(cirq_executor, noise_level=(0.1,))
    zero_state_shadow_output = shadow_quantum_processing(
        # zero circuit of 8 qubits
        circuit=cirq.Circuit(),
        num_total_measurements_shadow=n_total_measurements_calibration,
        executor=noisy_executor,
        qubits=qubits,
    )
else:
    zero_state_shadow_output = saved_data["shadow_outcomes_f_plot"]
f_est_results = pauli_twirling_calibrate(
    zero_state_shadow_outcomes=zero_state_shadow_output,
    k_calibration=5,
    locality=2,
)
```


```{code-cell} ipython3
# sort bitstrings by number of 1s
bitstrings = np.array(sorted(list(f_est_results.keys())))

counts = {bitstring: bitstring.count("1") for bitstring in bitstrings}
order = np.argsort(list(counts.values()))
reordered_bitstrings = bitstrings[order]

# compute theoretical Pauli fidelities for the noiseless case
f_theoretical = {}
bitstrings = list(f_est_results.keys())
for bitstring in bitstrings:
    n_ones = bitstring.count("1")
    f_val = 3 ** (-n_ones)
    f_theoretical[bitstring] = f_val
```


```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.plot(
    [np.abs(f_est_results[b]) for b in reordered_bitstrings],
    "-*",
    label="Noisy Channel",
)
plt.plot(
    [f_theoretical[b] for b in reordered_bitstrings], label="Noiseless Channel"
)
plt.xlabel(r"measurement basis states $b$")
plt.xticks(
    range(len(reordered_bitstrings)),
    reordered_bitstrings,
    rotation="vertical",
    fontsize=6,
)

plt.ylabel("Pauli fidelity")
plt.legend();
```


## Calibrated Operator Expectation Value Estimation

The expectation value for a series of operators $\{O_\iota\}_{\iota\leq M}$ has a snapshot estimator derived from the random Pauli measurement channel $\widetilde{\mathcal{M}}=\bigotimes_{i}\widetilde{\mathcal{M}}_{P_i}$ and the Pauli-twirling calibration $\widehat{\mathcal{M}}^{-1}=\sum_{b\in\{0,1\}^n}f_b^{-1}\bigotimes_{i}\Pi_{b_i\in b}$:
\begin{align}
\hat{o}_\iota &= \langle\!\langle O_\iota|{\hat{\rho}}\rangle\!\rangle \simeq \langle\!\langle O_\iota|\widehat{\mathcal{M}}^{-1}\widetilde{\mathcal{M}}|\rho\rangle\!\rangle=\sum_{b^{(1)}\in\{0,1\}^{n}}f_{b^{(1)}}^{-1}\left(\bigotimes_{i=1}^n \langle\!\langle P_i|\Pi_{b_i^{(1)}}\widehat{\mathcal{M}}_{P_i}\right)|\rho\rangle\!\rangle\nonumber\\
&=\sum_{b^{(1)}\in\{0,1\}^{n}}f_{b^{(1)}}^{-1}\prod_{i=1}^n \langle\!\langle P_i|\Pi_{b^{(1)}_i}\bigg|U_i^{(2)\dagger}|b_i^{(2)}\rangle\langle b_i^{(2)}|U_i^{(2)}\bigg\rangle\!\bigg\rangle
\end{align}
where $\{P_i\}_{i\leq n}$ are Pauli operators ($P\in\{I,X,Y,Z\}$), and superscripts $(1)$ and $(2)$ distinguish calibration from shadow-estimation quantities. Both conditions can be verified from the projection structure: the summand vanishes unless $\Pi_0$ acts on all sites outside $\mathrm{supp}(O_\iota)$ and $\Pi_1$ acts on all sites within $\mathrm{supp}(O_\iota)$, i.e.
\begin{align}
\Pi_i|I\rangle\!\rangle\equiv\delta_{i,0}|I\rangle\!\rangle,\qquad \Pi_{i}|P\rangle\!\rangle\equiv\delta_{i,1}|P\rangle\!\rangle,\qquad \mathrm{for}~i\in\{0,1\};~P\in\{X,\;Y,\;Z\}.
\end{align}
Therefore, the expectation value estimator simplifies to
\begin{align}
\hat{o}_\iota = \left(\sum_{b^{(1)}\in\{0,1\}^{n}}f_{b^{(1)}}^{-1}\prod_{j\in supp(O_\iota)}
\delta_{b_j^{(1)},1}\prod_{k\in supp(O_\iota)^c}\delta_{b_k^{(1)},0}\right)\prod_{i=1}^n  \langle b_i^{(2)}|U_i^{(2)}P_i U_i^{(2)\dagger}|b_i^{(2)}\rangle.
\end{align}
When $P_i = X_i$ (resp. $Y_i$, $Z_i$), $U_i^{(2)}$ must correspond to an $X$- (resp. $Y$-, $Z$-) basis measurement to yield a non-zero contribution. This is a direct consequence of the PTM representation of the single-Pauli measurement channel: $\widetilde{\mathcal{M}}_{P}=\frac{1}{2}(|I\rangle\!\rangle\langle\!\langle I|+|P\rangle\!\rangle\langle\!\langle P|)$.

The remaining steps follow the classical shadow protocol, using the median-of-means method with $R_2=N_2K_2$ total snapshots:
\begin{align}
\hat{o}_\iota(N_2,K_2):=\mathrm{median}\{\hat{o}_\iota^{(1)},\cdots,\hat{o}_\iota^{(K_2)}\}~~\mathrm{where}~~\hat{o}_\iota^{(j)}=N_2^{-1}\sum_{k=N_2(j-1)+1}^{N_2j}\mathrm{Tr}(O_\iota\hat{\rho}_k),\qquad \forall~1\leq j\leq K_2.
\end{align}

### Ground State Energy Estimation of the Ising Model

We compare the performance of robust and standard shadow estimation for ground state energy using the `compare_shadow_methods` helper function:


```{code-cell} ipython3
def compare_shadow_methods(
    circuit,
    observables,
    n_measurements_calibration,
    k_calibration,
    n_measurement_shadow,
    k_shadows,
    locality,
    noisy_executor,
    run_quantum_processing,
    shadow_measurement_result=None,
    zero_state_shadow_output=None,
):
    if run_quantum_processing:
        zero_state_shadow_output = shadow_quantum_processing(
            circuit=cirq.Circuit(),
            num_total_measurements_shadow=n_measurements_calibration,
            executor=noisy_executor,
            qubits=qubits,
        )
        shadow_measurement_result = shadow_quantum_processing(
            circuit,
            num_total_measurements_shadow=n_measurement_shadow,
            executor=noisy_executor,
        )
    else:
        assert shadow_measurement_result is not None
        assert zero_state_shadow_output is not None

    file_zsso = zero_state_shadow_output[1][0]
    file_k_cal = k_calibration
    file_locality = locality
    file_name = f"rshadows-tutorial-{file_zsso}-{file_k_cal}-{file_locality}"

    if not run_pauli_twirling_calibration and os.path.exists(f"{file_directory}/{file_name}.pkl"):
        with open(f"{file_directory}/{file_name}.pkl", "rb") as file:
            f_est = pickle.load(file)
    else:
        f_est = pauli_twirling_calibrate(
            zero_state_shadow_outcomes=zero_state_shadow_output,
            k_calibration=k_calibration,
            locality=locality,
        )

    output_shadow = classical_post_processing(
        shadow_outcomes=shadow_measurement_result,
        observables=observables,
        k_shadows=k_shadows,
    )

    output_shadow_cal = classical_post_processing(
        shadow_outcomes=shadow_measurement_result,
        calibration_results=f_est,
        observables=observables,
        k_shadows=k_shadows,
    )

    return {"standard": output_shadow, "robust": output_shadow_cal}
```

We use the ground state of the 1D Ising model with periodic boundary conditions, with $J = g = 1$, on 8 spins. The Hamiltonian is given by:


```{code-cell} ipython3
from mitiq import PauliString

# define the Ising model Hamiltonian as a list of observables
ising_hamiltonian = [
    PauliString("X", support=(i,), coeff=-g) for i in range(num_qubits)
] + [
    PauliString("ZZ", support=(i, (i + 1) % num_qubits), coeff=-1)
    for i in range(num_qubits)
]
```

Calculate the exact expectation values for comparison:


```{code-cell} ipython3
state_vector = circuit.final_state_vector()
expval_exact = []
for i, pauli_string in enumerate(ising_hamiltonian):
    exp = pauli_string._pauli.expectation_from_state_vector(
        state_vector, qubit_map={q: i for i, q in enumerate(qubits)}
    )
    expval_exact.append(exp.real)
```

We use the bit-flip channel as an example noise model. The bit-flip channel is a Pauli channel that flips the state of a qubit with probability $p$.


```{code-cell} ipython3
noise_levels = np.linspace(0, 0.06, 4)
noise_model = "bit_flip"

standard_results = []
robust_results = []
noise_model_fn = getattr(cirq, noise_model)
for noise_level in noise_levels:
    noisy_executor = partial(
        cirq_executor,
        noise_level=(noise_level,),
        noise_model_function=cirq.bit_flip,
    )

    experiment_name = f"{num_qubits}qubits_{noise_model}_{noise_level}"
    if run_quantum_processing:
        shadow_measurement_result, zero_state_shadow_output = None, None
    else:
        shadow_measurement_result = saved_data[experiment_name][
            "shadow_outcomes"
        ]
        zero_state_shadow_output = saved_data[experiment_name][
            "zero_shadow_outcomes"
        ]

    est_values = compare_shadow_methods(
        circuit=circuit,
        observables=ising_hamiltonian,
        n_measurements_calibration=60000,
        n_measurement_shadow=60000,
        k_shadows=6,
        locality=3,
        noisy_executor=noisy_executor,
        k_calibration=10,
        run_quantum_processing=False,
        shadow_measurement_result=shadow_measurement_result,
        zero_state_shadow_output=zero_state_shadow_output,
    )
    standard_results.append(est_values["standard"])
    robust_results.append(est_values["robust"])
```


```{code-cell} ipython3
import pandas as pd

rows_energy = []
for i, noise_level in enumerate(noise_levels):
    est_values = {}
    est_values["standard"] = list(standard_results[i].values())
    est_values["robust"] = list(robust_results[i].values())
    for ham, val in zip(ising_hamiltonian, expval_exact):
        rows_energy.append({
            "noise_level": noise_level,
            "method": "exact",
            "observable": str(ham),
            "value": val,
        })
    for method in ["standard", "robust"]:
        for ham, val in zip(ising_hamiltonian, est_values[method]):
            rows_energy.append({
                "noise_level": noise_level,
                "method": method,
                "observable": str(ham),
                "value": val,
            })

df_energy = pd.DataFrame(rows_energy)
```


```{code-cell} ipython3
df_hamiltonian = df_energy.groupby(["noise_level", "method"]).sum()
df_hamiltonian = df_hamiltonian.reset_index()
noise_model = "bit_flip"
```

```{code-cell} ipython3
palette = {"exact": "black", "robust": "red", "standard": "green"}

plt.figure()
sns.lineplot(
    data=df_hamiltonian,
    x="noise_level",
    y="value",
    hue="method",
    palette=palette,
    markers=True,
    style="method",
    dashes=False,
    errorbar=("ci", 95),
)
plt.title(f"Hamiltonian Estimation for {noise_model} Noise")
plt.xlabel("Noise Level")
plt.ylabel("Energy Value");
```


### Two-Point Correlation Function Estimation

Let's estimate the two-point correlation function $\langle Z_0 Z_i\rangle$ for a 16-spin 1D Ising model at its critical point ($g=1$).

We first load the ground state circuit for the 16-spin 1D Ising model with periodic boundary conditions:


```{code-cell} ipython3
num_qubits = 16
qubits = cirq.LineQubit.range(num_qubits)
if download_ising_circuits:
    with open(f"{file_directory}/rshadows-tutorial-1D_Ising_g=1_{num_qubits}qubits.json", "rb") as file:
        circuit = cirq.read_json(json_text=file.read())
    g = 1
else:
    qbs = cirq.GridQubit.rect(num_qubits, 1)
    circuits, labels, pauli_sums, addinfo = tfi_chain(qbs, "closed")
    lattice_idx = 40  # Critical point where g == 1
    g = addinfo[lattice_idx].g
    circuit = circuits[lattice_idx]
    qubit_map = {
        cirq.GridQubit(i, 0): cirq.LineQubit(i) for i in range(num_qubits)
    }
    circuit = circuit.transform_qubits(qubit_map=qubit_map)
```

Define the two-point correlation functions $\{\langle Z_0 Z_i\rangle\}_{0\leq i\leq n-1}$ as observables:


```{code-cell} ipython3
two_pt_correlation = [
    PauliString("ZZ", support=(0, i), coeff=-1) for i in range(1, num_qubits, 2)
]
```

Calculate the exact correlation function for comparison:


```{code-cell} ipython3
expval_exact = []
state_vector = circuit.final_state_vector()
for i, pauli_string in enumerate(two_pt_correlation):
    exp = pauli_string._pauli.expectation_from_state_vector(
        state_vector, qubit_map={q: i for i, q in enumerate(qubits)}
    )
    expval_exact.append(exp.real)
```

With depolarizing noise set to $0.1$, we compare the unmitigated and mitigated results:


```{code-cell} ipython3
noisy_executor = partial(cirq_executor, noise_level=(0.1,))
experiment_name = f"{num_qubits}qubits_depolarize_{noise_level}"
shadow_measurement_result = saved_data[experiment_name]["shadow_outcomes"]
zero_state_shadow_output = saved_data[experiment_name]["zero_shadow_outcomes"]

est_values = compare_shadow_methods(
    circuit=circuit,
    observables=two_pt_correlation,
    n_measurements_calibration=50000,
    n_measurement_shadow=50000,
    k_shadows=5,
    locality=2,
    noisy_executor=noisy_executor,
    k_calibration=5,
    run_quantum_processing=False,
    shadow_measurement_result=shadow_measurement_result,
    zero_state_shadow_output=zero_state_shadow_output,
)
```


```{code-cell} ipython3
qubit_idxes = [max(corr.support()) for corr in two_pt_correlation]
rows_corr = []
for method in ["standard", "robust"]:
    for corr, idx, val in zip(two_pt_correlation, qubit_idxes, est_values[method].values()):
        rows_corr.append({
            "method": method,
            "qubit_index": idx,
            "observable": str(corr),
            "value": val,
        })
for corr, idx, val in zip(two_pt_correlation, qubit_idxes, expval_exact):
    rows_corr.append({
        "method": "exact",
        "qubit_index": idx,
        "observable": str(corr),
        "value": val,
    })
df_corr = pd.DataFrame(rows_corr)
```


```{code-cell} ipython3
palette = {"exact": "black", "robust": "red", "standard": "green"}

plt.figure()
sns.lineplot(
    data=df_corr,
    x="qubit_index",
    y="value",
    hue="method",
    palette=palette,
    markers=True,
    style="method",
    dashes=False,
    errorbar=("ci", 95),
)
plt.title("Correlation Function Estimation with 0.1 Depolarizing Noise")
plt.xlabel(r"Correlation Function $\langle Z_0Z_i \rangle$")
plt.ylabel("Correlation");
```
