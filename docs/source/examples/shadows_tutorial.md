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

```{tags} cirq, shadows, intermediate
```

# Classical Shadows Protocol with Cirq

**Corresponding to:** Min Li (minl2@illinois.edu)

This notebook shows how to use classical shadows estimation with the Mitiq library, focused initially on local (Pauli) measurements. We show some common scenarios such as state tomography, and operator expectation value estimation. The method creates an approximate classical description of a quantum state with few measurements while effectively characterizing and mitigating noise in the [following notebook](https://mitiq.readthedocs.io/en/stable/examples/rshadows_tutorial.html).

```{code-cell} ipython3
import cirq
import numpy as np
import sys
sys.modules["tqdm"] = None # distable tqdm for cleaner notebook rendering
from mitiq.shadows.shadows import *
from mitiq.shadows.shadows_utils import *
from mitiq import MeasurementResult
from mitiq.interface.mitiq_cirq.cirq_utils import (
    sample_bitstrings as cirq_sample_bitstrings,
)
# set random seed
np.random.seed(666)
```

In the context of an $n$-qubit system, where $\rho$ is an unknown quantum state residing in a $2^n$-dimensional Hilbert space, the procedure of performing classical shadow involves extracting information from the state through repeated measurements. 

## 1. Define a test circuit


```{code-cell} ipython3
# number of qubits in the circuit
num_qubits: int = 4
# qubits in the circuit prepared in the $|0\rangle$ state
qubits: list[cirq.Qid] = cirq.LineQubit.range(num_qubits)

# defining random parameters for the circuit
# np.random.seed(666)
params: np.ndarray = np.random.randn(2 * num_qubits)

# define circuit
def simple_test_circuit(
    params: np.ndarray, qubits: list[cirq.Qid]
) -> cirq.Circuit:
    circuit: cirq.Circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.ry(params[i])(qubit))
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rz(params[i + num_qubits])(qubit))
    return circuit


# print the circuit
test_circuits = simple_test_circuit(params, qubits)
print(simple_test_circuit(params, qubits))
```

<script>
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>

## 2. Sampling random Pauli measurements 

This process involves applying a random unitary selected from a randomly fixed ensemble $\mathcal{U}\in U(2^n)$ to rotate the state $\rho\rightarrow U^\dagger \rho U$, followed by a computational-basis($Z$-basis) measurement, and storing a classical description $U^\dagger |\hat{b}\rangle\langle\hat{b}| U$. After the measurement, the inverse of $U$ is applied to the resulting computational basis state, collapsing $\rho$ to
  
\begin{equation}
U^\dagger|\hat{b}\rangle\langle\hat{b}| U\qquad \mathrm{where} \qquad \mathrm{Pr}[\hat{b}=b]=\langle b|U\rho U^\dagger|b\rangle.
\end{equation}
If the unitary group $\mathcal{U}$ is chosen to be the local Clifford group $\mathrm{CL}(2)^n$, this equivalent to performing a random Pauli measurement on each qubit. This means that for each qubit, we randomly decide to measure one of the Pauli operators. Below, we define the `cirq_executor` to take one shot of measurement and return the measurement result.


```{code-cell} ipython3
def cirq_executor(
    circuit: cirq.Circuit,
) -> MeasurementResult:
    return cirq_sample_bitstrings(
        circuit,
        noise_level=(0,),
        shots=1,
        sampler=cirq.Simulator(),
    )
```

In terms of implementation, considering that the only possible measurement to be performed is the $Z$-basis measurement, the random Pauli measurement is equivalent to randomly sampling a unitary from the unitary ensemble: $\mathcal{G}=\{\mathrm{id},\mathrm{H},\mathrm{H}\cdot \mathrm{S}^\dagger\}$. Afterward, the $Z$-basis measurement is conducted. We then record a sequence of Pauli gates $u_i:= U_i^\dagger ZU_i$ that have been measured for each qubit in the circuit. This sequence becomes one of the output lists of the measurement function `random_pauli_measurement`.

In the main function, the quantum measurement process is encapsulated within the `shadow_quantum_processing function`. This function takes the quantum circuit and the number of shots as input. It returns the measurement results as bit strings, for example, '01...0' is equivalent to the measurement basis eigenstate: $|0\rangle|1\rangle...|0\rangle$. Additionally, it provides the measured Pauli gates in string format. For instance, 'XY...Z' signifies a local X-basis measurement on the first qubit, a local Y-basis measurement on the second qubit, and a local Z-basis measurement on the last qubit in the circuit.


```{code-cell} ipython3
shadow_quantum_processing(test_circuits, cirq_executor, 2)
```

## 3. Obtain Snapshot and Classical Shadows.

This random measurement contains valuable information about $\rho$ in expectation:
\begin{equation}
    \mathbb{E}[U^\dagger |\hat{b}\rangle\langle\hat{b}|U]=\mathcal{M}(\rho),
\end{equation}
the expectation in the first expression has the form $\mathbf{Pr}[\hat{{b}}={b}]=\langle {b}|U\rho U^\dagger|b\rangle$. For any unitary ensemble $\mathcal{U}$, the expected value of the outer product of the classical snapshot corresponds to the operation of the quantum channel $\mathcal{M}$ on the quantum state $\rho$. If the measurements we sample from are tomographically complete, then the protocol $\mathcal{M}$ defines an invertible linear transformation $\mathcal{M}^{-1}$, which may not be a quantum channel, since it is not CP, which means that it could not be performed in the lab. But it will only be performed on the classical data stored in
a classical memory. If we apply $\mathcal{M}$ to all the snapshots, the expected value of these inverted snapshots equations with the density operator as defined by the protocol,

\begin{equation}
\hat{\rho}=\mathcal{M}^{-1}\left(U^\dagger|\hat{b}\rangle\langle\hat{b}|U\right)
\end{equation}
which has been named a single copy of **classical shadow**. Based on *Schur's Lemma* the quantum channel $\mathcal{M}$ is a depolarizing channel $\mathcal{D}_p$ with $p=\frac{1}{2^n+1}$. It is easy to solve for the inverted map 

\begin{equation}
\mathcal{M}^{-1}(\cdot)=[(2^n +1)-\mathbb{I}\cdot\mathrm{Tr}](\cdot),
\end{equation}
 which is indeed unitary, however, not CP, so it is not a physical map as expected.

In the case of random Pauli measurement, the unitary could be represented by the tensor product of all qubits, so it is with the state $|\hat{b}\rangle\in\{0,1\}^{\otimes n}$, i.e. $U^\dagger|\hat{b}\rangle=\bigotimes_{i\leq n}U_i|\hat{b}_i\rangle$. Therefore, based on Schur's Lemma, a snapshot would take the form:
\begin{equation}
\hat{\rho}=\bigotimes_{i=1}^{n}\left(3U_i^\dagger|\hat{b}_i\rangle\langle\hat{b}_i|U_i-\mathbb{I}\right),\qquad|\hat{b}_i\rangle\in\{0,1\}.
\end{equation}
which is a tensor product of $n$ qubits, each of which is a classical state. This step is realized by `classical_snapshot` function. Repeating this procedure $N$ times results in an array of $N$ independent classical snapshots of $\rho$:
\begin{equation}
    S(\rho,\; N)=\left\{\hat{\rho}_1=\mathcal{M}^{-1}\left(U_1^\dagger |\hat{b}_1\rangle\langle\hat{b}_1| U_1\right),\dots,\mathcal{M}^{-1}\left(U_N^\dagger |\hat{b}_N\rangle\langle\hat{b}_N| U_N\right)\right\} .
\end{equation}

## 4. State Reconstruction from Classical Shadows
### 4.1 State Reconstruction
The classical shadows state reconstruction are then obtained by taking the average of the snapshots, this process is designed to reproduce the underlying state $\rho$ exactly in expectation:
\begin{equation}
   \rho= \mathbb{E}[\hat{\rho}],
\end{equation}
this is realized in the function `state_reconstruction`. In the main function `classical_post_processing`, we take the output of `shadow_quantum_processing`, then apply the inverse channel to obtain the snapshots, and finally take the average of the snapshots to obtain the reconstructed state if *state_reconstruction =* **True**. **In the current notebook, we don't preform Pauli twirling calibration, and we set** *rshadow* = **False**.

#### 4.1.1 Error Analysis
We can take a visualization of the element wise difference between the reconstructed state and the original state. 
\begin{equation}
\Delta\rho_{ij}=|\rho^{\mathrm{shadow}}_{ij}-\rho_{ij}|
\end{equation}
The difference is very small, which means that the classical shadow is a good approximation of the original state even in the sense of state tomography. 

It is anticipated that the fidelity will not necessarily be lower than 1, as the state reconstructed through classical shadow estimation is not guaranteed to be a physical quantum state, given that $\mathcal{M}^{-1}$ is not a quantum channel. 

Fidelity is defined by $F(\rho,\sigma)=\mathrm{Tr}\sqrt{\rho^{1/2}\sigma\rho^{1/2}}$, when $\rho=|v\rangle\langle v|$ is a pure state $F(\rho,\sigma)=\langle v|\sigma|v\rangle$.
Based on the theorem, if the error rate of fidelity is $\epsilon$, i.e.
\begin{equation}
|F(\rho,\sigma)-1|\leq\epsilon,
\end{equation}
then the minimum number of measurements $N$ (number of snapshots) should be:
```{math}
:label: eq-label
N = \frac{34}{\epsilon^2}\left\|\rho-\mathrm{Tr}(\rho)/{2^n}\mathbb{I}\right\|_{\mathrm{shadow}}^2
```
with the shadow norm upper bound of the random Pauli measurement $\left\|\cdot\right\|_{\mathrm{shadow}}\leq 2^k\|\cdot\|_\infty$ when the operator acting on $k$ qubits, we have $N\leq 34\epsilon^{-2}2^{2n}+\mathcal{O}(e^{-n})$. Based on Fuchs–van de Graaf inequalities and properties of $L_p$ norm, $\|\rho-\sigma\|_2\leq \|\rho-\sigma\|_1 \leq (1-F(\rho,\sigma))^{1/2}$, the $L_2$ norm distance between the state reconstructed through classical shadow estimation and the state prepared by the circuit is upperbound by the fidelity error rate $\epsilon$. The dependency of the bound number of measurements $N$ to achieve the error rate $\epsilon$ is depicted in function `n_measurements_tomography_bound`.

```{note}
Equation {eq}`eq-label` comes from equation S13 in the paper {cite}`huang2020predicting`. It contains some numerical constants and as noted by Remark 1 these constants result from a worst case argument. You may see values much smaller in practice. 
```

```{code-cell} ipython3
# error rate of state reconstruction epsilon < 1.
epsilon = 1
# number of total measurements should perform for error rate epsilon
n_total_measurements = n_measurements_tomography_bound(epsilon, num_qubits)

print("n_total_measurements = {}".format(n_total_measurements))
shadow_outcomes = shadow_quantum_processing(
    test_circuits, cirq_executor, n_total_measurements
)
```                                                                      


```{code-cell} ipython3
# get shadow reconstruction of the density matrix
output = classical_post_processing(
    shadow_outcomes,
    state_reconstruction=True,
)
rho_shadow = output["reconstructed_state"]
```


```{code-cell} ipython3
# Compute the ideal state vector described by the input circuit.
state_vector = test_circuits.final_state_vector().reshape(-1, 1)
# Compute the density matrix.
rho_true = state_vector @ state_vector.conj().T
```

We can plot the element wise difference between the reconstructed state and the original state as a thermal diagram:


```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Setting the style
sns.set_style("white")

# Calculate min and max values across the datasets
vmin = min(
    np.abs(rho_shadow).min(),
    np.abs(rho_true).min(),
    np.abs(rho_true - rho_shadow).min(),
)
vmax = max(
    np.abs(rho_shadow).max(),
    np.abs(rho_true).max(),
    np.abs(rho_true - rho_shadow).max(),
)

# Creating a figure with three subplots (1 row, 3 columns)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plotting the first image on the first subplot
im1 = axs[0].imshow(np.real(rho_shadow), cmap="Blues", vmin=vmin, vmax=vmax)
axs[0].set_title(r"$\rho_{ij}^{\mathrm{shadow}}$")

# Plotting the second image on the second subplot
im2 = axs[1].imshow(np.real(rho_true), cmap="Blues", vmin=vmin, vmax=vmax)
axs[1].set_title(r"$\rho_{ij}$")

# Plotting the third image on the third subplot
im3 = axs[2].imshow(
    np.abs(rho_true - rho_shadow), cmap="Blues", vmin=vmin, vmax=vmax
)
axs[2].set_title(r"$|\rho_{ij}^{\mathrm{shadow}}-\rho_{ij}|$")

# Adjust the space between plots
plt.subplots_adjust(wspace=0.3)

# Add a shared colorbar
cbar = fig.colorbar(
    im3, ax=axs.ravel().tolist(), orientation="vertical", shrink=0.67, pad=0.05
)

# Show the figure with three side-by-side plots
plt.show()
```

Compute the fidelity and $L_2$ distance between the state reconstructed through classical shadow estimation (which is not a quantum state) and the state prepared by the circuit $\|\rho_{\mathrm{shadow}}-\rho\|_2$, with $\|\cdot\|_2:=\sqrt{\mathrm{Tr}[(\cdot)^\dagger(\cdot)]}$.


```{code-cell} ipython3
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# compute fidelity and operator 2-norm
b_strings, u_strings = shadow_outcomes
b_lists = np.array(b_strings)
u_lists = np.array(u_strings)
n_measurement_list = np.arange(
    int(n_total_measurements / 100),
    n_total_measurements,
    int(n_total_measurements / 10),
)
# repeat the experiment 3 times
n_runs = 3
fidelity_list = []
operator_2_norm_list = []

# Create a new dataframe to store the results
df = pd.DataFrame(
    columns=["n_measurement", "run", "fidelity", "operator_2_norm"]
)

# Loop over the different number of measurements
for n_measurement in n_measurement_list:
    # Repeat the experiment 3 times
    for run in range(n_runs):
        # randomly sample from the measurement outcomes, with replacement
        sample_idx = np.random.choice(
            len(b_lists), size=n_measurement, replace=True
        )
        shadow_subset = (b_lists[sample_idx], u_lists[sample_idx])
        # perform shadow state reconstruction
        rho_shadow = classical_post_processing(
            shadow_outcomes=shadow_subset,
            state_reconstruction=True,
        )["reconstructed_state"]

        # compute fidelity and operator 2-norm
        fidelity_val = fidelity(rho_true, rho_shadow)
        norm_val = np.linalg.norm(
            rho_shadow - rho_true, ord=None, axis=None, keepdims=False
        )
        # store the results
        df = pd.concat(
            [df, pd.DataFrame({
                "n_measurement": [n_measurement],
                "run": [run],
                "fidelity": [fidelity_val],
                "operator_2_norm": [norm_val],
            })],
            ignore_index=True,
        )
        
```




```{code-cell} ipython3
plt.figure()
sns.lineplot(
    data=df,
    x="n_measurement",
    y="fidelity",
)
plt.xlabel("Number of Measurements")
plt.legend()
plt.title(r"Fidelity: $F(\rho^{\rm shadow}, \rho)$")
plt.xlabel(r"N Measurements-$N$")
plt.ylabel(r"Fidelity")
# plot x range
plt.xlim(1000, n_total_measurements)

plt.figure()
sns.lineplot(
    data=df,
    x="n_measurement",
    y="operator_2_norm",
)
plt.title(r"$L_2$-Norm distance: $\|\rho^{\rm shadow} - \rho\|_2$")
plt.xlabel(r"N Measurements-$N$")
plt.ylabel(r"$L_2$ distance")
plt.legend()
# plot range 0.35 to 1.0
plt.ylim(0.35, 1.0)
# plot x range
plt.xlim(1000, n_total_measurements)
```

### 4.2 Use Classical Shadows to Estimate Expectation Values of Observables

To estimate the expectation value of some observable, we simply replace the unknown quantum state $\rho$ with a classical shadow $\hat{\rho}$. Since classical shadows are random, this produces a random variable that yields the correct prediction in expectation:
\begin{equation}
    \hat{o}_i = \mathrm{Tr}(O_i\hat{\rho})~~~\mathrm{obeys}\qquad \mathrm{Tr}(O_i\rho)\equiv \mathbb{E}[\hat{o}_i],~~ 1\leq i\leq M.
\end{equation}
 
One can prove that a snapshot can correctly predict **any** linear functions $f(\rho)$ of state, e.g. expectation values of obsevables $O_i$, i.e. $o_i=\mathrm{Tr}(O_i\rho)$, by taking average over the repeatedly $N$ independent classical shadows, 

\begin{equation}
\hat{o}_i(N)=\mathbb{E}_{j\in N}(\hat{o}_i^{(j)}\hat{\rho}_j)
\end{equation}

Actually in practical, with the statistical method of taking an average called "median of means" to achieve an acceptable failure probability of estimation, which need $R=NK$ snapshots acctually,
\begin{equation}
\hat{o}_i(N,K):=\mathrm{median}\{\hat{o}_i^{(1)},\cdots,\hat{o}_i^{(K)}\}~~\mathrm{where}~~\hat{o}_i^{(j)}=N^{-1}\sum_{k=N(j-1)+1}^{Nj}\mathrm{Tr}(O_i\hat{\rho}_k),
\end{equation}
for all $1\leq j\leq K$

 Now let's assume that our list of observables are a set of nearest nearist neighbour interactions on a 1D lattice, i.e. $O_i=P_i P_{i+1}$, where $P_i$ is the Pauli operator on the $i$-th qubit. We can use the classical shadow to estimate the expectation value of the observable $O_i$ by simply replacing the unknown quantum state $\rho$ with a classical shadow $\hat{\rho}$, which is a random variable that yields the correct prediction in expectation:


```{code-cell} ipython3
from mitiq import Observable, PauliString

# from cirq import LineQubit

r"""
 define the observables $\{X_iX_{i+1}\}_{i\leq n-1}$
"""
from mitiq import PauliString

list_of_paulistrings = (
    [
        PauliString("XX", support=(i, i + 1), coeff=1)
        for i in range(num_qubits - 1)
    ]
    + [PauliString("YY", support=(i, i + 1)) for i in range(num_qubits - 1)]
    + [
        PauliString("ZZ", support=(i, i + 1), coeff=1)
        for i in range(num_qubits - 1)
    ]
)

for observables in list_of_paulistrings:
    print(observables)
# print the type of the observables
```

```{code-cell} ipython3
r"""
Solve for the exact expectation values with mitiq
:math:`\langle O\rangle_{\rho} = \mathrm{Tr}(\rho O)`
"""
from functools import partial
from mitiq.interface import mitiq_cirq

expval_exact = []
for i, pauli_string in enumerate(list_of_paulistrings):
    obs = Observable(pauli_string)
    exp = obs.expectation(
        simple_test_circuit(params, qubits),
        execute=partial(mitiq_cirq.compute_density_matrix, noise_level=(0.0,)),
    )

    expval_exact.append(exp)
```

In the scenario of a random Pauli measurement, where a set of local observables acts on neighboring strings, denoted by $\{P_i P_{i+1}\}_{i\leq n-1}$, the expected value of the local observable $P_i P_{i + 1}$, where $P \in\{I,X,Y,Z\}$, can be expressed as follows:

\begin{equation}
\mathrm{Tr}(P_i P_{i + 1}\hat{\rho})=\prod_{i\in\mathrm{obs}}3\langle\hat{b}_i|U_i P_i U_i ^\dagger|\hat{b}_i\rangle ,\qquad|\hat{b}_i\rangle\in\{0,1\}, ~~0\leq i\leq n-1.
\end{equation}

Here,
\begin{equation}
\langle{b}_i|U_i P_i U_i ^\dagger|{b}_i\rangle
=\langle{b}_i|Z|{b}_i\rangle\cdot\delta(P_i ,U_i ^\dagger Z U_i )\qquad \mathrm{if}~~ P_i \in\{X,Y,Z\}    
\end{equation}
When we realize this code, it's important to consider that we record the equivalent Pauli measurement in the case of local Pauli measurement. The expectation value of the observable $O_i$ can be simply computed by counting the number of exact matches between the observable and the classical shadow, and then multiplying the result by the appropriate sign given the measurement result $b_i =\pm 1$. If the operator in the observable does not match the random Pauli measurement (recorded as output in the program) that has been performed on the particular qubit, i.e. if $u_i:= U_i ^\dagger Z U_i  \neq P_i $, the result vanishes.

Consequently, computing the mean estimator involves counting the number of exact matches between the observable and the classical shadow, and then multiplying the result by the appropriate sign. In the following, we present the function `expectation_estimation_shadow`, which allows for estimating any observable based on a classical shadow. This is realised by the main function `execute_with_shadows` when *state_reconstruction =* **False**.

### 4.3 Shadow Estimation Bound on Estimation of Expectation Values of Observables

The shadow estimation bound of operator expectation values is given by the following theorem:
_________________________________________________________________________
#### Theorem:
A sequence of observables $\{O_i\}_{i\leq M}$ acting on $n$ qubits 
\begin{equation}
K=2\log(2M \delta^{-1})\qquad N=34\epsilon^{-2}\max_i\left\|O_i-\frac{\mathrm{Tr}(O_i)\mathbb{I}}{2^n}\right\|_{\mathrm{shadow}}^2
\end{equation}
with error rates $\delta,\epsilon\leq 1$. 

Then, with probability at least $1-\delta$, a collection of $R= NK$ independent classical shadows $\{\hat{\rho}_k\}_{k\leq NK}$ allow for accurately predicting all features via median of means, i.e.
\begin{equation}
|\hat{o}_i(N,K)-\mathrm{Tr}(O_i\rho)|\leq \epsilon
\end{equation}
for all $1\leq i\leq M$.
_________________________________________________________________________


The general form of the shadow norm $\|\cdot\|_{\mathrm{shadow}}$ is not clear and depends on the ensemble $\mathcal{U}$ from which we sampled the unitaries, but there are special cases where the shadow norm computable. For example, if we sample from the local Clifford group $\mathcal{U}=\mathrm{CL}(2)^n$, the shadow norm is given by: 
\begin{equation}
\parallel O \parallel_{\mathrm{shadow}}\leq 4^{w}\parallel O \parallel^2,\qquad O\mathrm{~acting~on~}w\mathrm{~qubits}
\end{equation}
 The shadow norm, in this situation, correlates with the operator ($L_2$) norm. This guarantees the accurate prediction of many local observables from only a much smaller number of measurements. We realize the bound of the shadow estimation in the function `shadow_estimation_bound`, which is called in the main function `execute_with_shadows` when *state_reconstruction =* **False**.


```{code-cell} ipython3
r"""
Minimum number of snapshots N required for predicting the expectation values of the observables with error rate epsilon.
"""
# create a grid of errors epsilon = 0.2, 0.4, 0.6, 0.8 defined as epsilon in the Theorem
epsilon_grid = [1 - 0.2 * x for x in range(0, 5, 1)]
n_total_measurements = []
expectation_value_shadow = []
# define failure_rate delta in Theorem
failure_rate = 0.01
# For each error in epsilon_grid
for error in epsilon_grid:

    # get the number of total shadow measurements and groups need to split into
    # needed so that the absolute error < epsilon, and accuracy >= 1 - failure_rate.
    r, k = n_measurements_opts_expectation_bound(
        error, list_of_paulistrings, failure_rate
    )
    n_total_measurements.append(r)

    shadow_outputs = shadow_quantum_processing(test_circuits, cirq_executor, r)
    output = classical_post_processing(
        shadow_outcomes=shadow_outputs,
        observables=list_of_paulistrings,
        k_shadows=k,
    )

    # estimate all the observables in {O_i}_i with error rate epsilon and failure rate delta
    expectation_value_shadow.append(list(output.values()))

    # totle number of snpshots required for error rate = epsilon
    print(
        f"{r} totel number of snapshots required for error rate {int((error+1e-10)*10)/10}"
    )
```


```{code-cell} ipython3
import matplotlib.pyplot as plt

# plot bound
plt.plot(
    n_total_measurements,
    [e for e in epsilon_grid],
    linestyle="-.",
    color="gray",
    label=rf"$\epsilon$",
    marker=".",
)

# Plot exact expectation values
for i, obs in enumerate(expval_exact):
    if i < len(expval_exact) // 3:
        color = "red"
    elif len(expval_exact) // 3 <= i < 2 * len(expval_exact) // 3:
        color = "blue"
    else:
        color = "green"
    obs = expval_exact[i]
    for j, error in enumerate(epsilon_grid):
        plt.scatter(
            [n_total_measurements[j]],
            [np.abs(obs - expectation_value_shadow[j][i])],
            marker=".",
            color=color,
        )

plt.xlabel(r"$N$ (Shadow size) ")
plt.ylabel(r"$|\langle O \rangle_{\rho} - \hat{o}|$")

# legend dots
plt.scatter(
    [], [], marker=".", color="red", label=r"$\langle X_i X_{i+1} \rangle$"
)
plt.scatter(
    [], [], marker=".", color="blue", label=r"$\langle Y_i Y_{i+1} \rangle$"
)
plt.scatter(
    [], [], marker=".", color="green", label=r"$\langle Z_i Z_{i+1} \rangle$"
)
plt.legend()
# x log scale
plt.xscale("log")
plt.show()
```

**Acknowledgements**

This project contains code adapted from PennyLane's tutorial on Classical Shadows. We would like to acknowledge the original authors of the tutorial, PennyLane developers Brian Doolittle and Roeland Wiersema. The tutorial can be found at [this link](https://pennylane.ai/qml/demos/tutorial_classical_shadows).
