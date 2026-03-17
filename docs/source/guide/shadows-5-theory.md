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

# What is the Theory Behind Classical Shadow Estimation?

Classical shadow estimation {cite}`huang2020predicting` is a protocol for predicting many properties of an unknown quantum state $\rho$ from a small number of randomized measurements.
Rather than performing full state tomography, it constructs a compact classical representation — a *classical shadow* — from which expectation values of many observables can be estimated simultaneously.

The protocol works in two steps:

- **Step 1: Quantum processing.** Apply a random unitary $U$ sampled from a fixed ensemble $\mathcal{U}$, measure in the computational basis, and record the outcome $b$ and the unitary $U$.

- **Step 2: Classical post-processing.** Apply an inverse channel $\mathcal{M}^{-1}$ to each measurement record to obtain a *classical shadow* $\hat{\rho}$, then estimate observables as $\hat{o}_i = \mathrm{Tr}(O_i \hat{\rho})$.

The robust variant of this protocol, described in [Robust Shadow Estimation](#robust-shadow-estimation), additionally calibrates the inverse channel to account for noise on the rotation gates and measurements.

## The Classical Shadow Protocol

The quantities of interest are expectation values of a set of observables $\{O_i\}_{i=1}^M$:

$$o_i = \mathrm{Tr}(O_i \rho), \qquad 1 \leq i \leq M.$$

The protocol requires only $N$ measurements to predict all $M$ values simultaneously up to additive error $\epsilon$, provided that

$$N \geq \mathcal{O}\!\left(\epsilon^{-2} \log M \, \max_i \|O_i\|^2_{\mathrm{shadow}}\right).$$

In each measurement round, a random unitary $U \sim \mathcal{U} \subseteq U(2^n)$ is applied to $\rho$, and the system is measured in the computational basis to obtain an outcome $b \in \{0,1\}^n$ with probability $\mathrm{Pr}[b] = \langle b | U\rho U^\dagger | b\rangle$.
The pair $(U, b)$ is recorded as a *classical snapshot* $U^\dagger |b\rangle\langle b| U$.

These snapshots carry information about $\rho$ in expectation:

$$\mathbb{E}\left[U^\dagger |b\rangle\langle b| U\right] = \mathcal{M}(\rho),$$

where $\mathcal{M}$ is a quantum channel determined by the ensemble $\mathcal{U}$.
For any tomographically complete ensemble, $\mathcal{M}$ is invertible (as a linear map), so applying $\mathcal{M}^{-1}$ to each snapshot defines an unbiased estimator of $\rho$:

$$\hat{\rho} = \mathcal{M}^{-1}\!\left(U^\dagger |b\rangle\langle b| U\right).$$

This is the **classical shadow**.
Note that $\mathcal{M}^{-1}$ is linear but not completely positive, so it cannot be physically implemented — it is only applied to classical data in memory.

By **Schur's Lemma** {cite}`harrow2013church`, averaging over the full unitary group $U(d)$ yields a depolarizing channel $\mathcal{M} = \mathcal{D}_{(2^n+1)^{-1}}$, whose inverse is:

$$\mathcal{M}^{-1}(\cdot) = \left[(2^n + 1) - \mathbb{I}\cdot\mathrm{Tr}\right](\cdot).$$

Repeating the procedure $N$ times produces a collection of $N$ independent classical shadows:

$$S(\rho, N) = \left\{\hat{\rho}_1, \ldots, \hat{\rho}_N\right\}.$$

Since each shadow satisfies $\mathbb{E}[\hat{\rho}_k] = \rho$, any expectation value $\mathrm{Tr}(O_i\rho)$ can be estimated as $\hat{o}_i = \mathrm{Tr}(O_i\hat{\rho}_k)$.
In practice, the **median-of-means** estimator is used to achieve low failure probability from $R = NK$ total snapshots:

$$\hat{o}_i(N, K) := \mathrm{median}\left\{\hat{o}_i^{(1)}, \ldots, \hat{o}_i^{(K)}\right\}, \quad \hat{o}_i^{(j)} = \frac{1}{N}\sum_{k=N(j-1)+1}^{Nj} \mathrm{Tr}(O_i \hat{\rho}_k).$$

### Shadow norm and choice of ensemble

The shadow norm $\|O\|_{\mathrm{shadow}}$ determines how many measurements are needed to estimate $\mathrm{Tr}(O\rho)$ and depends on the ensemble $\mathcal{U}$.
Two practically important cases are:

- **Global Clifford measurements** ($\mathcal{U} = \mathcal{C}_n$): the shadow norm equals the Hilbert-Schmidt norm, $\|O\|_{\mathrm{shadow}} \leq 3\,\mathrm{Tr}[O^2]$.
Each snapshot takes the form $\hat{\rho} = (2^n+1)U^\dagger|b\rangle\langle b|U - \mathbb{I}$.

- **Random Pauli measurements** ($\mathcal{U} = \mathcal{C}_1^{\otimes n}$): $\|O\|_{\mathrm{shadow}} \leq 4^w \|O\|^2$ for an operator acting on $w$ qubits.
The unitary factorizes over qubits, so each snapshot also factorizes:

$$\hat{\rho} = \bigotimes_{i=1}^{n}\!\left(3U_i^\dagger|b_i\rangle\langle b_i|U_i - \mathbb{I}\right).$$

Random Pauli measurements are well-suited to estimating local observables efficiently.
Global Clifford measurements require circuit depth linear in system size, which is not currently feasible for large systems.
For this reason, Mitiq implements random Pauli measurements.
An intermediate approach is discussed in {cite}`hu2023classical`.

## Robust Shadow Estimation

The robust shadow estimation protocol {cite}`chen2021robust` extends the classical shadow framework to handle noise on the rotation gates and measurements.
The inherent randomization of the protocol converts arbitrary gate-independent, time-invariant, Markovian noise into an effective Pauli noise channel $\Lambda$, which can be characterized by a calibration experiment and then absorbed into the inverse channel.

```{figure} ../img/shadows_noisy_channel.png
---
width: 400px
name: shadows-noisy-channel
---
```

Decomposing the noisy unitary $\widetilde{U}$ and noisy measurement $\widetilde{M}_Z$ into noiseless and noise parts gives $\widetilde{U}\widetilde{M}_Z = U\Lambda\mathcal{M}_Z$.
The noisy shadow channel becomes:

$$\widehat{\mathcal{M}} = \mathbb{E}_{\mathcal{G}}\!\left[\mathcal{U}^\dagger \mathcal{M}_Z \Lambda \mathcal{U}\right] = \sum_\lambda \hat{f}_\lambda \Pi_\lambda, \qquad \hat{f}_\lambda := \frac{\mathrm{Tr}(\mathcal{M}_Z\Lambda\Pi_\lambda)}{\mathrm{Tr}(\Pi_\lambda)},$$

where $\Pi_\lambda$ are projectors onto the irreducible representation subspaces of $\mathcal{G}$.

### Pauli fidelities

For the local Clifford group $\mathcal{C}_1^{\otimes n}$, the projectors factorize as $\Pi_b = \bigotimes_{i=1}^n \Pi_{b_i}$, where:

$$\Pi_{b_i} = \begin{cases} |\sigma_0\rangle\!\rangle\langle\!\langle\sigma_0| & b_i = 0 \\ \mathbb{I} - |\sigma_0\rangle\!\rangle\langle\!\langle\sigma_0| & b_i = 1 \end{cases}$$

The expansion coefficients $\{\hat{f}_b\}_b$ are the **Pauli fidelities**.
The single-round estimator is:

$$\hat{f}^{(r)}_b = \prod_{i=1}^n \langle b_i | \mathcal{U}_i P_Z^{b_i} \mathcal{U}_i^\dagger | b_i\rangle, \qquad \mathcal{U}_i \in \mathcal{C}_1,\ b_i \in \{0, 1\}.$$

The final estimate uses the median-of-means estimator over $R = NK$ calibration rounds:

$$\begin{align}
\bar{f}^{(k)} &= \frac{1}{N}\sum_{r=(k-1)N+1}^{kN} \hat{f}^{(r)}, \\
\hat{f} &= \mathrm{median}\!\left\{\bar{f}^{(1)}, \ldots, \bar{f}^{(K)}\right\}.
\end{align}$$

In the noiseless case ($\Lambda \equiv \mathbb{I}$), the Pauli fidelities reduce to the known values $\hat{f}_b^{\mathrm{ideal}} = 3^{-|b|}$, where $|b|$ is the number of $|1\rangle$ components in $b$.
Deviations from these values quantify the noise.

### Calibrated inverse channel

Since it is generally infeasible to distinguish $\Lambda$ from the unknown state $\rho$, the noisy channel $\widehat{\mathcal{M}}$ is characterized using measurements on the known state $|0\rangle^{\otimes n}$.
Once the Pauli fidelities $\{\hat{f}_b\}$ are estimated, the calibrated inverse channel is:

$$\widehat{\mathcal{M}}^{-1} = \sum_{b \in \{0,1\}^n} \hat{f}_b^{-1}\, \Pi_b.$$

The robust shadow protocol then proceeds exactly as the standard protocol, with the ideal inverse channel replaced by $\widehat{\mathcal{M}}^{-1}$.
This yields an unbiased estimator even in the presence of noise, at the cost of additional calibration measurements.

