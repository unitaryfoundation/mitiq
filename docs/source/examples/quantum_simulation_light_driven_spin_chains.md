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

```{tags} qiskit, zne, advanced
```

# Use ZNE to simulate time evolution of light-driven spin chains

This tutorial demonstrates how to use Zero-Noise Extrapolation (ZNE) to mitigate noise in the quantum simulation of periodically driven (Floquet) spin chains. The model and parameters are based on the work: *Rodriguez-Vega et al. Phys. Rev. Research (2022)* {cite}`Rodriguez_Vega_2022_PRR` ([arXiv:2108.05975](https://arxiv.org/abs/2108.05975)).

## Physical Model and Background

Periodically driven quantum many-body systems (often called Floquet engineered systems) can realize exotic phases of matter that are otherwise inaccessible in static systems. Under a high-frequency periodic drive (such as a laser light), the effective time evolution of the system can be described by a time-independent effective Floquet Hamiltonian.

We consider a light-driven Hubbard model at half-filling defined on a one-dimensional chain:

$$
H(t) = -t_h \sum_{i, \sigma} \left( e^{i A \sin(\Omega t)} c_{i\sigma}^\dagger c_{i+1\sigma} + \text{H.c.} \right) + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

where $t_h$ is the hopping amplitude, $U$ is the on-site Coulomb interaction, $\Omega$ is the drive frequency, and $A$ is the dimensionless vector potential amplitude. In the limit $U \gg t_h$, second-order perturbation theory yields an effective time-dependent Heisenberg spin Hamiltonian:

$$
H_s(t) = J(t) \sum_{i} S_i \cdot S_{i+1}
$$

with a time-dependent exchange interaction:

$$
J(t) = \sum_{\alpha, \beta=-\infty}^{\infty} e^{i(\alpha-\beta)\Omega t} [ J_\alpha(A) J_{-\beta}(-A) + J_\alpha(-A) J_{-\beta}(A) ] \frac{2 t_h^2}{U - \beta \Omega}
$$

where $J_\alpha$ is the $\alpha$-th Bessel function of the first kind. Taking the time-average of $J(t)$ over one drive cycle, we obtain the effective Floquet exchange interaction:

$$
J_F = \sum_{\beta=-\infty}^{\infty} J_\beta^2(A) \frac{4 t_h^2}{U - \beta \Omega}
$$

which governs the effective static Floquet dynamics under the Hamiltonian:

$$
H_{\text{eff}} = J_F \sum_{i} S_i \cdot S_{i+1}
$$

For this tutorial, we set $t_h = 1$ as our energy unit, and use the drive parameters $U = 10.0$, $A = 2.8$, and $\Omega = 6.0$, mirroring the parameters of Fig. 4 in the main text of the reference paper.

Note that for real $A$, the term $J_\alpha(A) J_{-\beta}(-A) + J_\alpha(-A) J_{-\beta}(A)$ inside the sum for $J(t)$ simplifies to $2 J_\alpha(A) J_\beta(A)$ when $\alpha+\beta$ is even, and vanishes to zero when $\alpha+\beta$ is odd. This simplification holds because $J_{-\beta}(-A) = J_\beta(A)$ and $J_\alpha(-A) = (-1)^\alpha J_\alpha(A)$ for real arguments. We utilize this simplification in our code implementation below.


## Calculating the Exchange Coupling $J(t)$ and $J_F$

Below we implement the functions to compute the time-dependent coupling $J(t)$ and the effective Floquet coupling $J_F$. We truncate the infinite Bessel sums to a safe range $-M \le \alpha, \beta \le M$ with $M = 15$.

```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.linalg import expm

import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import QasmSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

from mitiq import zne
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise
```

```{code-cell} ipython3
def exchange_coupling(t: float, U: float = 10.0, A: float = 2.8, Omega: float = 6.0, th: float = 1.0, M: int = 15) -> float:
    """Calculate the time-dependent exchange coupling J(t).
    
    Note: The original sum contains terms J_alpha(A) J_{-beta}(-A) + J_alpha(-A) J_{-beta}(A).
    We simplify this to 2 * J_alpha(A) * J_beta(A) when alpha + beta is even (and 0 otherwise).
    This simplification holds because A is real, which implies:
      J_{-beta}(-A) = J_beta(A) and J_alpha(-A) = (-1)^alpha J_alpha(A).
    """
    val = 0.0
    for alpha in range(-M, M+1):
        for beta in range(-M, M+1):
            if (alpha + beta) % 2 == 0:
                val += np.cos((alpha - beta) * Omega * t) * 2.0 * jv(alpha, A) * jv(beta, A) * (2.0 * th**2) / (U - beta * Omega)
    return val

def floquet_exchange_coupling(U: float = 10.0, A: float = 2.8, Omega: float = 6.0, th: float = 1.0, M: int = 15) -> float:
    """Calculate the effective static Floquet exchange coupling J_F."""
    val = 0.0
    for beta in range(-M, M+1):
        val += (jv(beta, A) ** 2) * (4.0 * th**2) / (U - beta * Omega)
    return val
```

Let's print the value of the effective Floquet coupling $J_F$:

```{code-cell} ipython3
JF = floquet_exchange_coupling()
print(f"Effective Floquet exchange coupling J_F = {JF:.6f}")
```

## Spin Chain Quantum Simulation and Circuit Construction

We simulate a spin chain of length $L = 3$. We evolve the system from the initial antiferromagnetic (AFM) state $|\uparrow \downarrow \uparrow \rangle$ (which translates to the qubit state $|010\rangle$).

We decompose the Heisenberg exchange term $e^{-i \theta S_i \cdot S_j}$ on a single bond using CNOTs and single-qubit rotations. Since the spin operators are $S^\alpha = \frac{1}{2}\sigma^\alpha$, the exponent is $e^{-i \frac{\theta}{4}(X_i X_j + Y_i Y_j + Z_i Z_j)}$. The three terms commute, so they can be exponentiated exactly:

```{code-cell} ipython3
def add_heisenberg_bond(qc: QuantumCircuit, i: int, j: int, theta: float):
    """Add exact Heisenberg exchange interaction gates on bond (i, j)."""
    # exp(-i * (theta/4) * X_i X_j)
    qc.h(i)
    qc.h(j)
    qc.cx(i, j)
    qc.rz(theta / 2.0, j)
    qc.cx(i, j)
    qc.h(i)
    qc.h(j)

    # exp(-i * (theta/4) * Y_i Y_j)
    qc.rx(np.pi / 2, i)
    qc.rx(np.pi / 2, j)
    qc.cx(i, j)
    qc.rz(theta / 2.0, j)
    qc.cx(i, j)
    qc.rx(-np.pi / 2, i)
    qc.rx(-np.pi / 2, j)

    # exp(-i * (theta/4) * Z_i Z_j)
    qc.cx(i, j)
    qc.rz(theta / 2.0, j)
    qc.cx(i, j)
```

To evolve the system in time, we use a second-order symmetric Trotter step. For each time step $\Delta t$, the time evolution operator is approximated as:

$$
U_{\text{sym}}(\Delta t) \approx e^{-i H_{12}(t) \Delta t / 2} e^{-i H_{23}(t) \Delta t} e^{-i H_{12}(t) \Delta t / 2}
$$

We build the full evolution circuit below:

```{code-cell} ipython3
def trotter_evolution_circuit(L: int, U: float, A: float, Omega: float, dt: float, n_steps: int) -> QuantumCircuit:
    """Build the Qiskit quantum circuit for the light-driven spin chain evolution."""
    qc = QuantumCircuit(L)
    # Initialize AFM state |010> (q0=0, q1=1, q2=0) -> apply X on qubit 1
    for ii in range(1, L, 2):
        qc.x(ii)
        
    for step in range(n_steps):
        t = step * dt
        theta = exchange_coupling(t, U, A, Omega) * dt
        
        # Symmetric Trotter step: U_12(theta/2) -> U_23(theta) -> U_12(theta/2)
        add_heisenberg_bond(qc, 0, 1, theta / 2.0)
        add_heisenberg_bond(qc, 1, 2, theta)
        add_heisenberg_bond(qc, 0, 1, theta / 2.0)
        
    qc.measure_all()
    return qc
```

## Observable: Antiferromagnetic Order Parameter $\Delta(t)$

We measure the antiferromagnetic (AFM) order parameter (staggered magnetization) $\Delta(t)$ defined as:

$$
\Delta(t) = \frac{1}{L} \sum_{i=0}^{L-1} (-1)^i \langle \sigma_i^z(t) \rangle
$$

We write a helper function to compute this expectation value from the dictionary of measurement counts:

```{code-cell} ipython3
def compute_order_parameter(counts: dict) -> float:
    """Calculate the staggered magnetization order parameter from measurement counts."""
    shots = sum(counts.values())
    total_sz = 0.0
    for state, count in counts.items():
        val = 0.0
        # Qiskit measurement string ordering: state[0] corresponds to qubit L-1, state[-1] to qubit 0
        for i in range(len(state)):
            qubit_idx = len(state) - 1 - i
            bit = int(state[qubit_idx])
            z_i = 1.0 - 2.0 * bit # 0 -> +1, 1 -> -1
            val += ((-1) ** i) * z_i
        total_sz += val * count
    return total_sz / (len(state) * shots)
```

## Setup Simulators and Executor

We define the executor to run on noiseless and noisy simulators. The noisy backend uses a depolarizing noise model to simulate gate errors on NISQ hardware.

```{code-cell} ipython3
# Depolarizing noise model mimicking high-fidelity device noise levels (0.1% gate error rate)
noise_model = initialized_depolarizing_noise(noise_level=0.001)
backend_noisy = QasmSimulator(noise_model=noise_model)
backend_clean = QasmSimulator()

def execute_circuit(circuit: QuantumCircuit, noise: bool = True, shots: int = 2048) -> float:
    """Executor function to run circuit and return the AFM order parameter."""
    b = backend_noisy if noise else backend_clean
    pm = generate_preset_pass_manager(backend=b, optimization_level=0)
    exec_circuit = pm.run(circuit)
    
    sampler = Sampler(b)
    job = sampler.run([exec_circuit], shots=shots)
    counts = job.result()[0].data.meas.get_counts()
    
    return compute_order_parameter(counts)
```

## Calculating Exact Floquet Dynamics

To validate the simulation, we calculate the exact time evolution under the static effective Floquet Hamiltonian $H_{\text{eff}} = J_F (S_0 \cdot S_1 + S_1 \cdot S_2)$ via exact diagonalization. Note that this time-independent Floquet approximation holds in the high-frequency regime where $\Omega \gg J_F$. The exact time-dependent dynamics would require integrating $J(t)$; however, this static approximation matches the paper's Fig. 4 results and is sufficient for demonstrating ZNE.

```{code-cell} ipython3
# Pauli matrices
I_m = np.eye(2)
X_m = np.array([[0.0, 1.0], [1.0, 0.0]])
Y_m = np.array([[0.0, -1.0j], [1.0j, 0.0]])
Z_m = np.array([[1.0, 0.0], [0.0, -1.0]])

# Tensor products
X0_m = np.kron(np.kron(X_m, I_m), I_m)
X1_m = np.kron(np.kron(I_m, X_m), I_m)
X2_m = np.kron(np.kron(I_m, I_m), X_m)

Y0_m = np.kron(np.kron(Y_m, I_m), I_m)
Y1_m = np.kron(np.kron(I_m, Y_m), I_m)
Y2_m = np.kron(np.kron(I_m, I_m), Y_m)

Z0_m = np.kron(np.kron(Z_m, I_m), I_m)
Z1_m = np.kron(np.kron(I_m, Z_m), I_m)
Z2_m = np.kron(np.kron(I_m, I_m), Z_m)

S0_S1_m = 0.25 * (X0_m @ X1_m + Y0_m @ Y1_m + Z0_m @ Z1_m)
S1_S2_m = 0.25 * (X1_m @ X2_m + Y1_m @ Y2_m + Z1_m @ Z2_m)

H_eff = JF * (S0_S1_m + S1_S2_m)

# Initial state |010>
psi0 = np.kron(np.kron(np.array([1.0, 0.0]), np.array([0.0, 1.0])), np.array([1.0, 0.0]))

def exact_floquet_dynamics(t: float) -> float:
    psi_t = expm(-1.0j * H_eff * t) @ psi0
    z0 = np.real(psi_t.conj().T @ Z0_m @ psi_t)
    z1 = np.real(psi_t.conj().T @ Z1_m @ psi_t)
    z2 = np.real(psi_t.conj().T @ Z2_m @ psi_t)
    return (z0 - z1 + z2) / 3.0
```

## Running Simulations with ZNE Mitigation

We run the simulation over a series of time steps from $t = 0$ to $t = 2.0$. Note that increasing the number of Trotter steps (i.e., reducing the time step size $\Delta t$ for a fixed total time) improves the Trotter approximation accuracy, though the current choice of $dt = 0.1$ is sufficient for this demonstration.

For each step, we calculate:
1. The **exact Floquet dynamics** (numerical).
2. The **ideal (noiseless) Trotter simulation** results.
3. The **unmitigated noisy simulation** results.
4. The **ZNE mitigated** results (using a linear fit over noise scaling factors `[1., 1.5, 2.0]`). We use a linear fit over three noise scales, which is sufficient for this shallow circuit; for deeper circuits, a higher-order extrapolation (e.g., Richardson) might be more suitable.

```{code-cell} ipython3
U = 10.0
A = 2.8
Omega = 6.0
dt = 0.1
n_dt = 20

times = [step * dt for step in range(n_dt + 1)]
exact_vals = []
clean_vals = []
noisy_vals = []
mitigated_vals = []

for step in range(n_dt + 1):
    t = step * dt
    circuit = trotter_evolution_circuit(L=3, U=U, A=A, Omega=Omega, dt=dt, n_steps=step)
    
    # Exact Floquet
    exact_vals.append(exact_floquet_dynamics(t))
    
    # Noiseless Trotter
    clean_vals.append(execute_circuit(circuit, noise=False))
    
    # Noisy Trotter (Unmitigated)
    noisy_vals.append(execute_circuit(circuit, noise=True))
    
    # Mitigated Trotter (ZNE with Linear Extrapolation)
    linear_factory = zne.inference.LinearFactory(scale_factors=[1.0, 1.5, 2.0])
    mit_val = zne.execute_with_zne(
        circuit, 
        executor=lambda c: execute_circuit(c, noise=True), 
        factory=linear_factory
    )
    mitigated_vals.append(mit_val)
```

## Analysis and Plotting

Finally, we plot the staggered magnetization $\Delta(t)$ as a function of time:

```{code-cell} ipython3
plt.figure(figsize=(9, 6))
plt.plot(times, exact_vals, label="Exact Floquet Dynamics (Static)", color="black", linewidth=2.0)
plt.plot(times, clean_vals, label="Ideal Trotter Evolution", color="blue", marker="o", linestyle="None")
plt.plot(times, noisy_vals, label="Noisy Trotter (Unmitigated)", color="red", marker="x", linestyle="None")
plt.plot(times, mitigated_vals, label="Mitigated Trotter (ZNE)", color="green", marker="d", linestyle="None")

plt.xlabel(r"$t$")
plt.ylabel(r"AFM Order Parameter $\Delta(t)$")
plt.title("Simulating Time Evolution of Light-Driven Spin Chains")
plt.legend()
plt.grid(True)
plt.show()
```

As seen in the plot:
- The **Ideal Trotter Evolution** (blue circles) matches the **Exact Floquet Dynamics** (black line) very well, validating the symmetric Trotterization.
- The **Unmitigated Noisy Simulation** (red crosses) quickly degrades and decays towards zero due to the accumulating gate errors.
- The **ZNE Mitigated Simulation** (green diamonds) successfully corrects for the gate noise, staying close to the exact Floquet time evolution curve throughout the simulation.

## References

```{bibliography}
:filter: docname in docnames
```

