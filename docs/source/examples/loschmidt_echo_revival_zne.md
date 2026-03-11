---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{tags} zne, qiskit, intermediate
```

# ZNE with Qiskit: Simulation of Loschmidt Echo Revival

This tutorial replicates some of the results from {cite}`Javanmard_2022_arxiv`. We build a circuit that simulates the time-evolution of a transverse-field Ising model, then run ideal, noisy, and noise-mitigated simulations of the circuit.

In particular, let $\Lambda(t)$ be the _Loschmidt echo_, the probability that the system returns to its initial state at time $t$. $\Lambda(t)$ has a quasi-periodic series of peaks that are flattened as the noise level increases. Here, we demonstrate how to simulate the Loschmidt echo and use zero-noise extrapolation to mitigate the effects of noise.

The paper considers some additional effects of noise, which are outside the scope of this tutorial:
   
* Let $\lambda(t) = \lim_{N\to\infty} -\log(\Lambda(t))/N$, where $N$ is the number of sites in the Ising model. Dynamical quantum phase transitions (DQPTs) occur at values of $t$ where $\lambda(t)$ is not analytic. DQPTs are observed at different times in the ideal and noisy simulations, and occur more frequently in the noisy system.

* Noise weakens the correlations between adjacent sites.

+++

## Model definition

The Ising model that we will simulate has the Hamiltonian

$$H = H_{zz} + H_{xx} + H_{x}$$

where $H_{zz}$ and $H_{xx}$ are the interactions between neighboring sites and $H_x$ is the interaction with the external magnetic field. Specifically, for $N$ sites,

$$H_{zz} = -\frac{1}{2} \left[ \sum_{i=0}^{N-2}J_z Z_i Z_{i+1} \right], \hspace{0.4cm} H_{xx} = -\frac{1}{2} \left[ \sum_{i=0}^{N-2}J_x X_{i} X_{i+1} \right], \hspace{0.4cm} H_x = -\frac{1}{2} \left[ \sum_{i=0}^{N-1} h_x X_i \right]$$

where $X_i$ and $Z_i$ are the Pauli operators acting on site $i$, $J_z$ and $J_x$ are the $z$- and $x$-components of the spin-spin coupling, and $h_x$ is the strength of the external field. Here we will set $J_z > 0$ and $J_x > 0$, so that the spins at adjacent sites are correlated, and set $h_x > 0$ so that each spin prefers to have a positive $x$-component. (Strictly speaking, since $J_x \neq 0$ this is a Heisenberg model rather than an Ising model.) 

Assuming the system is in state $\ket{\psi_0}$ at $t = 0$, we want to compute the Loschmidt echo,

$$\Lambda(t) = \left|\bra{\psi_0}U(t)\ket{\psi_0}\right|^2,$$

where $U(t) = \exp(-iHt)$ is the time-evolution operator.

+++

## Reformulation as a quantum circuit

To simulate how the model behaves over $0 \leq t \leq t_{\textrm{max}}$, we divide the interval into $M$ steps. Letting $\delta t = t_{\textrm{max}}/M$, we have

$$U(k\delta t) = [\exp(-iH\delta t)]^k \hspace{0.25cm} (k = 0, \ldots, M)$$

Next we decompose $\exp(-iH\delta t)$. Up to an $\mathcal{O}(\delta t^2)$ error (since the terms in $H$ do not commute),

$$\exp(-i \left[H_{zz} + H_{xx} + H_{x}\right] \delta t) \approx \exp(-i H_{zz} \delta t)\exp(-i H_{xx} \delta t)\exp(-i H_{x} \delta t)$$

Now we observe that each term in the decomposition corresponds to a series of gates in an $N$-qubit circuit. For example,

$$\exp(-i H_{zz} \delta t) = \prod_{i=0}^{N-2} \exp\left( -i\frac{J_z\delta t }{2} Z_i Z_{i+1} \right)$$

Using the fact that $Z_i Z_{i+1} = I \otimes \cdots Z \otimes Z \cdots \otimes I$, we can rewrite this as a product of [$R_{ZZ}$ gates](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RZZGate),

$$\prod_{i=0}^{N-2} R_{ZZ}^{(i, i+1)}(J_z \delta t )$$

Similarly, the terms $\exp(-i H_{xx} \delta t)$ and $\exp(-i H_{x} \delta t)$ can be rewritten in terms of [$R_{XX}$](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RXXGate) and [$R_X$](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RXGate) gates, yielding

$$\exp(-iH\delta t) \approx \prod_{i=0}^{N-2} R_{ZZ}^{(i, i+1)}(J_z \delta t) \prod_{i=0}^{N-2} R_{XX}^{(i, i+1)}(J_x \delta t) \prod_{i=0}^{N-1} R_{X}^{i}(h_x \delta t)$$

To compute $\Lambda(k\delta t)$, we repeat this sequence of gates $k$ times. The circuit is implemented by the function in the following cell. Note that:
* We use $\ket{\psi_0} = H^{\otimes N}\ket{0^{\otimes N}}$, i.e. the spin at every site starts out parallel to the external magnetic field.
* The $R_{ZZ}$ gates commute with each other, so we can organize them into two "layers", one layer for the pairs of adjacent qubits $(i, i+1)$ with $i$ even and another for the pairs with $i$ odd. The $R_{XX}$ gates are organized in a similar way.

```{code-cell} ipython3
from qiskit import QuantumCircuit

def create_circuit(delta_t : float,
                   k : int,
                   n_qubits: int = 6,
                   measure : bool = True,
                   J_z : float = 1.0,
                   J_x : float = 0.1,
                   h_x : float = 0.1) -> QuantumCircuit:
    theta = J_z * delta_t
    phi = J_x * delta_t
    gamma = h_x * delta_t

    circuit = QuantumCircuit(n_qubits)
    
    circuit.h(circuit.qubits)
    
    for _ in range(k):
        for ii in range(0, n_qubits-1, 2):
            circuit.rzz(theta, ii, ii+1)
        for ii in range(1, n_qubits-1, 2):
            circuit.rzz(theta, ii, ii+1)
            
        for ii in range(0, n_qubits-1, 2):
            circuit.rxx(phi, ii, ii+1)
        for ii in range(1, n_qubits-1, 2):
            circuit.rxx(phi, ii, ii+1)
            
        circuit.rx(gamma, circuit.qubits)

    circuit.h(circuit.qubits)
    
    if measure:
        circuit.measure_all()

    return circuit
```

```{code-cell} ipython3
print(create_circuit(0.01, 1))
```

## Ideal simulation

To get a sense of the ideal behavior of the circuit, we will run a noiseless state-vector simulation. First we define a dataclass to hold all the simulation parameters, and functions to visualize the results.

```{code-cell} ipython3
import numpy as np
import dataclasses

@dataclasses.dataclass
class SimulationParams:
    '''Simulation parameters. Note that by default we use
    much coarser time steps than in the paper, so that
    the noisy simulations later in this demo run in
    a reasonably short time.'''
    t_max : float = 8.5
    M : int = 25
    n_qubits : int = 6
    dt : float = dataclasses.field(init=False)
    t_vals : np.ndarray[float] = dataclasses.field(init=False)

    # Only used in noisy simulations
    n_shots : int = 2048

    def __post_init__(self):
        self.dt = self.t_max / self.M
        self.t_vals = np.linspace(0., self.t_max, self.M + 1, endpoint=True)
```

```{code-cell} ipython3
from matplotlib import pyplot as plt

def setup_plot(title : str = None):
    plt.figure(figsize=(6.0, 4.0))
    plt.xlabel("$t$")
    plt.ylabel("$\\Lambda(t)$")
    if title is not None:
        plt.title(title)

def add_to_plot(x : np.ndarray[float],
                y : np.ndarray[float],
                label : str,
                legend : list[str]):
    if label == "ideal":
        plt.plot(x, y, color="black")
    elif label == "mitigated":
        plt.plot(x, y, marker='s', markersize=5)
    else:
        plt.plot(x, y, marker='.', markersize=10)
    legend.append(label)
```

To get the ideal result for $\Lambda(k\delta t)$, we omit the measurements from the circuit, and read the probability amplitude $\bra{0^{\otimes N}} H^{\otimes N} U(k \delta t) H^{\otimes N} \ket{0^{\otimes N}}$ directly from the final state vector.

```{code-cell} ipython3
from qiskit_aer import QasmSimulator

def simulate_ideal(circuit: QuantumCircuit) -> float:
    simulator = QasmSimulator(method="statevector", noise_model=None)

    circuit.save_statevector()
    job = simulator.run(circuit, shots=1)
    
    psi = job.result().data()["statevector"]
    
    # Get the probability of returning to |00...0>
    amp_0 = np.asarray(psi)[0]
    return np.abs(amp_0.real**2 + amp_0.imag**2)

def run_ideal(params : SimulationParams = SimulationParams()) -> tuple[np.ndarray[float]]:
    echo = np.array([
        simulate_ideal(
            create_circuit(params.dt, k, n_qubits=params.n_qubits, measure=False)
        )
        for k in range(0, params.M + 1)
    ])    

    return params.t_vals, echo
```

```{code-cell} ipython3
# Run the state-vector simulation
result_ideal = run_ideal(SimulationParams(t_max=25.0, M=200))
```

```{code-cell} ipython3
setup_plot("State-vector simulation")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
plt.legend(legend)
plt.show()
```

The result shows a series of Loschmidt echo peaks. Intuitively, for a noisy system we expect that as $t$ increases $\Lambda(t)$ will approach $1/2^N$, where $N$ is the number of qubits. This means that as the noise level increases, the peaks will be suppressed, starting at larger values of $t$. At higher levels of noise, we may not be able to detect the Loschmidt echo at all.

In the rest of this notebook, we will run simulations with two different noise models, and try to reconstruct the peak at $t \approx 6.5$ with zero-noise extrapolation. The next cell re-plots the ideal simulation result over the values of $t$ that we will consider.

```{code-cell} ipython3
result_ideal = run_ideal(SimulationParams(M=100))
setup_plot("State-vector simulation")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
plt.legend(legend)
plt.show()
```

## Simulation with depolarizing noise

The next few cells run a simulation with depolarizing noise. We transpile the circuit using 1-qubit rotations and CNOT as the basis gate set, and optionally use gate folding to scale the noise.

```{code-cell} ipython3
from qiskit.compiler import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import depolarizing_error
from mitiq.zne.scaling import fold_gates_at_random

def simulate_noisy(circuit: QuantumCircuit,
                   noise_model: NoiseModel,
                   n_shots : int,
                   scale_factor: float = None) -> float:
    # Transpile the circuit
    backend = QasmSimulator(noise_model=noise_model)
    exec_circuit = transpile(
        circuit,
        backend=backend,
        basis_gates=["u1", "u2", "u3", "cx"],
        optimization_level=0
    )

    # Apply gate folding
    folded_circuit = exec_circuit if scale_factor is None \
                     else fold_gates_at_random(exec_circuit, scale_factor)
    
    job = backend.run(folded_circuit, shots=n_shots)

    # Get the probability of returning to |00...0>
    counts = job.result().get_counts()
    ket = "0" * circuit.num_qubits
    if ket in counts:
        return counts[ket]/n_shots
    return 0.0

def run_depolarizing_noise(params : SimulationParams = SimulationParams(),
                           noise_level : float = 0.001,
                           scale_factor : float = None) -> tuple[np.ndarray[float]]:
    basis_gates = ["u1", "u2", "u3", "cx"]
    noise_model = NoiseModel(basis_gates)
    # Add depolarizing noise to the 1-qubit gates in the basis set
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise_level, 1), basis_gates[:3]
    )
    
    echo = np.array([
        simulate_noisy(
            create_circuit(params.dt, k, n_qubits=params.n_qubits, measure=True),
            noise_model,
            params.n_shots,
            scale_factor
        )
        for k in range(0, params.M + 1)
    ])    

    return params.t_vals, echo
```

```{code-cell} ipython3
low_noise = 0.0005
result_depolarizing = run_depolarizing_noise(noise_level=low_noise)
```

```{code-cell} ipython3
setup_plot("Ideal and noisy simulations")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
add_to_plot(
    *result_depolarizing,
    "depolarizing noise, $p = {}$".format(low_noise),
    legend)
plt.legend(legend)
plt.show()
```

As expected, the Loschmidt echo revival is weaker than in the ideal cases, and we get a lower peak. Applying gate folding suppresses the peak further; in the next cell we do this for scale factors $\alpha = 1, 2, 3$.

```{code-cell} ipython3
scale_factors = [1, 2, 3]
result_depolarizing_scaled = [
    run_depolarizing_noise(
        noise_level=low_noise,
        scale_factor=alpha)
    for alpha in scale_factors
]
```

```{code-cell} ipython3
setup_plot("Scaled noisy simulations")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_depolarizing_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
plt.legend(legend)
plt.show()
```

## Error mitigation with zero-noise extrapolation
At this level of noise, we can use ZNE to mostly recover the ideal result. Running the noisy simulation is expensive (especially as the scale factor $\alpha$ increases), so rather than using the high-level functions for ZNE, we apply the static method `RichardsonFactory.extrapolate` to the results from the previous cell.

```{code-cell} ipython3
from mitiq.zne.inference import RichardsonFactory

result_zne = RichardsonFactory.extrapolate(
    scale_factors,
    [r[1] for r in result_depolarizing_scaled]
)
```

```{code-cell} ipython3
setup_plot("ZNE for depolarizing noise model, $p = {}$".format(low_noise))
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_depolarizing_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
add_to_plot(result_depolarizing_scaled[0][0], result_zne, "mitigated", legend)
plt.legend(legend)
plt.show()
```

Increasing the baseline noise level makes it much harder to reconstruct the peak with ZNE.

```{code-cell} ipython3
high_noise = 0.005
scale_factors = [1, 2, 3]
result_depolarizing_scaled = [
    run_depolarizing_noise(
        noise_level=high_noise,
        scale_factor=alpha)
    for alpha in scale_factors
]
result_zne = RichardsonFactory.extrapolate(
    scale_factors,
    [r[1] for r in result_depolarizing_scaled]
)
```

```{code-cell} ipython3
setup_plot("ZNE for depolarizing noise model, $p = {}$".format(high_noise))
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_depolarizing_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
add_to_plot(result_depolarizing_scaled[0][0], result_zne, "mitigated", legend)
plt.legend(legend)
plt.show()
```

## Simulation with realistic device noise

Next, we run simulations using the noise model of the IBM Nairobi device. Again, the relatively high noise level makes it difficult to recover something close to the ideal signal.

```{code-cell} ipython3
from qiskit_ibm_runtime.fake_provider.backends import FakeNairobiV2

def run_ibm_nairobi_noise(params : SimulationParams = SimulationParams(),
                          scale_factor : float = None) -> tuple[np.ndarray[float]]:
    noise_model = NoiseModel.from_backend(FakeNairobiV2())
    
    echo = np.array([
        simulate_noisy(
            create_circuit(params.dt, k, n_qubits=params.n_qubits, measure=True),
            noise_model,
            params.n_shots,
            scale_factor
        )
        for k in range(0, params.M + 1)
    ])    

    return params.t_vals, echo
```

```{code-cell} ipython3
scale_factors = [1, 2, 3]
result_ibm_nairobi_scaled = [
    run_ibm_nairobi_noise(scale_factor=alpha) for alpha in scale_factors
]
```

```{code-cell} ipython3
result_zne = RichardsonFactory.extrapolate(
    scale_factors,
    [r[1] for r in result_ibm_nairobi_scaled]
)
```

```{code-cell} ipython3
setup_plot("ZNE for IBM Nairobi noise model")
legend = []
add_to_plot(*result_ideal, "ideal", legend)
for alpha, result in zip(scale_factors, result_ibm_nairobi_scaled):
    add_to_plot(*result, "$\\alpha = {}$".format(alpha), legend)
add_to_plot(result_depolarizing_scaled[0][0], result_zne, "mitigated", legend)
plt.legend(legend)
plt.show()
```

## Summary

For the system considered in this tutorial, the effectiveness of ZNE depends strongly on the level of noise. Under low-noise conditions, we can accurately recover the ideal Loschmidt echo signal. At higher noise levels, increasing the scale factor $\alpha$ almost completely flattens the first revival peak. As a result, the extrapolation procedure can qualitatively recover some of the signal, but is not quantitatively reliable.
