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

```{code-cell} ipython3
import qiskit
from qiskit import QuantumCircuit

from mitiq import zne
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise

import time
import numpy as np

import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import QasmSimulator
```

# Use ZNE to simulate quantum many body scars with Qiskit on IBMQ backends

+++

This tutorial shows how to error mitigate a quantum simulation using ZNE, and is applied to a case where the dynamics show signs of quantum many body scars (QMBS). The example in this tutorial is based on the model studied in *Chen et al. PRR (2022)* {cite}`Chen_2022_PRR`. A QMBS is a physical phenomenon where a systems dynamics gives rise to non-ergodic behaviour depending on the initial state. This is in contrast to the ergodic behaviour which is expected from the eigenstate thermalization hypothesis which predicts that a system would thermalize, i.e. a relaxation of the systems dynamics towards a thermal ensemble. For a more elaborate discussion, the interested reader is referred to *Serbyn et al. NatPhys (2021)* {cite}`Serbyn_2021_NatPhys`.

The Hamiltonian that is studied is the mixed-field Ising model (MFIM). Recently, in an analog quantum simulation of the MFIM the quantum many body scar phenomenon was observed experimentally using Rydberg atoms in optical tweezers (see *Bernien et al. Nat (2017)* {cite}`Bernien_2017_Nat`). 

The MFIM Hamiltonian can be written in terms of Pauli matrices as follows
\begin{equation}
H = H_{ZZ} + H_Z + H_X,
\end{equation}
\begin{equation}
H = V\sum_{i=1}^{L-1}Z_iZ_{i+1} - 2V \sum_{i=2}^{L-1}Z_i - V(Z_1 + Z_L) + \Omega\sum_{i=1}^L X_i.
\end{equation}
This is an Ising model with an Ising interaction strength $V$, with longitudinal field with strength proportional to $V$ and transverse field with strength proportional to $\Omega$. Note that this Ising chain is defined with open boundary conditions, i.e. the strength of the field at the boundaries ($i = 1$ and $i = L$) is a factor 2 smaller than at the other sites of the chain. More information on the model can be found in the article.

+++

The dynamics of this model is governed by the Schrödinger equation
\begin{equation}
\frac{d}{dt}\vert\Psi(t)\rangle = -i H\vert\Psi(t)\rangle,
\end{equation}
which can formally be solved as 
\begin{equation}
\vert\Psi(t + \Delta t)\rangle = e^{-i H\Delta t}\vert\Psi(t)\rangle = U(\Delta t)\vert\Psi(t)\rangle.
\end{equation}
To simulate the dynamics using a gate sequence one performs a Trotter decomposition of this unitary operator, that is
\begin{equation}
U(\Delta t) \approx e^{-iH_{ZZ}\Delta t}e^{-iH_{Z}\Delta t}e^{-iH_{X}\Delta t},
\end{equation}
which is a product of different unitary operators. Finally, one can express each of these unitary operators as a gate sequence of single-qubit gates or two-qubit gates that are subsequently applied.

+++

The resulting gate sequence is defined in the following function

```{code-cell} ipython3
def trotter_evolution_H(L: int, V: float,
                        Omega: float, dt: float) -> qiskit.QuantumCircuit:
    '''Return the circuit that performs a time step.
    
    Args:
        L: Length of the Ising chain
        V: Ising interaction strength
        Omega: Transverse field strength
        dt: Time step of unitary evolution
    '''
    
    cq = qiskit.QuantumCircuit(L)        
    
    # Apply Rx gates:
    for ii in range(L):
        cq.rx(2*Omega*dt, ii)
        
        
    # Apply Rz gates:
    cq.rz(-2*V*dt, 0)
    for ii in range(1, L-1):
        cq.rz(-4*V*dt, ii)
    cq.rz(-2*V*dt, L-1)
    
    
    # Mitiq ZNE raises an error for the usage of rzz.
    # We will give an explicit implementation of the 
    # 2-CNOT implementation of the Rzz gate:
    for ii in range(1, L-1, 2):
        cq.cx(ii, ii+1)
        cq.rz(2*V*dt, ii + 1)
        cq.cx(ii, ii+1)

    for ii in range(0, L-1, 2):
        cq.cx(ii, ii+1)
        cq.rz(2*V*dt, ii + 1)
        cq.cx(ii, ii+1)
        

    return cq
```

By subsequently applying this unitary operator (or its circuit equivalent), one (approximately) time evolves according to the system Hamiltonian.

In this example we will limit ourselves to calculating the behaviour of the staggered magnetization in the $z$-direction. This is defined as 
\begin{equation}
Z_\pi = \sum_{i=1}^{L}(-1)^i Z_i.
\end{equation}
The following function calculates the staggered z-magnetization for a given set of raw measurement counts

```{code-cell} ipython3
def staggered_mz(L: int, counts: qiskit.result.counts.Counts) -> list:
    '''Calculate the staggered z-magnetization for each count
    
    Args:
    L: Length of the Ising chain
    counts: raw measurement counts
    '''
    
    sz = 0
    ncounts = 0
    sz_list = []
    
    # We use (L-1)-ii since the state strings start with
    # the last qubit and end with the first qubit.
    stag = (-1)**(1 + np.array([(L-1)-ii for ii in range(L)]))
    
    for state, state_count in counts.items():
        
        bit_array = np.array([-2*int(i)+1 for i in state])
        ncounts += state_count
        sz_val = np.sum(stag*bit_array)/L
        
        for _ in range(state_count):

            sz_list.append(sz_val)
            
        
    return sz_list
```

For this tutorial we will not use real (quantum) hardware. If you do wish to do so, you can change the backends below to your desired backend.

+++

Note: Using an IBM quantum computer requires a valid IBMQ account. See https://quantum-computing.ibm.com/ for instructions to create an account, save credentials, and see online quantum computers.

```{code-cell} ipython3
USE_REAL_HARDWARE = False
```

```{code-cell} ipython3
if USE_REAL_HARDWARE:
    
    service = QiskitRuntimeService(channel="ibm_quantum") #, token="MY_IBM_QUANTUM_TOKEN")   # Get the API token in https://quantum-computing.ibm.com/account
    backend = service.least_busy(operational=True, simulator=False)
    noise_model = False
else:
    

    # Simulate the circuit with noise
    noise_model = initialized_depolarizing_noise(noise_level=0.02)
    backend = QasmSimulator(noise_model=noise_model)     # Default to a simulator.
    backend_noiseless = QasmSimulator()
```

We set up an executor that simulates the desired circuit a certain amount of shots and returns the measurement statistics of our desired expectation value in the following function

```{code-cell} ipython3
def ibmq_executor_full(circuit: qiskit.QuantumCircuit, 
                       shots: int = 8192) -> list:
    """Returns the expectation value of each shot

    Args:
        circuit: Circuit to run (can also be a list of circuits)
        shots: Number of times to execute the circuit to compute 
        the expectation value.
    """
    # Transpile the circuit so it can be properly run
    pm = generate_preset_pass_manager(
        backend=backend, 
        optimization_level=0, 
        basis_gates=noise_model.basis_gates if noise_model else None,
    )

    exec_circuit = pm.run(circuit)

    if not isinstance(exec_circuit, list):
        exec_circuit = [exec_circuit]

    # Run the circuit
    sampler = Sampler(backend)
    job = sampler.run(exec_circuit, shots=shots)

    # Convert from raw measurement counts to the expectation value
    if type(circuit) == list:
        # In case the input is a list of circuits
        all_counts = [job.result()[i].data.meas.get_counts() for i in range(len(folded_circuits))]
        sz_list = [staggered_mz(L, counts) for counts in all_counts]
        
    else:
        # In case the input is a single circuit
        counts = job.result()[0].data.meas.get_counts()
        sz_list = staggered_mz(L, counts)

    
    return sz_list


def ibmq_executor(circuit: qiskit.QuantumCircuit, shots: int = 8192) -> float:
    """Returns the expectation value to be mitigated (averaged over the shots).

    Args:
        circuit: Circuit to run (can also be a list of circuits)
        shots: Number of times to execute the circuit to compute the expectation value.
    """
    sz_list = ibmq_executor_full(circuit, shots)

    if type(sz_list[0]) == list:
        expectation_value = [np.mean(kk) for kk in sz_list]
        
    else:
        expectation_value = np.mean(sz_list)
        
    
    return expectation_value
```

We now have all necessary components to perform a time evolution of the Ising Hamiltonian. The Hamiltonian parameters used in the following are those used in the article to create Fig. 3 and 4 of Ref. [1]. However, in the following we will stick to a smaller Ising chain with 6 sites for the sake of simplicity. 

As an initial test we will time evolve the system over one time step $dt$ (you can change this by enlarging the parameter $n_{dt}$). The system is initialized in the Néel state, that is the $\vert 010101\rangle$ state. We simulate results with noise, and calculate the unmitigated and mitigated result. Additionally, we also calculate the result in case no noise is present in the system, that is, an ideal Trotted decomposed time evolution.

```{code-cell} ipython3
# System parameters
L = 6
V = 1
Omega = 0.24
dt = 1
n_dt = 11 #number of time steps
```

```{code-cell} ipython3
# Initialise quantum circuit
test_circuit = qiskit.QuantumCircuit(L) 

# Initialise time step quantum circuit
cc = trotter_evolution_H(L, V, Omega, dt)

# Initialise the Néel state |010101...>
for ii in range(1,L,2):
    test_circuit.x(ii)

# Time evolve n_dt time steps
for _ in range(n_dt):
    test_circuit = test_circuit.compose(cc)

# Measure in computational basis
test_circuit.measure_all() 
```

```{code-cell} ipython3
NO_NOISE = False
```

```{code-cell} ipython3
t1 = time.time()
unmitigated = ibmq_executor(test_circuit)
t2 = time.time()
print(f"Unmitigated result {unmitigated:.3f}, after {t2-t1:.4f}s.")
```

```{code-cell} ipython3
t1 = time.time()
mitigated = zne.execute_with_zne(test_circuit, ibmq_executor)
t2 = time.time()
print(f"Mitigated result {mitigated:.3f}, after {t2-t1:.4f}s.")
```

```{code-cell} ipython3
NO_NOISE = True
t1 = time.time()
unmitigated = ibmq_executor(test_circuit)
t2 = time.time()
print(f"No noise result {unmitigated:.3f}, after {t2-t1:.4f}s.")
```

The above zero noise extrapolation application uses the default scaling factors [1., 2., 3.] and applies a Richardson extrapolation.

+++

# Low level checks of scale factors and fitting factory (i.e. extrapolation scheme/ fit)

+++

In this section we will study in more detail the scaling factors used in our ZNE mitigation scheme. We will have a look at the expectation values as a function of various scaling factors to get some clues to decide which noise extrapolation scheme we should use. The obtained results are noisy since there are different sources of error, eg. shot noise and gate noise. When using an extrapolation scheme for the ZNE another source of error is extrapolation uncertainty.
To take this noise into account, and the corresponding uncertainty, we also calculate the standard deviation of our sample and the standard error (the uncertainty on the estimated expectation value). We continue with the earlier defined test circuit.

+++

We start with an arbitrary choice of scaling parameters below, and we chose a large number of them to get a grasp of the scaling behavior. Furthermore, will use the random gate folding function to incorporate the scaling factors into the folded circuit. 
Note that the random gate folding function applies the unitary folding map $G\rightarrow GG^\dagger G$ to a random subset of gates of the input circuit . In our specific case it is good to note that since the RZZ is explicitely given in terms of the CNOT and RZ gate, these individual gates can/ will also be folded.

The folded circuit is more sensitive to gate errors since it has a number of gates approximately equal to scale_factor * n, where n is the number of gates in the input circuit.

```{code-cell} ipython3
scale_factors = [1., 1.5, 2., 2.5, 3., 3.5, 4, 4.5, 5]
folded_circuits = [
        zne.scaling.fold_gates_at_random(test_circuit, scale)
        for scale in scale_factors
]

# Check that the circuit depth is (approximately) scaled as expected
for j, c in enumerate(folded_circuits):
    print(f"Number of gates of folded circuit {j} scaled by: {len(c) / len(test_circuit):.3f}")
```

We can now calculate the expectation value, standard deviation and standard error for each of these folded circuits and show a plot that will allow us to investigate the behaviour of this circuit for the different scaling factors.

```{code-cell} ipython3
NO_NOISE = False

t1 = time.time()
sz_list = ibmq_executor_full(folded_circuits)
t2 = time.time()

exp_vals = [np.mean(kk) for kk in sz_list]
err = [np.std(kk)/np.sqrt(len(kk)) for kk in sz_list]


plt.figure()
plt.errorbar(scale_factors, exp_vals, yerr=err, 
             linestyle="", marker=".", c="k", ecolor="red")
plt.xlabel(r"scaling factor $\lambda$")
plt.ylabel(r"$<Z_\pi>/L$")
plt.xlim([0, np.max(scale_factors)+0.1])
plt.show()
```

First, we note that the overview of the behavior of the expectation value under influence of the various scaling factors need not be similar for the various circuits used in our quantum simulation. That is, the various circuits used to reach a certain time in the time evolution/ simulation of the system of interest. It can be advisable to check the behavior in more detail for the various circuits one uses to obtain the optimal extrapolation routines. For the sake of simplicity, in the following, we will limit ourselves to one kind of extrapolation scheme for a fixed set of scaling factors.

The above figure suggests that if one decides to perform the extrapolation with fewer scaling factors, e.g. only three, one has to make a proper choice of scaling factors depending on the extrapolation function one chooses. In this particular case, an exponential fit would seem reasonable over this specific range of scaling factors. If one choses for a set of scaling parameters that are all smaller than 2, a linear fit might also yield reasonable results.

+++

Below we choose a set of scaling factors and perform several extrapolation schemes to make an initial choice of extrapolation scheme in the further simulations. First, we choose the scaling factors [1, 1.25, 1.5] and create the corresponding folded gates using random gate folding.

```{code-cell} ipython3
scale_factors = [1., 1.25, 1.5]
folded_circuits = [
        zne.scaling.fold_gates_at_random(test_circuit, scale)
        for scale in scale_factors
]

# Check that the circuit depth is (approximately) scaled as expected
for j, c in enumerate(folded_circuits):
    print(f"Number of gates of folded circuit {j} scaled by: {len(c) / len(test_circuit):.3f}")
```

We can use calculate the expectation values predicted from these folded gates for each scaling factor and use various fitting factories to determine the ZNE expectation value.

```{code-cell} ipython3
NO_NOISE = False

t1 = time.time()
sz_vals = ibmq_executor_full(folded_circuits)

exp_vals = [np.mean(kk) for kk in sz_vals]
std_vals = [np.std(kk) for kk in sz_vals]
```

```{code-cell} ipython3
zero_noise_value = zne.ExpFactory.extrapolate(scale_factors, exp_vals, asymptote=0.5)
print(f"Extrapolated zero-noise value (exponential factory): {zero_noise_value:.3f}")
```

```{code-cell} ipython3
zero_noise_value = zne.LinearFactory.extrapolate(scale_factors, exp_vals)
print(f"Extrapolated zero-noise value (linear factory): {zero_noise_value:.3f}")
```

```{code-cell} ipython3
zero_noise_value = zne.RichardsonFactory.extrapolate(scale_factors, exp_vals)
print(f"Extrapolated zero-noise value (Richardson factory): {zero_noise_value:.3f}")
```

```{code-cell} ipython3
NO_NOISE = True
t1 = time.time()
unmitigated = ibmq_executor(test_circuit)
t2 = time.time()
print(f"No noise result {unmitigated:.3f}")
```

These results suggest that for this particular choice of scaling factors (and this particular noisy simulation), the linear extrapolation scheme as well as the Richardson extrapolation scheme are closest to the no-noise result. In the following, we will chose the linear extrapolation scheme and peform perform the earlier quantum simulation once more with the scaling factors [1., 1.25, 1.5].

+++

# Linear factory

+++

In the following we will chose the linear extrapolation as our fitting factory. The zero noise extrapolated value and its standard error can be straight forwardly calculated (see e.g. *James et al. (2021)* {cite}`James_2021_statlearning`). One can calculate the intercept $b_0$ and its standard error of a linear fit as follows
\begin{equation}
    b_0 = \bar{y} - b_1\bar{x},
\end{equation}
\begin{equation}
    SE(b_0)^2 = \sigma^2\left(\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n(x_i - \bar{x})^2} \right),
\end{equation}
where the slope $b_1$ and its standard error can be calculated as
\begin{equation}
    b_1 = \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^n(x_i - \bar{x})^2},
\end{equation}
\begin{equation}
    SE(b_1)^2 = \frac{\sigma^2}{\sum_{i=1}^n(x_i - \bar{x})^2},
\end{equation}
and where $\sigma^2$ is the variance of the residual errors $\epsilon_i = y_i - b_0 - b_1 x_i$ , one can estimate this variance by calculating the residual standard error (RSE)
\begin{equation}
    RSE = \sqrt{\frac{RSS}{n - 2}},
\end{equation}
where RSS is the residual sum of squares given by
\begin{equation}
    RSS = \sum_{i=1}^n \epsilon_i^2.
\end{equation}


Note that $(x_i, y_i)$ are all data points, that is the various predictions of each shot for our expectation values at the different scaling factors. Also note that in the above, strictly speaking one assumes that errors $\epsilon_i$ are uncorrelated with common variance $\sigma^2$. Even if this is not exactly valid, the estimates on the standard error are usually good approximations.

In practice, the extrapolation of the intercept is performed using the linear fitting factory. To obtain the standard error on the intercept we write the following function

```{code-cell} ipython3
def LinearFit(xvals: list, yvals: list) -> list: 
    """Returns the intercept, slope and their standard errors
    of a linear fit Y = b0 + b1*X performed on the input data.
    
    Args:
        xvals: list of x values of the data set
        yvals: list of corresponding y values of the data set
    """
    # Transform lists to numpy arrays
    xvals = np.array(xvals)
    yvals = np.array(yvals)

    # Calculate the fitting parameters
    b1_num = np.sum((xvals - np.mean(xvals))*(yvals - np.mean(yvals)))
    b1_den = np.sum((xvals - np.mean(xvals))**2)

    b1 = b1_num/b1_den
    b0 = np.mean(yvals) - b1*np.mean(xvals)

    
    # Estimate the standard deviation 
    e_i = yvals - b0 - b1*xvals
    RSS = np.sum(e_i*e_i)
    RSE = np.sqrt(RSS/(len(e_i) - 2))
    
    ystd = np.sqrt(RSE)

    se_b0 = ystd*np.sqrt(1/len(xvals) + np.mean(xvals)**2/b1_den)
    se_b1 = ystd/np.sqrt(b1_den)

    # Store the fitting parameters
    fit_parameters = [b0, b1, se_b0, se_b1]
    
    return fit_parameters
```

```{code-cell} ipython3
# Prepare the x data set and the y data set
xvals = []
yvals = []
for kk in range(len(sz_vals)):
    for jj in range(len(sz_vals[kk])):
        xvals.append(scale_factors[kk])
        yvals.append(sz_vals[kk][jj])

# Perform the linear fit
fit_params = LinearFit(xvals, yvals)
b0, b1, se_b0, se_b1 = fit_params
```

```{code-cell} ipython3
print(f"The linear fit yields the intercept b0 = {b0:.3f} with standard error SE(b0) = {se_b0:.3f}.")
```

# Create a time evolution plot up to a certain time

+++

Finally, we will simulate the time evolution of the system up to various times to create a figure of the behaviour of the staggered $z$-magnetization throughout time. The system is initialized in the Néel state and we time evolve for a maximum of $n_{dt} = 40$ steps.

```{code-cell} ipython3
def simulate_circuits(circuit: qiskit.QuantumCircuit) -> list:
    '''Calculate the expectation value and its standard error of
    a given circuit or list of circuits.

    Args:
    circuit:     single circuit or list of (folded) circuits
    '''
    
    sz_vals = ibmq_executor_full(circuit)

    if type(sz_vals[0]) == list:
        # Linear fitting of mitigated value for folded circuits:
        exp_vals = [np.mean(kk) for kk in sz_vals]
        std_vals = [np.std(kk) for kk in sz_vals]
        expectation_value = zne.LinearFactory.extrapolate(scale_factors, exp_vals)

        # Prepare the x data set and the y data set
        xvals = []
        yvals = []
        for kk in range(len(sz_vals)):
            for jj in range(len(sz_vals[kk])):
                xvals.append(scale_factors[kk])
                yvals.append(sz_vals[kk][jj])
        
        # Perform the linear fit
        fit_params = LinearFit(xvals, yvals)
        b0, _, se_value, _ = fit_params
        
    else:
        # Calculate mitigated value and its standard 
        # error for single circuit
        expectation_value = np.mean(sz_vals)
        se_value = np.std(sz_vals)/np.sqrt(len(sz_vals))

    
    return [expectation_value, se_value]
```

```{code-cell} ipython3
# System parameters
L = 6
V = 1
Omega = 0.24
dt = 1
n_dt = 40

unmitigated_list = []
mitigated_list = []
mitigated_alt_list = []
nonoise_list = []

se_unmitigated_list = []
se_mitigated_list = []
se_mitigated_alt_list = []
se_nonoise_list = []
```

```{code-cell} ipython3
T1 = time.time()

# Results for t = 0:
# Initialise quantum circuit
circuit = qiskit.QuantumCircuit(L) 

# Initialise the Néel state |010101...>
for ii in range(1,L,2):
    circuit.x(ii)

# Measure in computational basis
circuit.measure_all() 


# Run circuit on the backends
NO_NOISE = False

# Unmitigated simulation
unmitigated, se_unmitigated = simulate_circuits(circuit)

# Mitigated simulation with the custom linear error bars
folded_circuits = [
        zne.scaling.fold_gates_at_random(circuit, scale)
        for scale in scale_factors
]
mitigated, se_mitigated = simulate_circuits(folded_circuits)

# Mitigated simulation with error bars from repeated simulations
# The advantage of this method is that it can be used for various
# fitting factories. However, one has to repeat the simulation 
# multiple times, possibly with a smaller number of shots (for 
# simulation time reasons we choose a small number of repetitions 
# here)
LinFac = zne.inference.LinearFactory(scale_factors=[1.,1.25,1.5])
mit_list = []
n_runs = 3
for _ in range(n_runs):
    mit = zne.execute_with_zne(circuit, ibmq_executor, factory=LinFac)
    mit_list.append(mit)
    
mit_alt, se_mit_alt = [np.mean(mit_list), 
                       np.std(mit_list, ddof=1)/np.sqrt(n_runs)] 


NO_NOISE = True

# No noise simulation
nonoise, se_nonoise = simulate_circuits(circuit)

# Store results
unmitigated_list.append(unmitigated)
mitigated_list.append(mitigated)
mitigated_alt_list.append(mit_alt)
nonoise_list.append(nonoise)
se_unmitigated_list.append(se_unmitigated)
se_mitigated_list.append(se_mitigated)
se_mitigated_alt_list.append(se_mit_alt)
se_nonoise_list.append(se_nonoise)


# Results for t > 0:
for ndt in range(1, n_dt + 1):
    t1 = time.time()
    
    # Initialise quantum circuit
    circuit = qiskit.QuantumCircuit(L) 

    # Initialise time step quantum circuit
    cc = trotter_evolution_H(L, V, Omega, dt)

    # Initialise the Néel state |010101...>
    for ii in range(1,L,2):
        circuit.x(ii)

    # Time evolve n_dt time steps
    for _ in range(ndt):
        circuit = circuit.compose(cc)

    # Measure in computational basis
    circuit.measure_all() 

    
    # Run circuit on the backends
    NO_NOISE = False

    # Unmitigated simulation
    unmitigated, se_unmitigated = simulate_circuits(circuit)

    # Mitigated simulation with the custom linear error bars
    folded_circuits = [
            zne.scaling.fold_gates_at_random(circuit, scale)
            for scale in scale_factors
    ]
    mitigated, se_mitigated = simulate_circuits(folded_circuits)

    # Mitigated simulation with error bars from repeated simulations
    LinFac = zne.inference.LinearFactory(scale_factors=[1.,1.25,1.5])
    mit_list = []
    for _ in range(n_runs):
        mit = zne.execute_with_zne(circuit, ibmq_executor, factory=LinFac)
        mit_list.append(mit)
        
    mit_alt, se_mit_alt = [np.mean(mit_list), 
                           np.std(mit_list, ddof=1)/np.sqrt(n_runs)]

    
    NO_NOISE = True
    # No noise simulation
    nonoise, se_nonoise = simulate_circuits(circuit)

    
    # Store results    
    unmitigated_list.append(unmitigated)
    mitigated_list.append(mitigated)
    mitigated_alt_list.append(mit_alt)
    nonoise_list.append(nonoise)
    se_unmitigated_list.append(se_unmitigated)
    se_mitigated_list.append(se_mitigated)
    se_mitigated_alt_list.append(se_mit_alt)
    se_nonoise_list.append(se_nonoise)
    

print("Time elapsed: {:.3f}s".format(time.time() - T1))
```

```{code-cell} ipython3
# Creation of the figure
time_list = np.arange(0, n_dt + 1, 1)

plt.figure()
plt.errorbar(time_list, unmitigated_list, yerr=se_unmitigated_list, 
             marker =".", c="r", linestyle =":", label = "Unmitigated")
plt.errorbar(time_list, mitigated_list, yerr=se_mitigated_list, 
             marker = ".", c="C1", linestyle ="--", label = "Mitigated (ZNE)")
plt.errorbar(time_list, mitigated_alt_list, yerr=se_mitigated_alt_list, 
             marker = ".", c="C0", linestyle ="--", label = "Mitigated (ZNE) alternative")
plt.errorbar(time_list, nonoise_list, yerr=se_nonoise_list, 
             marker = '.', c="k", label = "Ideal Trotter")

plt.xlim([time_list[0], time_list[-1]])
plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75], 
           ['-1', '-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75'])
plt.xlabel(r"$Vt$")
plt.ylabel(r"$<Z_\pi>/L$")
plt.legend()
plt.show()
```
