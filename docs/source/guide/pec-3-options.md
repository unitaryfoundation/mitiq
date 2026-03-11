---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What additional options are available in PEC?

+++

The application of probabilistic error cancellation (PEC) with Mitiq requires two main steps:

1. Building {class}`.OperationRepresentation` objects expressing ideal gates as linear combinations of noisy gates;
2. Estimating expectation values with PEC, making use of the representations obtained in the previous step.

Both steps can be achieved with Mitiq in different ways and with different options.

+++

In the following code we use Qiskit as a frontend, but the workflow is the same for other frontends.

```{code-cell} ipython3
import numpy as np
import qiskit # Frontend library
from mitiq import pec  # Probabilistic error cancellation module
```

## Building {class}`.OperationRepresentation` objects

+++

Given the superoperator of an ideal gate $\mathcal G$, its quasi-probability representation is:

$$\mathcal G = \sum_\alpha \eta_\alpha \mathcal O_\alpha$$

where $\{\eta_\alpha\}$ are real coefficients and $\{\mathcal  O_\alpha\}$ are the implementable noisy gates,
i.e., those which can be actually applied by a noisy quantum computer.

For trace-preserving operations, the coefficients $\{\eta_\alpha\}$ form a quasi-probability distribution, i.e.:

$$\sum_\alpha \eta_\alpha = 1, \quad  \gamma = \| \eta \|_1= \sum_\alpha   |\eta_\alpha| \ge 1.$$

The value of $\gamma$ is related to the negative "volume" of the distribution and quantifies to the sampling
cost of PEC (see [What is the theory behind PEC?](pec-5-theory.md)).

+++

### Defining {class}`.NoisyOperation` objects

The noisy operations $\{\mathcal O_\alpha\}$ in the equation above can correspond to single noisy gates.
However, it is often useful to define a noisy operation as a sequence multiple noisy gates.
To have this flexibility, we associate to each noisy operation a small `QPROGRAM`,
i.e., a quantum circuit defined via a supported frontend. For example a basis
of noisy operations that are useful to represent the Hadamard gate in the presence of depolarizing noise is:

```{code-cell} ipython3
basis_circuit_h = qiskit.QuantumCircuit(1)
basis_circuit_h.h(0)

basis_circuit_hx = qiskit.QuantumCircuit(1)
basis_circuit_hx.h(0)
basis_circuit_hx.x(0)

basis_circuit_hy = qiskit.QuantumCircuit(1)
basis_circuit_hy.h(0)
basis_circuit_hy.y(0)

basis_circuit_hz = qiskit.QuantumCircuit(1)
basis_circuit_hz.h(0)
basis_circuit_hz.z(0)

basis_circuits = [basis_circuit_h, basis_circuit_hx, basis_circuit_hy, basis_circuit_hz] 

for c in basis_circuits:
    print(c)
```

Each element of `basis_circuits` describes "how to physically implement" a noisy operation 
$\mathcal O_\alpha$ on a noisy backend. To completely characterize a noisy operation we can also
specify the actual (non-unitary) quantum channel associated to it.
In Mitiq, this can be done using the {class}`.NoisyOperation` class.

+++

For example, assuming that each of the previous basis circuits is affected by a final depolarizing
channel, the following code cell generates the corresponding {class}`.NoisyOperation` objects.

```{code-cell} ipython3
from mitiq.pec.representations import local_depolarizing_kraus
from mitiq.pec.channels import kraus_to_super

# Compute depolarizing superoperator
BASE_NOISE = 0.2
depo_super = kraus_to_super(local_depolarizing_kraus(BASE_NOISE, num_qubits=1))

# Define the superoperator matrix of each noisy operation
super_matrices = [
    depo_super @ kraus_to_super([qiskit.quantum_info.Operator(c).data]) 
    for c in basis_circuits
]

# Define NoisyOperation objects combining circuits with channel matrices
noisy_operations = [
    pec.NoisyOperation(circuit=c, channel_matrix=m)
    for c, m in zip(basis_circuits, super_matrices)
]

print(f"{len(noisy_operations)} NoisyOperation objects defined.")
```

***Note:*** *A {class}`.NoisyOperation` can also be instantiated with `channel_matrix=None`.
In this case, however, the quasi-probability distribution must be known to the user
and cannot be derived by Mitiq with the procedure shown in the next section.*

+++

### Finding an optimal `OperationRepresentation`

+++

Similar to what we did for `basis_circuits`, we also define the `ideal_operation` that we aim to represent in the
form of a `QPROGRAM`. Assuming that we aim to represent the Hadamard gate, we have:

```{code-cell} ipython3
ideal_operation = qiskit.QuantumCircuit(1)
ideal_operation.h(0)
print(f"The ideal operation to expand in the noisy basis is:\n{ideal_operation}")
```

The Mitiq function {func}`.find_optimal_representation`
can be used to numerically obtain an  {class}`.OperationRepresentation` of the `ideal_operation`
in the basis of the noisy implementable gates (`noisy_operations`).

```{code-cell} ipython3
from mitiq.pec.representations import find_optimal_representation

h_rep = find_optimal_representation(ideal_operation, noisy_operations)
print(f"Optimal representation:\n{h_rep}")
```

The representation is optimal in the sense that, among all the possible representations,
it minimizes the one-norm of the quasi-probability distribution.
Behind the scenes, {func}`.find_optimal_representation` solves the following optimization problem:

$$\gamma^{\rm opt} = \min_{\substack{ \{ \eta_{\alpha} \}  \\ \{ \mathcal O_{ \alpha} \}}}
\left[ \sum_\alpha |\eta_{\alpha}| \right], \quad \text{ such that} \quad \mathcal G 
= \sum_\alpha \eta_\alpha \mathcal O_\alpha \, .$$

+++

### Manually defining an `OperationRepresentation`

+++

Instead of solving the previous optimization problem, an {class}`.OperationRepresentation` can
also be manually defined. This approach can be applied if the user already knows the quasi-probability
distribution ${\eta_\alpha}$.

```{code-cell} ipython3
# We assume to know the quasi-distribution
quasi_dist = h_rep.coeffs

# Manual definition of the OperationRepresentation
manual_h_rep = pec.OperationRepresentation(
    ideal=ideal_operation,
    noisy_operations=noisy_operations,
    coeffs=quasi_dist,
)

# Test that the manual definition is equivalent to h_rep
assert manual_h_rep == h_rep
```

**Note:** *For the particular case of depolarizing noise, Mitiq can directly create the
{class}`.OperationRepresentation` of an arbitrary `ideal_operation`, as shown in the next cell.*

```{code-cell} ipython3
from mitiq.pec.representations.depolarizing import represent_operation_with_local_depolarizing_noise

depolarizing_h_rep = represent_operation_with_local_depolarizing_noise(
    ideal_operation,
    noise_level=BASE_NOISE,
)

assert depolarizing_h_rep == h_rep
```

### Qubit-independent representations

It is possible to define a qubit-independent {class}`.OperationRepresentation` by setting the option `is_qubit_dependent` to `False`.

In this case, a single {class}`.OperationRepresentation` representing a gate acting on some arbitrary qubits can be used to mitigate
the same gate even if acting on different qubits.

```{code-cell} ipython3

qreg = qiskit.QuantumRegister(2) 
circuit = qiskit.QuantumCircuit(qreg)
circuit.h(0)
circuit.h(1)

# OperationRepresentation defined on different qubits
rep_qreg = qiskit.QuantumRegister(1, "rep_reg") 
ideal_op = qiskit.QuantumCircuit(rep_qreg)
ideal_op.h(rep_qreg)
hxcircuit = qiskit.QuantumCircuit(rep_qreg)
hxcircuit.h(0)
hxcircuit.x(0)
hzcircuit = qiskit.QuantumCircuit(rep_qreg)
hzcircuit.h(0)
hzcircuit.z(0)
noisy_hxop = pec.NoisyOperation(hxcircuit)
noisy_hzop = pec.NoisyOperation(hzcircuit)

rep = pec.OperationRepresentation(
    ideal=ideal_op,
    noisy_operations=[noisy_hxop, noisy_hzop],
    coeffs=[0.5, 0.5],
    is_qubit_dependent=False,
)
print(rep)
print("Using the same rep on a circuit with H gates acting on different qubits:")
sampled_circuits, _, _ = pec.sample_circuit(circuit, representations=[rep])
print(*sampled_circuits)
```

### Methods of the `OperationRepresentation` class

+++

The main idea of PEC is to estimate the average with respect to a 
quasi-probability distribution over noisy circuits with a probabilistic Monte-Carlo 
approach.
This can be obtained rewriting $\mathcal G = \sum_\alpha \eta_\alpha \mathcal O_\alpha \$ as:

$$\mathcal G = \gamma \sum_\alpha p(\alpha) \textrm{sign}(\eta_\alpha) \mathcal O_\alpha 
\quad p(\alpha):= |\eta_\alpha| / \gamma,$$

where $p(\alpha)$ is a (positive) well-defined probability distribution.
If we take a single sample from $p(\alpha)$, we obtain a noisy operation $\mathcal O_\alpha$ that
should be multiplied by the sign of the associated coefficient $\eta_\alpha$ and by $\gamma$. 

The method {meth}`.OperationRepresentation.sample()` can be used for this scope:

```{code-cell} ipython3
noisy_op, sign, coeff = h_rep.sample()
print(f"The sampled noisy operation is: {noisy_op}.")
print(f"The associated coefficient is {coeff:g}, whose sign is {sign}.")
```

**Note:** try re-executing the previous cell to get different samples.

+++

Other useful methods of {class}`.OperationRepresentation` are shown in the next cells.

```{code-cell} ipython3
# One-norm "gamma" quantifying the mitigation cost
h_rep.norm
```

```{code-cell} ipython3
# Quasi-probability distribution
print(h_rep.coeffs)
assert sum([abs(eta) for eta in h_rep.coeffs]) == h_rep.norm
```

```{code-cell} ipython3
# Positive and normalized distribution p(alpha)=|eta_alpha|/gamma
h_rep.distribution
```

## Estimating expectation values with PEC

+++

The second main step of PEC is to make use of the previously defined {class}`.OperationRepresentation`s to estimate
expectation values with respect to a quantum state prepared by a circuit of interest.
In the previous section we defined the representation of the Hadamard gate.
So, for simplicity, we consider a circuit that contains only Hadamard gates.

+++

### Defining a circuit of interest and an Executor

```{code-cell} ipython3
circuit = qiskit.QuantumCircuit(1)
for _ in range(4):
    circuit.h(0)
print(circuit)
```

In this case, the list of {class}`.OperationRepresentation`s that we need for PEC is simply:

```{code-cell} ipython3
representations = [h_rep]
```

In general `representations` will contain as many representations as the number of ideal
gates involved in `circuit`.

**Note:** *If a gate is in `circuit` but its {class}`.OperationRepresentation` is not listed in 
`representations`, Mitiq can still apply PEC. However, any errors associated to
that gate will not be mitigated. In practice, all the gates without {class}`.OperationRepresentation`s 
are treated by Mitiq as if they were noiseless.*

The executor must be defined by the user since it depends on the specific frontend and backend
(see the [Executors](executors.md) section).
Here, for simplicity, we import the basic {func}`.execute_with_noise` function from the Qiskit utilities of Mitiq.

```{code-cell} ipython3
from mitiq import Executor
from mitiq.interface.mitiq_qiskit import execute_with_noise, initialized_depolarizing_noise

def execute(circuit):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    circuit_copy = circuit.copy()    
    noise_model = initialized_depolarizing_noise(BASE_NOISE)
    projector_on_zero = np.array([[1, 0], [0, 0]])
    return execute_with_noise(circuit_copy, projector_on_zero, noise_model)

# Cast the execute function into an Executor class to record the execution history
executor = Executor(execute)
```

### Options for estimating expectation values

+++

In the ["How do I use PEC?"](pec-1-intro.md) section, we have shown how to apply PEC
with the minimum amount of options by simply calling the function {func}`.execute_with_pec()`
with the basic arguments `circuit`, `executor`, and `representations`.

In the next code-cell, we show additional options that can be used:

```{code-cell} ipython3
pec_value, pec_data = pec.execute_with_pec(
    circuit,
    executor,
    observable=None, # In this example the observable is implicit in the executor
    representations=representations,
    num_samples=5, # Number of PEC samples
    random_state=0, # Seed for reproducibility of PEC sampling
    full_output=True, # Return "pec_data" in addition to "pec_value"
)
```

Similar to other error mitigation modules, `observable` is an optional argument of 
{func}`.execute_with_pec`. If `observable` is `None` the executor must return an expectation value,
otherwise the executor must return a `mitiq.QuantumResult` from which the expectation value of the input
`observable` can be computed. See the [Executors](executors.md) section for more details. 


Another option that can be used, instead of `num_samples`, is `precision`.
Its default value is `0.03` and  quantifies the desired estimation accuracy. 

For a bounded observable $\|A\|\le 1$, `precision` approximates
$|\langle  A \rangle_{ \rm ideal} - \langle  A \rangle_{ \rm PEC}|$ (up to constant factors and up to
statistical fluctuations).  In practice, `precision` is used by Mitiq to automatically determine `num_samples`,
according to the formula: `num_samples` = $(\gamma /$ `precision`$)^2$, where $\gamma$ is the one-norm the circuit
quasi-probability distribution.
See ["What is the theory behind PEC?"](pec-5-theory.md) for more details on the sampling cost.

```{code-cell} ipython3
# Optional Executor re-initialization to clean the history
executor = Executor(execute)

pec_value, pec_data = pec.execute_with_pec(
    circuit,
    executor,
    observable=None, # In this example, the observable is implicit in the executor.
    representations=representations,
    precision=0.5, # The estimation accuracy.
    random_state=0, # Seed for reproducibility of probabilistic sampling of circuits.
    full_output=True, # Return pec_data in addition to pec_value
)
```

***Hint:** The value of `precision` used above is very large, in order to reduce the execution 
time. Try re-executing the previous cell with smaller values of `precision` to improve the result.*

### Analyzing the executed circuits

+++

As discussed in the [Executors](executors.md) section, we can extract the execution history
from `executor`. This is a way to see what Mitiq does behind the scenes which is independent from the error
mitigation technique.

```{code-cell} ipython3
print(
    f"During the previous PEC process, {len(executor.executed_circuits)} ",
    "circuits have been executed."
)
print(f"The first 5 circuits are:")

for c in executor.executed_circuits[:5]:
    print(c)
    
print(f"The corresponding noisy expectation values are:")  
for c in executor.quantum_results[:5]:
    print(c)
```

### Analyzing data of the PEC process

+++

Beyond the executed circuits, one may be interested in analyzing additional data related to the specifc PEC technique.
Setting `full_output=True`, this data is returned in `pec_data` as a dictionary.

```{code-cell} ipython3
print(pec_data.keys())
```

```{code-cell} ipython3
print(pec_data["num_samples"], "noisy circuits have been sampled and executed.")
```

The unbiased raw results, whose average is equal to `pec_value`, are stored under the `unbiased_estimators` key.
For example, the first 5 unbiased samples are:

```{code-cell} ipython3
pec_data["unbiased_estimators"][:5]
```

The statistical error `pec_error` corresponds to `pec_std` / `sqrt(num_samples)`, where `pec_std` is
the standard deviation of the unbiased samples, i.e., the square root of the mean squared deviation of
`unbiased_estimators` from `pec_value`.

```{code-cell} ipython3
pec_data["pec_error"]
```

Instead of the error printed above, one could use more advanced statistical techniques to estimate the
uncertainty of the PEC result. For example, given the raw samples contained in  `pec_data["unbiased_estimators"]`,
one can apply a [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) approach. Alternatively, 
a simpler but computationally more expensive approach is to perform multiple PEC estimations of the same expectation
value and compute the standard deviation of the results.


## Applying learning-based PEC

+++

Mitiq also contains functions for learning the quasiprobability representations of an ideal gate from Clifford circuit simulations,
with biased noise or depolarizing noise models.
The resulting quasiprobability representations can be used to obtain an error-mitigated expectation value via the `representations` argument of
{func}`.pec.execute_with_pec`.
The learning-based PEC workflow was inspired by the procedure described in *Strikis et al. PRX Quantum (2021)* {cite}`Strikis_2021_PRXQuantum`.

+++

The learning process is based on the execution of a set of training circuits on a noisy backend via a noisy executor and on a classical simulator via an ideal executor (for more information on executors, check out the {doc}`executors` portion of the documentation).
The training circuits are near-Clifford approximations of the input circuit. 
During training, the noise strength parameter is used to calculate quasiprobability representations of the ideal gate with
a depolarizing noise model.
The representations are then input into {func}`.pec.execute_with_pec` to obtain an error-mitigated expecation value from execution of the
training circuit, for comparison with the ideal expecation value obtained from classical simulation of the training circuit.
The optimizer used in the learning function is from {py:func}`scipy.optimize.minimize`.
The default optimization method in the learning function is `Nelder-Mead`, as that appears to work best for this particular problem setup.


In addition to specifying the input operation, the circuit of interest, and the ideal and noisy executors, the user should specify the number
of training circuits, the fraction of non-Clifford gates in the training circuits, an initial guess for noise strength, and in the case of
biased noise, an initial guess for a noise bias. 
The user can also set options for the intermediate executions of {func}`.pec.execute_with_pec()` during the training process as a dictionary in
`pec_kwargs`, specify the {doc}`observable <observables>` of which the expecation value is to be computed, and
enter a dictionary of additional data and options including optimization method (supported by {py:func}`scipy.optimize.minimize`) and settings
for the chosen optimization method.


```{note}
Using `learn_depolarizing_noise_parameter` and `learn_biased_noise_parameters` may require some tuning.
One of the main challenges is setting a good value of `num_samples` in the PEC options `pec_kwargs`.
Setting a small value of `num_samples` is typically necessary to obtain a reasonable execution time.
On the other hand, using a number of PEC samples that is too small can result in a large statistical error, ultimately causing the optimization
process to fail.
```

### Learning quasiprobability representations with a depolarizing noise model

In cases where the noise of the backend can be approximated by a depolarizing noise model,
{func}`.pec.representations.learning.learn_depolarizing_noise_parameter` can be used to learn the noise strength `epsilon` associated to a set
of input operations.

```{code-cell} ipython3
:tags: ["skip-execution"]

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import (
    depolarizing_error,
)
from mitiq import Observable, PauliString
from mitiq.pec.representations.learning import (
    learn_depolarizing_noise_parameter,
)

circuit = qiskit.QuantumCircuit(2)
circuit.rx(1.14 * np.pi, 0)
circuit.rz(0, 0)
circuit.cx(1, 0)
circuit.rx(1.71 * np.pi, 0)
circuit.rx(1.14 * np.pi, 1)

observable = Observable(PauliString("XZ"), PauliString("YY")).matrix()

# set up ideal simulator
def ideal_execute(circuit):
    """Simulate (training) circuits without noise"""
    circuit_copy = circuit.copy()
    noise_model = initialized_depolarizing_noise(0.0)
    return execute_with_noise(circuit_copy, observable, noise_model)


epsilon = 0.05  # noise level for simulation in noisy executor


def noisy_execute(circuit):
    """Simulate circuit with a depolarizing noise model"""
    circuit_copy = circuit.copy()
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(epsilon, 2), ["cx"])
    return execute_with_noise(circuit_copy, observable, noise_model)


operations_to_learn = qiskit.QuantumCircuit(2)
operations_to_learn.cx(1, 0)


[success, epsilon_opt] = learn_depolarizing_noise_parameter(
    operations_to_learn=[operations_to_learn],  # learn rep of Hadamard
    circuit=circuit,
    ideal_executor=Executor(ideal_execute),
    noisy_executor=Executor(noisy_execute),
    pec_kwargs={"num_samples": 500},
    num_training_circuits=5,
    fraction_non_clifford=0.2,
    
    epsilon0=0.9 * epsilon,  # initial guess for epsilon
)
```

Upon completing the optimization loop, {func}`.representations.learning.learn_depolarizing_noise_parameter` returns a flag indicating
whether the optimizer exited successfully, in addition to the optimized noise strength, which can then be input into 
{func}`.representations.depolarizing.represent_operation_with_local_depolarizing_noise` to calculate quasiprobability representations of the
ideal gate in terms of noisy gates.

```{code-cell} ipython3
:tags: ["skip-execution"]

representations = represent_operation_with_local_depolarizing_noise(
    operations_to_learn, epsilon_opt
)
```

### Learning quasiprobability representations with a biased noise model

In cases where the noise of the backend can be approximated by a combined depolarizing and dephasing noise model with a bias factor, also
referred to as a biased noise model, `pec.representations.learning.learn_biased_noise_parameters` can be used to learn the noise strength
`epsilon` and noise bias `eta`  associated to a set of input operations. 


The single-qubit biased noise channel is given by:

$$
\mathcal{D}(\epsilon) = (1 - \epsilon)\mathbb{1}
+ \epsilon\Big(\frac{\eta}{\eta + 1} \mathcal{Z}
+ \frac{1}{3}\frac{1}{\eta + 1}(\mathcal{X} + \mathcal{Y}
+ \mathcal{Z})\Big)
$$ 


For multi-qubit operations, the noise channel used is the tensor product of the local single-qubit channels.

```{code-cell} ipython3
:tags: ["skip-execution"]

from mitiq.pec.representations.learning import (
    learn_biased_noise_parameters,
)

epsilon = 0.05
eta = 0

# simulate biased noise occurs on the CNOT gates
def noisy_execute(circuit):
    circuit_copy = circuit.copy()
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(epsilon, 2), ["cx"])
    return execute_with_noise(circuit_copy, observable, noise_model)


operations_to_learn = qiskit.QuantumCircuit(2)
operations_to_learn.cx(1, 0)

[success, epsilon_opt, eta_opt] = learn_biased_noise_parameters(
    operations_to_learn=[operations_to_learn],  # learn rep of CNOT
    circuit=circuit,
    ideal_executor=Executor(ideal_execute),
    noisy_executor=Executor(noisy_execute),
    pec_kwargs={"num_samples": 500, "random_state": 1},
    num_training_circuits=5,
    fraction_non_clifford=0.2,
    training_random_state=np.random.RandomState(1),
    epsilon0=1.01 * 0.05,  # initial guess for noise strength
    eta0= eta + 0.01,  # initial guess for noise bias
)
```

Upon completing the optimization loop, {func}`.representations.learning.learn_biased_noise_parameters` returns a flag indicating whether the
optimizer exited successfully, in addition to the optimized noise strength and noise bias, which can then be input into 
{func}`.represent_operation_with_local_biased_noise` to calculate quasiprobability representations of the ideal gate in terms of noisy gates.

```{code-cell} ipython3
:tags: ["skip-execution"]

from pec.representations import (
    represent_operation_with_local_biased_noise,
)

representations = represent_operation_with_local_biased_noise(
    operations_to_learn, epsilon_opt, eta_opt
)
```
