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
orphan: true
---

# Solution to hands-on lab on error mitigation with Mitiq.

+++

This is a hands-on notebook created for the [`SQMS/GGI 2022 Summer School on Quantum Simulation of Field Theories`](https://sqmscenter.fnal.gov/opportunities/summer-schools/#2022). 

It is a guided tutorial on error mitigation with Mitiq and is focused on the zero-noise extrapolation (ZNE) technique. As this is
intended to be a hands-on exercise, the solutions to the examples are provided in this notebook.  

Useful links :

- [`Mitiq repository`](https://github.com/unitaryfoundation/mitiq)
- [`Mitiq documentation`](https://mitiq.readthedocs.io/en/stable/)
- [`Mitiq docs on ZNE`](https://mitiq.readthedocs.io/en/stable/guide/zne.html)
- [`Mitiq white paper`](https://arxiv.org/abs/2009.04417)
- [`Unitary Foundation`](https://unitary.foundation)

```{figure} ../img/zne_workflow2_steps.png
---
width: 400
name: figzne1
---
The diagram shows the workflow of the zero noise extrapolation (ZNE) technique in Mitiq.
```

## Checking Python packages are installed correctly

This notebook was tested with **Mitiq v0.23.0** and **qiskit v0.41.0**. It probably works with other versions too. Moreover, with minor changes, it can be adapted to quantum libraries that are different from Qiskit: Cirq, Braket, PyQuil, etc..

If you need to install Mitiq and/or Qiskit, you can uncomment and run the following cells.

```{code-cell} ipython3
# !pip install mitiq==0.23.0
```

```{code-cell} ipython3
# !pip install qiskit==0.41.0
```

You can check your locally installed version of Mitiq and of the associated frontend libraries by running the next cell.

```{code-cell} ipython3
from mitiq import about

about()
```

## Computing a quantum expectation value without error mitigation

+++

### Define the circuit of interest

For example, we define a circuit $U$ that prepares the GHZ state for $n$ qubits.

$$
U |00...0\rangle =  \frac{|00...0\rangle + |11...1\rangle}{\sqrt{2}}
$$

This can be done by manually defining a Qiskit circuit or by calling the Mitiq function `mitiq.benchmarks.generate_ghz_circuit()`.

```{code-cell} ipython3
from mitiq.benchmarks import generate_ghz_circuit


n_qubits = 7

circuit = generate_ghz_circuit(n_qubits=n_qubits, return_type="qiskit")
print("GHZ circuit:")
print(circuit)
```

Let us define the Hermitian observable:

$$ A = |00...0\rangle\langle 00...0| +  |11...1\rangle\langle 11...1|.$$

In the **absence of noise**, the expectation value of $A$ is equal to 1:  

$${\rm tr}(\rho_{\rm} A)= \langle 00...0| U^\dagger A U |00...0\rangle= \frac{1}{2} + \frac{1}{2}=1.$$

In practice this means that, when measuring the state in the computational basis, we can only obtain either the bitstring  $00\dots 0$ or the biststring $11\dots 1$.

In the **presence of noise** instead, the expectation value of the same observable $A$ will be smaller.
Let's verify this fact, before applying any error mitigation.

+++

### Run the circuit with a noiseless backend and with a noisy backend

**Hint:** As a noiseless backend you can use the `AerSimulator` class. As a noisy backend you can use a _fake_ (simulated) device as shown [here](https://qiskit.org/documentation/apidoc/providers_fake_provider.html).

```{code-cell} ipython3
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2 as FakeJakarta  # Fake (simulated) QPUs

# Number of measurements
shots = 10 ** 5
```

We first execute the circuit on an ideal noiseless simulator.

```{code-cell} ipython3
ideal_backend = AerSimulator()

# Append measurement gates
circuit_to_run = circuit.copy()
circuit_to_run.measure_all()

ideal_result = ideal_backend.run(circuit_to_run, shots=shots).result()
ideal_counts = ideal_result.get_counts(circuit_to_run)
plot_histogram(ideal_counts, title='Counts for an ideal GHZ state')
```

We now execute the same circuit on a noisy backend (a classical emulator of a real QPU)

```{code-cell} ipython3
noisy_backend = FakeJakarta() # QPU emulator

# Compile the circuit into the native gates of the backend
compiled_circuit = transpile(circuit_to_run, noisy_backend)
```

```{code-cell} ipython3
# Run the simulation on the noisy backend
noisy_result = noisy_backend.run(compiled_circuit, shots=shots).result()
noisy_counts = noisy_result.get_counts(compiled_circuit)
plot_histogram(noisy_counts, title='Counts for a noisy GHZ state', figsize=(15, 5))
```

```{code-cell} ipython3
ideal_expectation_value = (ideal_counts[n_qubits * "0"] + ideal_counts[n_qubits * "1"]) / shots
print(f"The ideal expectation value is <A> = {ideal_expectation_value}")

noisy_expectation_value = (noisy_counts[n_qubits * "0"] + noisy_counts[n_qubits * "1"]) / shots
print(f"The noisy expectation value is <A> = {noisy_expectation_value}")
```

## Apply zero-noise extrapolation with Mitiq

Before using Mitiq we need wrap the previous code into a function that takes as input a circuit and returns the noisy expectation value of the observable $A$. This function will be used by Mitiq as a black box during the error mitigation process.

```{code-cell} ipython3
def execute(compiled_circuit):
    """Executes the input circuits and returns the expectation value of A=|00..0><00..0| + |11..1><11..1|."""
    print("Executing a circuit of depth:", compiled_circuit.depth())
    noisy_backend = FakeJakarta()
    noisy_result = noisy_backend.run(compiled_circuit, shots=shots).result()
    noisy_counts = noisy_result.get_counts(compiled_circuit)
    noisy_expectation_value = (noisy_counts[n_qubits * "0"] + noisy_counts[n_qubits * "1"]) / shots
    return noisy_expectation_value
```

Let us check if the function works as expeted.

```{code-cell} ipython3
print(f"The noisy expectation value is <A> = {execute(compiled_circuit)}")
```

We can now apply zero-noise extrapolation with Mitiq. Without advanced options, this requires a single line of code.

```{code-cell} ipython3
from mitiq import zne

zne_value = zne.execute_with_zne(compiled_circuit, executor=execute)
print(f"The error mitigated expectation value is <A> = {zne_value}")
```

**Note:** As you can see from the printed output, Mitiq calls the execute function multiple times (3 in this case) to evaluate circuits of different depths in order to extrapolate the ideal result.

Let us compare the absolute estimation error obtained with and without Mitiq.

```{code-cell} ipython3
print(f"Error without Mitiq: {abs(ideal_expectation_value - noisy_expectation_value)}")
print(f"Error with Mitiq: {abs(ideal_expectation_value - zne_value)}")
```

## Explicitly selecting the noise-scaling method and the extrapolation method

```{code-cell} ipython3
from mitiq import zne

# Select a noise scaling method
folding_function = zne.scaling.fold_global

# Select an inference method
factory = zne.inference.RichardsonFactory(scale_factors = [1.0, 2.0, 3.0])

zne_value = zne.execute_with_zne(
    compiled_circuit, 
    executor=execute,
    scale_noise = folding_function,
    factory = factory,
)
factory.plot_fit()
print(f"The error mitigated expectation value is <A> = {zne_value}")
```

## What happens behind the scenes? A low-level application of ZNE

In Mitiq one can indirectly amplify noise by intentionally increasing the depth of the circuit in different ways.

For example, the function `zne.scaling.fold_gates_at_random()` applies transformation $G \rightarrow G G^\dagger G$ to each gate of the circuit (or to a random subset of gates).

+++

### STEP 1: Noise-scaled expectation values are evaluated via gate-level "unitary folding" transformations

```{code-cell} ipython3
locally_folded_circuit = zne.scaling.fold_gates_at_random(circuit, scale_factor=3)

print("Locally folded GHZ circuit:")
print(locally_folded_circuit)
```

**Note:** To get a simple visualization, we did't apply the preliminary circuit transpilation that we used in the previous section.

Alternatively, the function `zne.scaling.fold_global()` applies the transformation $U \rightarrow U U^\dagger U$ to the full circuit.

```{code-cell} ipython3
globally_folded_circuit = zne.scaling.fold_global(circuit, scale_factor=3)
print("Globally folded GHZ circuit:")
print(globally_folded_circuit)
```

In both cases, the results are longer circuits which are more sensitive to noise. Those circuits can be used to evaluate noise scaled expectation values.

For example, let's use global folding to evaluate a list of noise scaled expectation values.

```{code-cell} ipython3
scale_factors = [1.0, 2.0, 3.0]
# It is usually better apply unitary folding to the compiled circuit
noise_scaled_circuits = [zne.scaling.fold_global(compiled_circuit, s) for s in scale_factors]

# We run all the noise scaled circuits on the noisy backend
noise_scaled_vals = [execute(c) for c in noise_scaled_circuits]

print("Noise-scaled expectation values:", noise_scaled_vals)
```

### STEP 2: Inference of the ideal result via zero-noise extrapolation
Given the list of noise scaled expectation values, one can extrapolate the zero-noise limit. This is the final classical post-processing step.

```{code-cell} ipython3
# Initialize a Richardson extrapolation object
richardson_factory = zne.RichardsonFactory(scale_factors)

# Load the previously measured data
for s, val in zip(scale_factors, noise_scaled_vals):
    richardson_factory.push({"scale_factor": s}, val)

print("The Richardson zero-noise extrapolation is:", richardson_factory.reduce())
_ = richardson_factory.plot_fit()
```

```{code-cell} ipython3
# Initialize a linear extrapolation object
linear_factory = zne.LinearFactory(scale_factors)

# Load the previously measured data
for s, val in zip(scale_factors, noise_scaled_vals):
    linear_factory.push({"scale_factor": s}, val)

print("The linear zero-noise extrapolation is", linear_factory.reduce())
_ = linear_factory.plot_fit()
```

**Note:** We evaluated two different extrapolations without measuring the system twice. This is possible since the final extrapolation step is simply a classical post-processing of the same measured data.

+++

## References

1. _Mitiq: A software package for error mitigation on noisy quantum computers_, R. LaRose at al., [arXiv:2009.04417](https://arxiv.org/abs/2009.04417) (2020).

2. _Efficient variational quantum simulator incorporating active error minimisation_, Y. Li, S. C. Benjamin, [arXiv:1611.09301](https://arxiv.org/abs/1611.09301) (2016).

3. _Error mitigation for short-depth quantum circuits_, K. Temme, S. Bravyi, J. M. Gambetta, [arXiv:1612.02058](https://arxiv.org/abs/1612.02058) (2016).

4. _Digital zero noise extrapolation for quantum error mitigation_, 
T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, W. J. Zeng,
[arXiv:2005.10921](https://arxiv.org/abs/2005.10921) (2020).
