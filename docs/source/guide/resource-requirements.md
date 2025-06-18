---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Resource Requirements

When running error mitigation protocols it is important to have an understanding as to how much and what type of additional resources will be used in the protcol.
Most techniques use some combination of

1. additional circuit executions,
2. additional gates,
3. additional qubits, and
4. additional classical post-processing.

This guide will focus on the additional quantum resources required by QEM techniques (i.e. 1, 2, and 3) as additional classical post-processing is generally "free" in comparison to quantum resources.

```{note}
Ideally, the cost of running a quantum error mitigation (QEM) experiment would be measured in dollars or QPU runtime.
However, due to the lack of a standardized API from providers to estimate cost, a more practical approach is to measure cost in terms of the items above.

The term "overhead" is sometimes used to refer to the additional quantum resources required by a QEM technique.
```

## Measuring Resource Requirements

This can be done by using mitiq's two stage application of `mitiq.xyz.construct_circuits` and `mitiq.xyz.combine_results`.

The example below demonstrates this using Layerwise Richardson Extrapolation (LRE) applied to a Greenberger-Horne-Zeilinger (GHZ) circuit, but the technique and particular circuit are not important.

## Setup

Before measuring either value, the circuit and a method to count the gates should be set up. In this example, we will set up a 3 qubit GHZ circuit and circuit counting function in cirq.

```{code-cell} ipython3
import cirq
from mitiq.benchmarks import generate_ghz_circuit
from mitiq import lre

num_qubits = 3
circuit = generate_ghz_circuit(num_qubits)

def count_gates(circ):
     # all_operations is a generator
    return sum(1 for _ in circ.all_operations())
```

## Measuring the number of circuits required

After using `construct_circuits` for a technique, a list of folded circuits should be returned.
The length of this list is the number of circuits required by the technique.

```{code-cell} ipython3
original_gate_count = count_gates(circuit)
print(f"Original circuit gate count: {original_gate_count}")

degree = 2
fold_multiplier = 2  # scaling parameter
folded_circuits = lre.construct_circuits(circuit, degree, fold_multiplier)

print(f"Number of circuits required (folded): {len(folded_circuits)}")
```

## Measuring the number of additional gates

The number of additional gates can be measured by comparing the original circuit to each folded circuit and computing the difference.

```{code-cell} ipython3
# Count the original number of gates
original_gate_count = count_gates(circuit)
print(f"Original circuit gate count: {original_gate_count}")

# Compare the number of folded gates to the original count
for i, folded in enumerate(folded_circuits, start=1):
    gate_count = count_gates(folded)
    added_gates = gate_count - original_gate_count
    print(f"Folded Circuit {i}: {gate_count} extra gates")
```

## Further Reading

More information on simulating and executing our GHZ circuit with LRE can be found [here](./lre-1-intro.md).
