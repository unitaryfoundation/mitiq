# Code Citations

## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q
```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q:
```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qu
```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit
```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map
```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map[q
```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map[q])


```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map[q])

       
```


## License: GPL-3.0
https://github.com/unitaryfund/mitiq/blob/39579b26550bae86b8e651253c6f265bdd0fa631/mitiq/observable/pauli.py

```
(
        circuit: cirq.Circuit, paulis: "PauliStringCollection"
    ) -> cirq.Circuit:
        # Transform circuit to canonical qubit layout.
        qubit_map = dict(
            zip(
                sorted(circuit.all_qubits()),
                cirq.LineQubit.range(len(circuit.all_qubits())),
            )
        )
        circuit = circuit.transform_qubits(lambda q: qubit_map[q])

        if
```

