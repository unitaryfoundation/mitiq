# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import cirq
import numpy as np


def random_x_z_circuit(qubits, n_moments, random_state) -> cirq.Circuit:
    angles = np.linspace(0.0, 2 * np.pi, 6)
    oneq_gates = [cirq.ops.rz(a) for a in angles] + [cirq.ops.rx(np.pi / 2)]
    gate_domain = {oneq_gate: 1 for oneq_gate in oneq_gates}

    return cirq.testing.random_circuit(
        qubits=qubits,
        n_moments=n_moments,
        op_density=1.0,
        gate_domain=gate_domain,
        random_state=random_state,
    )


def random_x_z_cnot_circuit(qubits, n_moments, random_state) -> cirq.Circuit:
    angles = np.linspace(0.0, 2 * np.pi, 8)
    oneq_gates = [
        gate(a) for a in angles for gate in (cirq.ops.rx, cirq.ops.rz)
    ]
    gate_domain = {oneq_gate: 1 for oneq_gate in oneq_gates}
    gate_domain.update({cirq.ops.CNOT: 2})

    return cirq.testing.random_circuit(
        qubits=qubits,
        n_moments=n_moments,
        op_density=1.0,
        gate_domain=gate_domain,
        random_state=random_state,
    )
