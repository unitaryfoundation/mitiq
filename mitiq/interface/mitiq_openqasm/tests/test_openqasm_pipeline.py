# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests that error-mitigation entrypoints accept raw OpenQASM
program strings (the frontend added in v0.49.0)."""

import numpy as np
import pytest
from openqasm3 import parse

from mitiq import ddd, lre, pec, zne
from mitiq.ddd import rules
from mitiq.interface import convert_to_mitiq
from mitiq.pec.representations import (
    represent_operations_in_circuit_with_local_depolarizing_noise,
)
from mitiq.pt import (
    generate_pauli_twirl_variants,
    twirl_CNOT_gates,
    twirl_CZ_gates,
)
from mitiq.typing import QasmStringType

QASM3_TWO_QUBIT = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

QASM2_TWO_QUBIT = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

QASM3_MEAS = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

QASM3_MULTI_REG = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] a;
qubit[1] b;
h a[0];
cx a[0], b[0];
"""


def _executor(_circuit):
    return 0.5


def _assert_qasm_list(circuits, expected_len):
    """Assert circuits is a list of QASM strings that re-parse as OpenQASM."""
    assert isinstance(circuits, list)
    assert len(circuits) == expected_len
    for c in circuits:
        assert isinstance(c, str)
        # The current mitiq to_openqasm implementation emits OpenQASM 2.0
        # via the qiskit backend. Parsing with openqasm3 handles both.
        parse(c)


# ---- Regression: bare str used to crash with AttributeError ----


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_zne_construct_circuits_accepts_openqasm_string(qasm):
    scale_factors = [1.0, 3.0, 5.0]
    out = zne.construct_circuits(qasm, scale_factors=scale_factors)
    _assert_qasm_list(out, len(scale_factors))


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_zne_execute_with_zne_accepts_openqasm_string(qasm):
    result = zne.execute_with_zne(qasm, _executor)
    assert isinstance(result, float)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_ddd_construct_circuits_accepts_openqasm_string(qasm):
    out = ddd.construct_circuits(qasm, rule=rules.xx, num_trials=2)
    _assert_qasm_list(out, 2)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_ddd_execute_with_ddd_accepts_openqasm_string(qasm):
    result = ddd.execute_with_ddd(qasm, _executor, rule=rules.xx)
    assert isinstance(result, float)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_lre_construct_circuits_accepts_openqasm_string(qasm):
    out = lre.construct_circuits(qasm, degree=2, fold_multiplier=1)
    assert isinstance(out, list) and len(out) > 0
    for c in out:
        assert isinstance(c, str)
        parse(c)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_lre_execute_with_lre_accepts_openqasm_string(qasm):
    result = lre.execute_with_lre(qasm, _executor, degree=2, fold_multiplier=1)
    assert isinstance(result, float)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_pt_generate_pauli_twirl_variants_accepts_openqasm_string(qasm):
    variants = generate_pauli_twirl_variants(qasm, num_circuits=3)
    _assert_qasm_list(variants, 3)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_pt_twirl_CNOT_gates_accepts_openqasm_string(qasm):
    variants = twirl_CNOT_gates(qasm, num_circuits=2)
    _assert_qasm_list(variants, 2)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_pt_twirl_CZ_gates_accepts_openqasm_string(qasm):
    variants = twirl_CZ_gates(qasm, num_circuits=2)
    _assert_qasm_list(variants, 2)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_pec_construct_circuits_accepts_openqasm_string(qasm):
    cirq_circuit, _ = convert_to_mitiq(qasm)
    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        cirq_circuit, noise_level=0.01
    )
    circuits, signs, norm = pec.construct_circuits(
        qasm,
        representations=reps,
        num_samples=4,
        random_state=0,
        full_output=True,
    )
    _assert_qasm_list(circuits, 4)
    assert np.asarray(signs).shape == (4,)
    assert isinstance(norm, float)


@pytest.mark.parametrize("qasm", [QASM3_TWO_QUBIT, QASM2_TWO_QUBIT])
def test_pec_execute_with_pec_accepts_openqasm_string(qasm):
    cirq_circuit, _ = convert_to_mitiq(qasm)
    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        cirq_circuit, noise_level=0.01
    )
    result = pec.execute_with_pec(
        qasm,
        _executor,
        representations=reps,
        num_samples=4,
        random_state=0,
    )
    assert isinstance(result, float)


# ---- Type-preservation ----


def test_construct_circuits_return_is_qasm_string_subclass():
    """The wrapped input propagates: outputs are QasmStringType instances,
    which have a ``__module__`` attribute so downstream ``.__module__`` reads
    do not re-crash on the return path."""
    out = zne.construct_circuits(QASM3_TWO_QUBIT, scale_factors=[1.0, 3.0])
    for c in out:
        assert isinstance(c, QasmStringType)
        assert hasattr(c, "__module__")
        assert "openqasm" in c.__module__


# ---- Additional real-world QASM shapes ----


def test_zne_with_measurement_bearing_qasm():
    """Circuit with an explicit measurement instruction."""
    out = zne.construct_circuits(QASM3_MEAS, scale_factors=[1.0, 3.0])
    _assert_qasm_list(out, 2)


def test_zne_with_multi_qreg_qasm():
    """Circuit with more than one quantum register."""
    out = zne.construct_circuits(QASM3_MULTI_REG, scale_factors=[1.0, 3.0])
    _assert_qasm_list(out, 2)


# ---- Negative test: garbage input surfaces CircuitConversionError,
# not AttributeError ----


def test_garbage_qasm_raises_circuit_conversion_error():
    from mitiq.interface import CircuitConversionError

    with pytest.raises(CircuitConversionError):
        zne.construct_circuits(
            "this is not valid qasm", scale_factors=[1.0, 3.0]
        )
