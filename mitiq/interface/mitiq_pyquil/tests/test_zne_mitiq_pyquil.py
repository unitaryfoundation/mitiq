# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for zne.py with PyQuil backend."""

import numpy as np
import pyquil

from mitiq import benchmarks, zne
from mitiq.interface.mitiq_pyquil.compiler import basic_compile

TEST_DEPTH = 30
QVM = pyquil.get_qc("1q-qvm")
QVM.qam.random_seed = 1337


def noiseless_executor(program: pyquil.Program) -> float:
    program.measure_all()
    program.num_shots = 1_000
    program = basic_compile(program)
    executable = QVM.compiler.native_quil_to_executable(program)
    results = QVM.run(executable).readout_data.get("ro")

    num_shots = len(results)
    return (
        num_shots - np.count_nonzero(np.count_nonzero(results, axis=1))
    ) / num_shots


def test_run_factory():
    (qp,) = benchmarks.generate_rb_circuits(
        n_qubits=1,
        num_cliffords=TEST_DEPTH,
        trials=1,
        return_type="pyquil",
    )

    fac = zne.inference.RichardsonFactory([1.0, 2.0, 3.0])

    fac.run(
        qp, noiseless_executor, scale_noise=zne.scaling.fold_gates_at_random
    )
    result = fac.reduce()
    assert np.isclose(result, 1.0, atol=1e-5)


def test_execute_with_zne():
    (qp,) = benchmarks.generate_rb_circuits(
        n_qubits=1,
        num_cliffords=TEST_DEPTH,
        trials=1,
        return_type="pyquil",
    )
    result = zne.execute_with_zne(qp, noiseless_executor)
    assert np.isclose(result, 1.0, atol=1e-5)


def test_mitigate_executor():
    (qp,) = benchmarks.generate_rb_circuits(
        n_qubits=1,
        num_cliffords=TEST_DEPTH,
        trials=1,
        return_type="pyquil",
    )

    new_executor = zne.mitigate_executor(noiseless_executor)
    result = new_executor(qp)
    assert np.isclose(result, 1.0, atol=1e-5)


@zne.zne_decorator(scale_noise=zne.scaling.fold_gates_at_random)
def decorated_executor(qp: pyquil.Program) -> float:
    return noiseless_executor(qp)


def test_zne_decorator():
    (qp,) = benchmarks.generate_rb_circuits(
        n_qubits=1,
        num_cliffords=TEST_DEPTH,
        trials=1,
        return_type="pyquil",
    )

    result = decorated_executor(qp)
    assert np.isclose(result, 1.0, atol=1e-5)
