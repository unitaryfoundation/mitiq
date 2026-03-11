# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for zero-noise extrapolation."""

import functools
import random
from unittest.mock import Mock

import cirq
import numpy as np
import pytest
import qiskit
import qiskit.circuit
from matplotlib.figure import Figure
from qiskit_aer import AerSimulator

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES
from mitiq.benchmarks.randomized_benchmarking import generate_rb_circuits
from mitiq.interface import accept_any_qprogram_as_input, convert_from_mitiq
from mitiq.interface.mitiq_braket import to_braket
from mitiq.interface.mitiq_cirq import (
    compute_density_matrix,
    sample_bitstrings,
)
from mitiq.interface.mitiq_pennylane import to_pennylane
from mitiq.interface.mitiq_pyquil import to_pyquil
from mitiq.interface.mitiq_qibo import to_qibo
from mitiq.interface.mitiq_qiskit import (
    execute_with_shots_and_noise,
    initialized_depolarizing_noise,
    to_qiskit,
)
from mitiq.observable import Observable, PauliString
from mitiq.zne import (
    execute_with_zne,
    inference,
    mitigate_executor,
    scaling,
    visualize_fits,
    zne_decorator,
)
from mitiq.zne.inference import (
    AdaExpFactory,
    ExtrapolationError,
    LinearFactory,
    PolyFactory,
    RichardsonFactory,
)
from mitiq.zne.scaling import (
    fold_all,
    fold_gates_at_random,
    fold_global,
    get_layer_folding,
    insert_id_layers,
)
from mitiq.zne.zne import combine_results, construct_circuits

BASE_NOISE = 0.007
TEST_DEPTH = 30
ONE_QUBIT_GS_PROJECTOR = np.array([[1, 0], [0, 0]])

npX = np.array([[0, 1], [1, 0]])
"""Defines the sigma_x Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

npZ = np.array([[1, 0], [0, -1]])
"""Defines the sigma_z Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

# Default qubit register and circuit for unit tests
qreg = cirq.GridQubit.rect(2, 1)
circ = cirq.Circuit(cirq.ops.H.on_each(*qreg), cirq.measure_each(*qreg))


@accept_any_qprogram_as_input
def generic_executor(circuit, noise_level: float = 0.1) -> float:
    """Executor that simulates a circuit of any type and returns
    the expectation value of the ground state projector."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    result = cirq.DensityMatrixSimulator().simulate(noisy_circuit)
    return result.final_density_matrix[0, 0].real


# Default executor for unit tests
def executor(circuit) -> float:
    wavefunction = circuit.final_state_vector(
        ignore_terminal_measurements=True
    )
    return np.real(wavefunction.conj().T @ np.kron(npX, npZ) @ wavefunction)


@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_with_observable_batched_factory(executor):
    observable = Observable(PauliString(spec="Z"))
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0))) * 20
    executor = functools.partial(
        executor, noise_model_function=cirq.depolarize
    )

    real_factory = PolyFactory(scale_factors=[1, 3, 5], order=2)
    mock_factory = Mock(spec_set=PolyFactory, wraps=real_factory)
    zne_val = execute_with_zne(
        circuit,
        executor=executor,
        observable=observable,
        factory=mock_factory,
    )

    mock_factory.run.assert_called_with(
        circuit, executor, observable, fold_gates_at_random, 1
    )
    assert 0 <= zne_val <= 2


@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_with_observable_adaptive_factory(executor):
    observable = Observable(PauliString(spec="Z"))
    circuit = cirq.Circuit(cirq.H.on(cirq.LineQubit(0))) * 20

    noisy_value = observable.expectation(circuit, sample_bitstrings)
    zne_value = execute_with_zne(
        circuit,
        executor=functools.partial(
            executor, noise_model_function=cirq.amplitude_damp
        ),
        observable=observable,
        factory=AdaExpFactory(steps=4, asymptote=0.5),
    )
    true_value = observable.expectation(
        circuit, functools.partial(compute_density_matrix, noise_level=(0,))
    )

    assert abs(zne_value - true_value) <= abs(noisy_value - true_value)


def test_with_observable_two_qubits():
    observable = Observable(
        PauliString(spec="XX", coeff=-1.21), PauliString(spec="ZZ", coeff=0.7)
    )
    circuit = cirq.Circuit(
        cirq.H.on(cirq.LineQubit(0)), cirq.CNOT.on(*cirq.LineQubit.range(2))
    )
    circuit += [circuit.copy(), cirq.inverse(circuit.copy())] * 20
    executor = compute_density_matrix

    noisy_value = observable.expectation(circuit, executor)
    zne_value = execute_with_zne(
        circuit,
        executor=functools.partial(
            executor, noise_model_function=cirq.depolarize
        ),
        observable=observable,
        factory=PolyFactory(scale_factors=[1, 3, 5], order=2),
    )
    true_value = observable.expectation(
        circuit, functools.partial(executor, noise_level=(0,))
    )

    assert abs(zne_value - true_value) <= 3 * abs(noisy_value - true_value)


@pytest.mark.parametrize(
    "fold_method",
    [
        fold_gates_at_random,
        insert_id_layers,
    ],
)
@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
@pytest.mark.parametrize("num_to_average", [1, 2, 5])
def test_execute_with_zne_no_noise(fold_method, factory, num_to_average):
    """Tests execute_with_zne with noiseless simulation."""
    zne_value = execute_with_zne(
        circ,
        executor,
        num_to_average=num_to_average,
        scale_noise=fold_method,
        factory=factory([1.0, 2.0, 3.0]),
    )
    assert np.isclose(zne_value, 0.0)


@pytest.mark.parametrize("factory", [LinearFactory, RichardsonFactory])
@pytest.mark.parametrize(
    "fold_method",
    [
        fold_gates_at_random,
        insert_id_layers,
    ],
)
def test_averaging_improves_zne_value_with_fake_noise(factory, fold_method):
    """Tests that averaging with Gaussian noise produces a better ZNE value
    compared to not averaging with several folding methods.

    For non-deterministic folding, the ZNE value with average should be better.
    For deterministic folding, the ZNE value should be the same.
    """
    for seed in range(5):
        rng = np.random.RandomState(seed)

        def noisy_executor(circuit) -> float:
            return executor(circuit) + rng.randn()

        zne_value_no_averaging = execute_with_zne(
            circ,
            noisy_executor,
            num_to_average=1,
            scale_noise=fold_gates_at_random,
            factory=factory([1.0, 2.0, 3.0]),
        )

        zne_value_averaging = execute_with_zne(
            circ,
            noisy_executor,
            num_to_average=10,
            scale_noise=fold_method,
            factory=factory([1.0, 2.0, 3.0]),
        )

        # True (noiseless) value is zero. Averaging should ==> closer to zero.
        assert abs(zne_value_averaging) <= abs(zne_value_no_averaging)


def test_execute_with_zne_bad_arguments():
    """Tests errors are raised when execute_with_zne is called with bad
    arguments.
    """
    with pytest.raises(TypeError, match="Argument `factory` must be of type"):
        execute_with_zne(circ, executor, factory=RichardsonFactory)

    with pytest.raises(TypeError, match="Argument `scale_noise` must be"):
        execute_with_zne(circ, executor, scale_noise=None)


def test_error_zne_decorator():
    """Tests that the proper error is raised if the decorator is
    used without parenthesis.
    """
    with pytest.raises(TypeError, match="Decorator must be used with paren"):

        @zne_decorator
        def test_executor(circuit):
            return 0


def test_doc_is_preserved():
    """Tests that the doc of the original executor is preserved."""

    def first_executor(circuit):
        """Doc of the original executor."""
        return 0

    mit_executor = mitigate_executor(first_executor)
    assert mit_executor.__doc__ == first_executor.__doc__

    @zne_decorator()
    def second_executor(circuit):
        """Doc of the original executor."""
        return 0

    assert second_executor.__doc__ == first_executor.__doc__


def qiskit_measure(circuit, qid) -> qiskit.QuantumCircuit:
    """Helper function to measure one qubit."""
    # Ensure that we have a classical register of enough size available
    if len(circuit.clbits) == 0:
        reg = qiskit.ClassicalRegister(qid + 1, "cbits")
        circuit.add_register(reg)
    circuit.measure(0, qid)
    return circuit


def qiskit_executor(qp: QPROGRAM, shots: int = 10000) -> float:
    # initialize a qiskit noise model
    expectation = execute_with_shots_and_noise(
        qp,
        shots=shots,
        obs=ONE_QUBIT_GS_PROJECTOR,
        noise_model=initialized_depolarizing_noise(BASE_NOISE),
        seed=1,
    )
    return expectation


def get_counts(circuit: qiskit.QuantumCircuit):
    return AerSimulator().run(circuit, shots=100).result().get_counts()


def test_qiskit_execute_with_zne():
    true_zne_value = 1.0

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)
    zne_value = execute_with_zne(circuit, qiskit_executor)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


@zne_decorator()
def qiskit_decorated_executor(qp: QPROGRAM) -> float:
    return qiskit_executor(qp)


def batched_qiskit_executor(circuits) -> list[float]:
    return [qiskit_executor(circuit) for circuit in circuits]


def test_qiskit_mitigate_executor():
    true_zne_value = 1.0

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)

    mitigated_executor = mitigate_executor(qiskit_executor)
    zne_value = mitigated_executor(circuit)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)
    batched_mitigated_executor = mitigate_executor(batched_qiskit_executor)
    batched_zne_values = batched_mitigated_executor([circuit] * 3)
    assert [
        abs(true_zne_value - batched_zne_value) < abs(true_zne_value - base)
        for batched_zne_value in batched_zne_values
    ]


def test_qiskit_zne_decorator():
    true_zne_value = 1.0

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)

    zne_value = qiskit_decorated_executor(circuit)
    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)


def test_qiskit_run_factory_with_number_of_shots():
    true_zne_value = 1.0

    scale_factors = [1.0, 3.0]
    shot_list = [10_000, 30_000]

    fac = inference.ExpFactory(
        scale_factors=scale_factors,
        shot_list=shot_list,
        asymptote=0.5,
    )

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)
    zne_value = fac.run(
        circuit,
        qiskit_executor,
        scale_noise=scaling.fold_gates_at_random,
    ).reduce()

    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)

    for i in range(len(fac._instack)):
        assert fac._instack[i] == {
            "scale_factor": scale_factors[i],
            "shots": shot_list[i],
        }


def test_qiskit_mitigate_executor_with_shot_list():
    true_zne_value = 1.0

    scale_factors = [1.0, 3.0]
    shot_list = [10_000, 30_000]

    fac = inference.ExpFactory(
        scale_factors=scale_factors,
        shot_list=shot_list,
        asymptote=0.5,
    )
    mitigated_executor = mitigate_executor(qiskit_executor, factory=fac)

    circuit = qiskit_measure(
        *generate_rb_circuits(
            n_qubits=1,
            num_cliffords=TEST_DEPTH,
            trials=1,
            return_type="qiskit",
        ),
        0,
    )
    base = qiskit_executor(circuit)
    zne_value = mitigated_executor(circuit)

    assert abs(true_zne_value - zne_value) < abs(true_zne_value - base)

    for i in range(len(fac._instack)):
        assert fac._instack[i] == {
            "scale_factor": scale_factors[i],
            "shots": shot_list[i],
        }


@pytest.mark.parametrize("order", [(0, 1), (1, 0), (0, 1, 2), (1, 2, 0)])
def test_qiskit_measurement_order_is_preserved_single_register(order):
    """Tests measurement order is preserved when folding, i.e., the dictionary
    of counts is the same as the original circuit on a noiseless simulator.
    """
    qreg, creg = (
        qiskit.QuantumRegister(len(order)),
        qiskit.ClassicalRegister(len(order)),
    )
    circuit = qiskit.QuantumCircuit(qreg, creg)

    circuit.x(qreg[0])
    for i in order:
        circuit.measure(qreg[i], creg[i])

    folded = scaling.fold_gates_at_random(circuit, scale_factor=1.0)

    assert get_counts(folded) == get_counts(circuit)


def test_qiskit_measurement_order_is_preserved_two_registers():
    """Tests measurement order is preserved when folding, i.e., the dictionary
    of counts is the same as the original circuit on a noiseless simulator.
    """
    n = 4
    qreg = qiskit.QuantumRegister(n)
    creg1, creg2 = (
        qiskit.ClassicalRegister(n // 2),
        qiskit.ClassicalRegister(n // 2),
    )
    circuit = qiskit.QuantumCircuit(qreg, creg1, creg2)

    circuit.x(qreg[0])
    circuit.x(qreg[2])

    # Some order of measurements.
    circuit.measure(qreg[0], creg2[1])
    circuit.measure(qreg[1], creg1[0])
    circuit.measure(qreg[2], creg1[1])
    circuit.measure(qreg[3], creg2[1])

    folded = scaling.fold_gates_at_random(circuit, scale_factor=1.0)

    assert get_counts(folded) == get_counts(circuit)


@pytest.mark.parametrize("circuit_type", SUPPORTED_PROGRAM_TYPES.keys())
def test_execute_with_zne_with_supported_circuits(circuit_type):
    # Define a circuit equivalent to the identity
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        cirq.H.on_each(qreg),
        cirq.CNOT(*qreg),
        cirq.CNOT(*qreg),
        cirq.H.on_each(qreg),
    )
    # Convert to one of the supported program types
    circuit = convert_from_mitiq(cirq_circuit, circuit_type)
    expected = generic_executor(circuit, noise_level=0.0)
    unmitigated = generic_executor(circuit)
    # Use odd scale factors for deterministic results
    fac = RichardsonFactory([1.0, 3.0, 5.0])
    zne_value = execute_with_zne(circuit, generic_executor, factory=fac)
    # Test zero noise limit is better than unmitigated expectation value
    assert abs(unmitigated - expected) > abs(zne_value - expected)


def test_layerwise_folding_with_zne():
    # Define a circuit equivalent to the identity
    qreg = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H.on_each(qreg),
        cirq.CNOT(*qreg),
        cirq.CNOT(*qreg),
        cirq.H.on_each(qreg),
    )
    circuit_depth = len(circuit)
    mock_executor = Mock(side_effect=lambda _: random.random())
    layer_to_fold = 0
    fold_layer_func = get_layer_folding(layer_to_fold)
    scale_factors = [1, 3, 5]
    factory = RichardsonFactory(scale_factors)

    execute_with_zne(
        circuit, mock_executor, factory=factory, scale_noise=fold_layer_func
    )
    assert mock_executor.call_count == len(scale_factors)
    circuit_depths = [
        len(args[0]) for args, kwargs in mock_executor.call_args_list
    ]
    assert circuit_depths == [
        circuit_depth,
        circuit_depth + 2,
        circuit_depth + 4,
    ]


def test_execute_with_zne_transpiled_qiskit_circuit():
    """Tests ZNE when transpiling to a Qiskit device. Note transpiling can
    introduce idle (unused) qubits to the circuit.
    """
    from qiskit_ibm_runtime.fake_provider import FakeSantiagoV2

    santiago = FakeSantiagoV2()
    backend = AerSimulator.from_backend(santiago)

    def execute(circuit: qiskit.QuantumCircuit, shots: int = 8192) -> float:
        job = backend.run(circuit, shots=shots)
        return job.result().get_counts().get("00", 0.0) / shots

    qreg = qiskit.QuantumRegister(2)
    creg = qiskit.ClassicalRegister(2)
    circuit = qiskit.QuantumCircuit(qreg, creg)
    for _ in range(10):
        circuit.x(qreg)

    circuit.measure(qreg, creg)
    circuit = qiskit.transpile(circuit, backend, optimization_level=0)

    true_value = 1.0
    zne_value = execute_with_zne(circuit, execute)

    # Note: Unmitigated value is also (usually) within 10% of the true value.
    # This is more to test usage than effectiveness.
    assert abs(zne_value - true_value) < 0.1


def test_execute_zne_on_qiskit_circuit_with_QFT():
    """Tests ZNE of a Qiskit device with a QFT gate."""

    def qs_noisy_simulation(
        circuit: qiskit.QuantumCircuit, shots: int = 1
    ) -> float:
        noise_model = initialized_depolarizing_noise(noise_level=0.02)
        backend = AerSimulator(noise_model=noise_model)
        job = backend.run(circuit.decompose(), shots=shots)
        return job.result().get_counts().get("0", 0.0) / shots

    circuit = qiskit.QuantumCircuit(1)
    circuit &= qiskit.circuit.library.QFT(1)
    circuit.measure_all()

    mitigated = execute_with_zne(circuit, qs_noisy_simulation)
    assert abs(mitigated) < 1000


@pytest.mark.parametrize(
    "noise_scaling_method",
    [fold_gates_at_random, insert_id_layers, fold_global, fold_all],
)
@pytest.mark.parametrize(
    "extrapolation_factory", [RichardsonFactory, LinearFactory]
)
@pytest.mark.parametrize(
    "conversion_func",
    [None, to_qiskit, to_braket, to_pennylane, to_pyquil, to_qibo],
)
def test_two_stage_zne(
    noise_scaling_method, extrapolation_factory, conversion_func
):
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        cirq.H.on_each(qreg),
        cirq.CNOT(*qreg),
        cirq.CNOT(*qreg),
        cirq.H.on_each(qreg),
    )
    if conversion_func is not None:
        frontend_circuit = conversion_func(cirq_circuit)
    else:
        frontend_circuit = cirq_circuit

    scale_factors = [1, 3, 5]
    circs = construct_circuits(
        frontend_circuit, scale_factors, noise_scaling_method
    )

    assert len(circs) == len(scale_factors)

    np.random.seed(42)

    def executor(circuit):
        return np.random.random()

    results = [executor(cirq_circuit) for _ in range(3)]
    extrapolation_method = extrapolation_factory.extrapolate
    two_stage_zne_res = combine_results(
        scale_factors, results, extrapolation_method
    )

    assert isinstance(two_stage_zne_res, float)

    np.random.seed(42)
    zne_res = execute_with_zne(
        cirq_circuit,
        executor,
        factory=extrapolation_factory(scale_factors),
        scale_noise=noise_scaling_method,
    )
    assert np.isclose(zne_res, two_stage_zne_res)


def test_default_scaling_option_two_stage_zne():
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        cirq.H.on_each(qreg),
        cirq.CNOT(*qreg),
        cirq.CNOT(*qreg),
        cirq.H.on_each(qreg),
    )

    scale_factors = [3, 5, 6, 8]

    circs_default_scaling_method = construct_circuits(
        cirq_circuit, scale_factors
    )

    for i in range(len(scale_factors)):
        assert len(circs_default_scaling_method[i]) > len(cirq_circuit)


def test_visualize_fits_returns_figure():
    scale_factors = [1.0, 2.0, 3.0, 4.0]
    exp_values = [0.5 + 0.7 * np.exp(-0.4 * x) for x in scale_factors]
    fig = visualize_fits(scale_factors, exp_values)
    assert isinstance(fig, Figure)
    # One scatter and four fits
    assert len(fig.axes[0].lines) == 5
    legend_labels = [text.get_text() for text in fig.axes[0].legend().texts]
    assert legend_labels == [
        "Linear",
        "Polynomial (order=2)",
        "Exponential",
        "Richardson",
    ]


def test_visualize_fits_default_exponential_threshold():
    """Tests that the exponential fit is added when three data points
    are provided."""
    scale_factors = [1.0, 2.0, 3.0]
    exp_values = [0.5 + 0.7 * np.exp(-0.4 * x) for x in scale_factors]

    fig = visualize_fits(scale_factors, exp_values)

    assert isinstance(fig, Figure)
    legend_labels = [text.get_text() for text in fig.axes[0].legend().texts]
    assert "Exponential" in legend_labels


def test_visualize_fits_with_ideal_value():
    scale_factors = [1.0, 2.0, 3.0, 4.0]
    exp_values = [0.5 + 0.7 * np.exp(-0.4 * x) for x in scale_factors]
    fig = visualize_fits(scale_factors, exp_values, ideal_value=0.2)
    assert isinstance(fig, Figure)
    # One scatter, one ideal marker, and four fits
    assert len(fig.axes[0].lines) == 6
    legend_labels = [text.get_text() for text in fig.axes[0].legend().texts]
    assert legend_labels == [
        "Ideal",
        "Linear",
        "Polynomial (order=2)",
        "Exponential",
        "Richardson",
    ]


def test_visualize_fits_bad_lengths():
    with pytest.raises(ValueError):
        visualize_fits([1.0, 2.0], [0.1])


def test_visualize_fits_with_failing_factory():
    class FailingFactory:
        _options = {}

        @staticmethod
        def extrapolate(*args, **kwargs):
            raise ExtrapolationError("oops")

    scale_factors = [1.0, 2.0, 3.0]
    exp_values = [0.1, 0.2, 0.3]
    factories = [FailingFactory(), LinearFactory(scale_factors)]

    with pytest.warns(UserWarning, match="FailingFactory"):
        fig = visualize_fits(scale_factors, exp_values, factories=factories)

    assert isinstance(fig, Figure)
    # One scatter and one successful fit
    assert len(fig.axes[0].lines) == 2
