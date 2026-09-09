# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for allocating a ZNE shot budget and executing the allocation."""

from copy import deepcopy

import cirq
import numpy as np
import pytest

from mitiq import Executor, MeasurementResult, Observable, PauliString
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import (
    ExpFactory,
    FakeNodesFactory,
    LinearFactory,
    PolyExpFactory,
    PolyFactory,
    RichardsonFactory,
)


@pytest.mark.parametrize(
    "factory,coefficients",
    [
        (RichardsonFactory([1, 2, 3]), [3, -3, 1]),
        (PolyFactory([1, 2, 3], order=2), [3, -3, 1]),
        (LinearFactory([1, 2, 3]), [4 / 3, 1 / 3, -2 / 3]),
        (PolyFactory([1, 2, 3], order=0), [1 / 3] * 3),
        # The fake nodes are 2-sqrt(2), 2, 2+sqrt(2).
        (
            FakeNodesFactory([1, 2, 3]),
            [1 + 1 / np.sqrt(2), -1, 1 - 1 / np.sqrt(2)],
        ),
    ],
)
def test_shot_allocation_matches_linear_estimator(factory, coefficients):
    budget = 30_000
    shots = factory.get_optimal_shot_list(budget)
    coefficients = np.asarray(coefficients)
    fractions = np.abs(coefficients) / np.sum(np.abs(coefficients))

    np.testing.assert_allclose(np.array(shots) / budget, fractions, atol=1e-4)
    allocated_variance = np.sum(coefficients**2 / shots)
    uniform_variance = np.sum(coefficients**2) / (budget / len(shots))
    assert allocated_variance <= uniform_variance
    if not np.allclose(fractions, fractions[0]):
        assert allocated_variance < uniform_variance


@pytest.mark.parametrize("budget", [3, 4, 7, 703, np.int64(10_001), 10**100])
def test_shot_allocation_uses_exact_budget(budget):
    factory = RichardsonFactory([1, 2, 3])
    shots = factory.get_optimal_shot_list(budget)

    assert len(shots) == 3
    assert all(type(shot) is int and shot >= 1 for shot in shots)
    assert sum(shots) == budget
    assert shots == factory.get_optimal_shot_list(budget)
    if budget == 3:
        assert shots == [1, 1, 1]
    elif budget == 703:
        assert shots == [301, 301, 101]


def test_shot_allocation_rounding_equal_weights():
    shots = PolyFactory([1, 2, 3], order=0).get_optimal_shot_list(7)
    assert sorted(shots) == [2, 2, 3]


@pytest.mark.parametrize("budget,expected", [(4, [3, 1]), (8, [6, 2])])
def test_shot_allocation_preserves_feasible_optimum(budget, expected):
    # Coefficients [3/2, -1/2] give an already feasible 3:1 allocation.
    assert LinearFactory([1, 3]).get_optimal_shot_list(budget) == expected


def test_shot_allocation_keeps_zero_coefficient_point_executable():
    # Linear extrapolation has coefficients [1, 1/2, 0, -1/2].
    shots = LinearFactory([1, 2, 3, 4]).get_optimal_shot_list(4_000)
    assert shots[2] == 1
    assert sum(shots) == 4_000


def test_shot_allocation_preserves_scale_factor_order():
    assert RichardsonFactory([3, 1, 2]).get_optimal_shot_list(703) == [
        101,
        301,
        301,
    ]


@pytest.mark.parametrize(
    "coefficients,budget,expected",
    [
        # Fixing the last quota at one makes the middle quota fall below one.
        ([10.0, 2.1, 0.1], 6, [4, 1, 1]),
        ([0.0, 1.0, 0.0], 100, [1, 98, 1]),
        # Exact quotas must remain defined even for subnormal coefficients.
        ([5e-324, 1e-323, 0.0], 10, [3, 6, 1]),
    ],
)
def test_shot_allocation_redistributes_constrained_quotas(
    monkeypatch, coefficients, budget, expected
):
    monkeypatch.setattr(
        LinearFactory,
        "extrapolate",
        staticmethod(lambda scales, values: np.dot(coefficients, values)),
    )
    assert LinearFactory([1, 2, 3]).get_optimal_shot_list(budget) == expected


@pytest.mark.parametrize(
    "kind", ["linear", "polynomial", "richardson", "fake"]
)
def test_shot_allocation_seeded_independent_coefficient_oracle(kind):
    rng = np.random.default_rng(1709)
    for _ in range(25):
        size = int(rng.integers(2, 8))
        scales = 1 + np.cumsum(rng.uniform(0.5, 2.0, size))
        if kind == "fake":
            scales = np.linspace(scales[0], scales[-1], size)
            factory = FakeNodesFactory(scales)
            endpoint = scales[0] + scales[-1]
            nodes = endpoint * (1 - np.cos(np.pi * scales / endpoint)) / 2
        else:
            rng.shuffle(scales)
            nodes = scales
            if kind == "richardson":
                factory = RichardsonFactory(scales)
            elif kind == "linear":
                factory = LinearFactory(scales)
            else:
                order = int(rng.integers(0, min(3, size - 1) + 1))
                factory = PolyFactory(scales, order=order)

        if kind in ("richardson", "fake"):
            # Lagrange interpolation coefficients evaluated at zero.
            coefficients = np.array(
                [
                    np.prod(
                        np.delete(nodes, index)
                        / (np.delete(nodes, index) - node)
                    )
                    for index, node in enumerate(nodes)
                ]
            )
        else:
            degree = 1 if kind == "linear" else order
            design = np.vander(
                scales / max(scales), degree + 1, increasing=True
            )
            coefficients = np.linalg.pinv(design)[0]

        weights = np.abs(coefficients) / max(np.abs(coefficients))
        for budget in (size, size + 1, size + 3, 1001, 10_003):
            shots = factory.get_optimal_shot_list(budget)
            assert len(shots) == size
            assert all(type(shot) is int and shot >= 1 for shot in shots)
            assert sum(shots) == budget

            # Independent continuous oracle: solve sum(max(1, c*w)) = budget.
            low, high = 0.0, float(budget)
            for _ in range(80):
                midpoint = (low + high) / 2
                if np.maximum(1, midpoint * weights).sum() < budget:
                    low = midpoint
                else:
                    high = midpoint
            quotas = np.maximum(1, ((low + high) / 2) * weights)
            assert np.all(np.abs(np.array(shots) - quotas) < 1 + 1e-5)


@pytest.mark.parametrize(
    "budget", [True, False, np.bool_(True), 3.0, "3", None]
)
def test_shot_allocation_rejects_noninteger_budgets(budget):
    with pytest.raises(TypeError):
        RichardsonFactory([1, 2, 3]).get_optimal_shot_list(budget)


@pytest.mark.parametrize("budget", [-10, 0, 1, 2])
def test_shot_allocation_rejects_insufficient_budget(budget):
    with pytest.raises(ValueError):
        RichardsonFactory([1, 2, 3]).get_optimal_shot_list(budget)


@pytest.mark.parametrize(
    "factory",
    [ExpFactory([1, 2, 3]), PolyExpFactory([1, 2, 3], order=1)],
)
def test_shot_allocation_rejects_nonlinear_factories(factory):
    with pytest.raises(NotImplementedError):
        factory.get_optimal_shot_list(100)


def test_shot_allocation_rejects_custom_extrapolation():
    class NonlinearFactory(LinearFactory):
        @staticmethod
        def extrapolate(scale_factors, exp_values, full_output=False):
            return exp_values[0] ** 2

    with pytest.raises(NotImplementedError):
        NonlinearFactory([1, 2]).get_optimal_shot_list(100)


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), 0.0])
def test_shot_allocation_rejects_invalid_coefficients(
    monkeypatch, coefficient
):
    monkeypatch.setattr(
        LinearFactory,
        "extrapolate",
        staticmethod(lambda scales, values: coefficient),
    )
    with pytest.raises(ValueError, match="coefficients"):
        LinearFactory([1, 2]).get_optimal_shot_list(100)


@pytest.mark.parametrize("state", ["fresh", "run", "reduced", "pushed"])
def test_shot_allocation_does_not_mutate_factory(state):
    factory = LinearFactory([1, 2, 3], shot_list=[10, 20, 30])
    expected = factory.get_optimal_shot_list(703)
    if state in ("run", "reduced"):
        factory.run_classical(lambda scale, shots: 0.1 * scale**2)
    if state == "reduced":
        factory.reduce()
    if state == "pushed":
        factory.push({"scale_factor": 10, "shots": 50}, 0.25)
    before = deepcopy(factory.__dict__)

    assert factory.get_optimal_shot_list(703) == expected
    assert factory.__dict__.keys() == before.keys()
    for key, value in before.items():
        np.testing.assert_equal(factory.__dict__[key], value)


def test_shot_allocation_runs_classical_and_reuses_factory():
    scales = [1, 2, 3]
    shots = RichardsonFactory(scales).get_optimal_shot_list(703)
    factory = RichardsonFactory(scales, shot_list=shots)
    calls = []

    def executor(scale, shots):
        calls.append((scale, shots))
        return 0.5 + 0.1 * scale

    for _ in range(2):
        assert factory.run_classical(executor).reduce() == pytest.approx(0.5)
    assert calls == list(zip(scales, shots)) * 2


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("repetitions", [1, 3])
@pytest.mark.parametrize("budget", [3, 703])
def test_shot_allocation_reaches_executor(batched, repetitions, budget):
    shots = RichardsonFactory([1, 2, 3]).get_optimal_shot_list(budget)
    factory = RichardsonFactory([1, 2, 3], shot_list=shots)
    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    recorded_shots = []

    def serial_executor(circuit, shots) -> float:
        recorded_shots.append(shots)
        return 0.5

    def batch_executor(circuits, shots) -> list[float]:
        return [serial_executor(circuit, shots) for circuit in circuits]

    executor = Executor(batch_executor if batched else serial_executor)
    result = execute_with_zne(
        circuit,
        executor,
        factory=factory,
        # Identical circuits must still consume independent shot allocations.
        scale_noise=lambda circuit, scale: circuit,
        num_to_average=repetitions,
    )

    assert result == pytest.approx(0.5)
    assert recorded_shots == [
        shot for shot in shots for _ in range(repetitions)
    ]
    assert sum(recorded_shots) == budget * repetitions
    assert len(executor.executed_circuits) == 3 * repetitions


def test_shot_allocation_budget_is_per_measurement_group():
    shots = LinearFactory([1, 2]).get_optimal_shot_list(30)
    factory = LinearFactory([1, 2], shot_list=shots)
    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    observable = Observable(PauliString("X"), PauliString("Z"))
    recorded_shots = []

    def executor(circuit, shots) -> MeasurementResult:
        recorded_shots.append(shots)
        return MeasurementResult([[0]] * shots, qubit_indices=(0,))

    result = execute_with_zne(
        circuit,
        executor,
        observable,
        factory=factory,
        scale_noise=lambda circuit, scale: circuit,
    )

    assert result == pytest.approx(2.0)
    assert observable.ngroups == 2
    assert recorded_shots == [shot for shot in shots for _ in range(2)]
    assert sum(recorded_shots) == 30 * observable.ngroups
