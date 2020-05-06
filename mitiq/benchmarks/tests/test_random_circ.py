from itertools import product
import pytest

import numpy as np

from mitiq.factories import LinearFactory, RichardsonFactory, PolyFactory
from mitiq.folding import fold_gates_at_random, fold_gates_from_left, \
    fold_gates_from_right
from mitiq.benchmarks.random_circ import rand_benchmark_zne

SCALE_FUNCTIONS = [
    fold_gates_at_random,
    fold_gates_from_left,
    fold_gates_from_right
]

FACTORIES = [
    RichardsonFactory([1.0, 1.4, 2.1]),
    LinearFactory([1.0, 1.6]),
    PolyFactory([1.0, 1.4, 2.1], order=2)
]

# Set the seed for testing
np.random.seed(808)


@pytest.mark.parametrize(["scale_noise", "fac"],
                         product(SCALE_FUNCTIONS, FACTORIES))
def test_random_benchmarks(scale_noise, fac):
    exact, unmitigated, mitigated = rand_benchmark_zne(n_qubits=2,
                                                       depth=20,
                                                       trials=8,
                                                       op_density=0.99,
                                                       noise=0.003,
                                                       fac=fac,
                                                       scale_noise=scale_noise)

    unmit_err = np.abs(exact - unmitigated)
    mit_err = np.abs(exact - mitigated)

    assert np.average(unmit_err) >= np.average(mit_err)
