# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Methods for benchmarking zero-noise extrapolation on MaxCut-QAOA."""
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from scipy.optimize import minimize

from cirq import (
    Circuit,
    DensityMatrixSimulator,
    identity_each,
    NamedQubit,
    X,
    ZZ,
    H,
)
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import Factory
from mitiq.benchmarks.utils import noisy_simulation


SIMULATOR = DensityMatrixSimulator()


def _make_noisy_backend(
    noise: float, obs: Union[np.ndarray, float]
) -> Callable[[Circuit], float]:
    """Returns a (noisy) execute function for Cirq circuits.

    Args:
        noise: The level of depolarizing noise.
        obs: The observable that the backend should measure.
    """

    def noisy_backend(circ: Circuit) -> float:
        return noisy_simulation(circ, noise, cast(np.ndarray, obs))

    return noisy_backend


def make_maxcut(
    graph: List[Tuple[int, int]],
    noise: float = 0,
    scale_noise: Optional[Callable[[Circuit, float], Circuit]] = None,
    factory: Optional[Factory] = None,
) -> Tuple[
    Callable[[np.ndarray], Optional[float]],
    Callable[[np.ndarray], Circuit],
    float,
]:
    """Makes an executor that evaluates the QAOA ansatz at a given beta
    and gamma parameters.

    Args:
        graph: The MAXCUT graph as a list of edges with integer labelled nodes.
        noise: The level of depolarizing noise.
        scale_noise: The noise scaling method for ZNE.
        factory: The factory to use for ZNE.

    Returns:
        (ansatz_eval, ansatz_maker, cost_obs) as a triple where

        * **ansatz_eval** -- function that evalutes the maxcut ansatz on the
            noisy cirq backend.
        * **ansatz_maker** -- function that returns an ansatz circuit.
        * **cost_obs** -- the cost observable as a dense matrix.

    """
    # get the list of unique nodes from the list of edges
    nodes = list({node for edge in graph for node in edge})
    nodes = list(range(max(nodes) + 1))

    # one qubit per node
    qreg = [NamedQubit(str(nn)) for nn in nodes]

    def cost_step(beta: float) -> Circuit:
        return Circuit(ZZ(qreg[u], qreg[v]) ** beta for u, v in graph)

    def mix_step(gamma: float) -> Circuit:
        return Circuit(X(qq) ** gamma for qq in qreg)

    def qaoa_ansatz(params: np.ndarray) -> Circuit:
        half = int(len(params) / 2)
        betas, gammas = params[:half], params[half:]
        qaoa_steps = sum(
            [
                cost_step(beta) + mix_step(gamma)
                for beta, gamma in zip(betas, gammas)
            ],
            Circuit(),
        )
        return Circuit(H.on_each(qreg)) + qaoa_steps

    # make the cost observable
    identity_matrix = np.eye(2 ** len(nodes))
    cost_mat = -0.5 * sum(
        identity_matrix
        - Circuit([identity_each(*qreg), ZZ(qreg[i], qreg[j])]).unitary()
        for i, j in graph
    )
    noisy_backend = _make_noisy_backend(noise, cost_mat)

    # must have this function signature to work with scipy minimize
    def qaoa_cost(params: np.ndarray) -> Optional[float]:
        qaoa_prog = qaoa_ansatz(params)
        if scale_noise is None and factory is None:
            return noisy_backend(qaoa_prog)
        else:
            assert scale_noise is not None
            return execute_with_zne(
                qaoa_prog,
                executor=noisy_backend,  # type: ignore
                scale_noise=scale_noise,  # type: ignore
                factory=factory,
            )

    return qaoa_cost, qaoa_ansatz, cost_mat


def run_maxcut(
    graph: List[Tuple[int, int]],
    x0: Iterable[float],
    noise: float = 0,
    scale_noise: Optional[Callable[[Circuit, float], Circuit]] = None,
    factory: Optional[Factory] = None,
    optimizer: str = "Nelder-Mead",
    verbose: bool = False,
) -> Tuple[float, np.ndarray, List[Optional[float]]]:
    """Optimizes MaxCut cost function on the given graph using QAOA.

    Args:
        graph: The MAXCUT graph as a list of edges with integer labelled nodes.
        x0: The initial parameters for QAOA [betas, gammas]. The size of x0
            determines the number of steps.
        noise: Depolarizing noise strength.
        scale_noise: The noise scaling method for ZNE.
        factory: The factory to use for ZNE.
        optimizer: Scipy optimization method to use.
        verbose: An option to pass to scipy.minimize.

    Returns:
        A triple of the minimum cost, the values of beta and gamma that
        obtained that cost, and a list of costs at each iteration step.

    Example:
        Run MAXCUT with 2 steps such that betas = [1.0, 1.1] and
        gammas = [1.4, 0.7] on a graph with four edges and four nodes.

        >>> from mitiq.benchmarks.maxcut import run_maxcut
        >>> graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> x0 = [1.0, 1.1, 1.4, 0.7]
        >>> fun, x, traj = run_maxcut(graph, x0, verbose=True)
        Optimization terminated successfully.
                 Current function value: -4.000000
                 Iterations: 108
                 Function evaluations: 188
    """
    qaoa_cost, *_ = make_maxcut(graph, noise, scale_noise, factory)

    # store the optimization trajectories
    traj = []

    def callback(xk: np.ndarray) -> bool:
        traj.append(qaoa_cost(xk))
        return True

    res = minimize(
        qaoa_cost,
        x0=np.array(x0),
        method=optimizer,
        callback=callback,
        options={"disp": verbose},
    )

    return res.fun, res.x, traj
