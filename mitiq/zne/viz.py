# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Visualization utilities for zero-noise extrapolation."""

import warnings
from typing import Any, Callable, Optional, Protocol, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from mitiq.zne.inference import (
    ExpFactory,
    ExtrapolationError,
    ExtrapolationResult,
    LinearFactory,
    PolyFactory,
    RichardsonFactory,
)


class SupportsExtrapolate(Protocol):
    """Objects providing an ``extrapolate`` method."""

    @staticmethod
    def extrapolate(*args: Any, **kwargs: Any) -> ExtrapolationResult: ...


def visualize_fits(
    scale_factors: Sequence[float],
    exp_values: Sequence[float],
    *,
    factories: Optional[Sequence[SupportsExtrapolate]] = None,
    order: int = 2,
    ideal_value: Optional[float] = None,
) -> Figure:
    """Visualizes extrapolation fits together with data.

    The curves are evaluated slightly before zero and beyond the largest
    scale factor to provide a smoother look.

    Args:
        scale_factors: Noise scale factors at which expectation values were
            measured.
        exp_values: Expectation values corresponding to ``scale_factors``.
        factories: Sequence of objects implementing ``extrapolate``. If
            ``None``, fits are produced by :class:`.LinearFactory`,
            :class:`.PolyFactory`, :class:`.ExpFactory`, and
            :class:`.RichardsonFactory`.
        order: Order for the default :class:`.PolyFactory`.
            The polynomial fit requires at least ``order + 1`` data points,
            while the exponential fit is only performed when three or more
            points are provided.
        ideal_value: Optional expectation value for zero noise. When
            specified, a marker is drawn at ``(0.0, ideal_value)`` to
            highlight the target value.

    Returns:
        The matplotlib figure containing the data and fit curves.

    Raises:
        ValueError: If ``scale_factors`` and ``exp_values`` lengths do not
            match.
    """

    if len(scale_factors) != len(exp_values):
        raise ValueError(
            "`scale_factors` and `exp_values` must have the same length."
        )

    if factories is None:
        factories = [LinearFactory(scale_factors)]
        if len(scale_factors) >= order + 1:
            factories.append(PolyFactory(scale_factors, order))
        else:
            warnings.warn(
                "Skipping PolyFactory: requires at least"
                f" {order + 1} data points.",
                UserWarning,
            )
        if len(scale_factors) >= 3:
            factories.append(ExpFactory(scale_factors))
        else:
            warnings.warn(
                "Skipping ExpFactory: requires at least three data points.",
                UserWarning,
            )

        factories.append(RichardsonFactory(scale_factors))

    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.plot(
        scale_factors,
        exp_values,
        "o",
        markersize=10,
        markeredgecolor="black",
        alpha=0.8,
        label=None,
    )

    if ideal_value is not None:
        ax.plot(
            0.0,
            ideal_value,
            "D",
            markersize=8,
            color="red",
            label="Ideal",
        )

    # Use a resolution proportional to the number of data points
    # to better visualize fits when few scale factors are provided.
    n_points = max(1, 5 * len(scale_factors))
    max_factor = max(scale_factors)
    padding = 0.1 * max_factor
    smooth_scale_factors = np.linspace(
        -padding, max_factor + padding, n_points
    )

    for factory in factories:
        try:
            result = cast(
                tuple[
                    float,
                    Optional[float],
                    list[float],
                    npt.NDArray[Any] | None,
                    Callable[[float], float],
                ],
                factory.extrapolate(
                    scale_factors,
                    exp_values,
                    full_output=True,
                    **getattr(factory, "_options", {}),
                ),
            )
            curve = result[4]
        except (ExtrapolationError, ValueError) as exc:
            warnings.warn(
                f"{factory.__class__.__name__} failed: {exc}",
                UserWarning,
            )
            continue

        if isinstance(factory, LinearFactory):
            label = "Linear"
        elif isinstance(factory, PolyFactory):
            poly_order = getattr(factory, "_options", {}).get("order")
            order_label = (
                f" (order={poly_order})" if poly_order is not None else ""
            )
            label = f"Polynomial{order_label}"
        elif isinstance(factory, ExpFactory):
            label = "Exponential"
        elif isinstance(factory, RichardsonFactory):
            label = "Richardson"
        else:
            label = getattr(
                factory,
                "method_label",
                factory.__class__.__name__,
            )
        ax.plot(
            smooth_scale_factors,
            [curve(s) for s in smooth_scale_factors],
            "--",
            lw=2,
            label=label,
        )

    plt.xlim(-padding, max_factor + padding)
    ax.grid(True)
    plt.xlabel("Noise scale factor")
    plt.ylabel("Expectation value")
    plt.legend()
    return fig
