# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for layer-wise unitary folding on supported circuits."""

from collections.abc import Callable
from copy import deepcopy

import cirq
import numpy as np
from cirq import Moment, inverse

from mitiq import QPROGRAM
from mitiq.interface import accept_qprogram_and_validate
from mitiq.utils import _append_measurements, _pop_measurements
from mitiq.zne.scaling.folding import _check_foldable


@accept_qprogram_and_validate
def layer_folding(
    circuit: cirq.Circuit, layers_to_fold: list[int]
) -> cirq.Circuit:
    """Applies a variable amount of folding to select layers of a circuit.

    Note that this method only works for the univariate extrapolation methods.
    It allows a user to choose which layers in the input circuit will be
    scaled.

    .. seealso::

        If you would prefer to
        use a multivariate extrapolation method for unitary
        folding, use
        :func:`mitiq.lre.multivariate_scaling.layerwise_folding` instead.

        The layerwise folding required for multivariate extrapolation is
        different as the layers in the input circuit have to be scaled in
        a specific pattern. The required specific pattern for multivariate
        extrapolation does not allow a user to provide a choice of which
        layers to fold.

    Args:
        circuit: The input circuit.
        layers_to_fold: A list with the index referring to the layer number,
                        and the element filled by an integer representing the
                        number of times the layer is folded.

    Returns:
        The folded circuit.
    """
    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)
    layers = []

    for i, layer in enumerate(folded):
        layers.append(layer)
        # Apply the requisite number of folds to each layer.
        num_fold = layers_to_fold[i]
        for _ in range(num_fold):
            # We only fold the layer if it does not contain a measurement.
            if not cirq.is_measurement(layer):
                layers.append(Moment(inverse(layer)))
                layers.append(Moment(layer))

    # We combine each layer into a single circuit.
    combined_circuit = cirq.Circuit()
    for layer in layers:
        combined_circuit.append(layer)

    _append_measurements(combined_circuit, measurements)
    return combined_circuit


def get_layer_folding(
    layer_index: int,
) -> Callable[[QPROGRAM, float], QPROGRAM]:
    """Return function to perform folding. The function return can be used as
    an argument to define the noise scaling within the `execute_with_zne`
    function.

    Args:
        layer_index: The layer of the circuit to apply folding to.

    Returns:
        The function for folding the ith layer.
    """

    @accept_qprogram_and_validate
    def fold_ith_layer(
        circuit: cirq.Circuit, scale_factor: int
    ) -> cirq.Circuit:
        """Returns a circuit folded according to integer scale factors.

        Args:
            circuit: Circuit to fold.
            scale_factor: Factor to scale the circuit by.

        Returns:
            The folded quantum circuit.
        """
        _check_foldable(circuit)

        layers = [0] * len(circuit)
        num_folds = (scale_factor - 1) // 2
        if np.isclose(num_folds, int(num_folds)):
            num_folds = int(num_folds)
        layers[layer_index] = num_folds

        folded = layer_folding(circuit, layers_to_fold=layers)

        return folded

    return fold_ith_layer
