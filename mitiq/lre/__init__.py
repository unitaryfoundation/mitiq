# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Methods for scaling noise in circuits by layers and using multivariate extrapolation."""

from mitiq.lre.multivariate_scaling.layerwise_folding import multivariate_layer_scaling

from mitiq.lre.inference.multivariate_richardson import (
    multivariate_richardson_coefficients,
    sample_matrix,
)

from mitiq.lre.lre import execute_with_lre, mitigate_executor, lre_decorator