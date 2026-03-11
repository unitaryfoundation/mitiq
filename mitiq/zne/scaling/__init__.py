# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Methods for scaling noise in circuits by adding or modifying gates."""
from mitiq.zne.scaling.folding import (
    fold_all,
    fold_gates_at_random,
    fold_global,
)
from mitiq.zne.scaling.layer_scaling import (
    layer_folding,
    get_layer_folding,
)
from mitiq.zne.scaling.parameter import (
    scale_parameters,
    compute_parameter_variance,
)
from mitiq.zne.scaling.identity_insertion import insert_id_layers
