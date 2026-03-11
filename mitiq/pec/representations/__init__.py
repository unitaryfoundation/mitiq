# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


from mitiq.pec.representations.depolarizing import (
    represent_operation_with_global_depolarizing_noise,
    represent_operation_with_local_depolarizing_noise,
    represent_operations_in_circuit_with_global_depolarizing_noise,
    represent_operations_in_circuit_with_local_depolarizing_noise,
    global_depolarizing_kraus,
    local_depolarizing_kraus,
)

from mitiq.pec.representations.damping import (
    _represent_operation_with_amplitude_damping_noise,
    amplitude_damping_kraus,
)

from mitiq.pec.representations.optimal import (
    minimize_one_norm,
    find_optimal_representation,
)

from mitiq.pec.representations.biased_noise import (
    represent_operation_with_local_biased_noise,
)

from mitiq.pec.representations.learning import (
    depolarizing_noise_loss_function,
    biased_noise_loss_function,
    learn_depolarizing_noise_parameter,
    learn_biased_noise_parameters,
)
