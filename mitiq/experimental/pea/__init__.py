# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import warnings

warnings.warn(
    "mitiq.experimental.pea is experimental and its API may change without notice in "
    "future releases. It is not covered by mitiq's semantic versioning guarantees.",
    FutureWarning,
    stacklevel=2,
)

from mitiq.experimental.pea.pea import combine_results, construct_circuits, execute_with_pea
