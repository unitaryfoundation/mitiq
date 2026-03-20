# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import warnings

warnings.warn(
    "mitiq.experimental.vd is experimental and its API may change without notice in "
    "future releases. It is not covered by mitiq's semantic versioning guarantees.",
    FutureWarning,
    stacklevel=2,
)

from mitiq.experimental.vd.vd import execute_with_vd, construct_circuits, combine_results
