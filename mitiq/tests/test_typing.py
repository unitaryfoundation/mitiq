# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import os

from mitiq.typing import SUPPORTED_PROGRAM_TYPES


def test_supported_program_types_definition():
    directory_of_this_file = os.path.dirname(os.path.abspath(__file__))
    with open(f"{directory_of_this_file}/../../INTEGRATIONS.txt", "r") as file:
        integrations_from_setup = file.read().splitlines()

    assert list(SUPPORTED_PROGRAM_TYPES.keys()) == integrations_from_setup
