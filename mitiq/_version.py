# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Reads in version information."""

import os
from importlib.metadata import PackageNotFoundError, version

directory_of_this_file = os.path.dirname(os.path.abspath(__file__))

try:
    __version__ = version("mitiq")
except PackageNotFoundError:
    directory_of_this_file = os.path.dirname(os.path.abspath(__file__))
    with open(f"{directory_of_this_file}/../VERSION.txt", "r") as f:
        __version__ = f.read().strip()
