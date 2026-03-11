# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for mitiq.about()."""

import mitiq


def test_result_and_stdout(capsys):
    mitiq.about()
    captured = capsys.readouterr()
    assert captured.out.startswith(
        "\nMitiq: A Python toolkit for implementing"
    )
