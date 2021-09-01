# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Defines MeasurementResult, a result obtained by measuring qubits on a
quantum computer."""
from dataclasses import dataclass
from typing import cast, Iterable, List

import numpy as np


Bitstring = List[int]


@dataclass
class MeasurementResult:
    result: List[Bitstring]

    def __post_init__(self) -> None:
        self._bitstrings = np.array(self.result)
        if isinstance(self.result, np.ndarray):
            self.result = cast(List[Bitstring], self.result.tolist())

    @property
    def shots(self) -> int:
        return self._bitstrings.shape[0]

    @property
    def nqubits(self) -> int:
        return (
            self._bitstrings.shape[1]
            if len(self._bitstrings.shape) >= 2
            else 0
        )

    @property
    def asarray(self) -> np.ndarray:
        return self._bitstrings

    def __getitem__(self, indices: List[int]) -> np.ndarray:
        return self._bitstrings[:, indices]

    def __iter__(self) -> Iterable[Bitstring]:
        yield from self.result
