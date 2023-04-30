from __future__ import annotations

from collections import namedtuple
from enum import Enum, auto

Tile = namedtuple("Tile", ("name", "bitmap", "sides", "weight"))


class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def reverse(self) -> Direction:
        return {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
        }[self]
