import numpy as np
from numpy.typing import NDArray

from wfc.basic.definitions import Direction


def find_true(array: NDArray) -> list:
    """
    Like np.nonzero, except it makes sense.
    """
    transform = int if len(np.asarray(array).shape) == 1 else tuple
    return list(map(transform, np.transpose(np.nonzero(array))))


def get_neighbors(
    location: tuple[int, int], height: int, width: int
) -> list[tuple[Direction, int, int]]:
    res = []
    x, y = location
    if x != 0:
        res.append((Direction.UP, x - 1, y))
    if y != 0:
        res.append((Direction.LEFT, x, y - 1))
    if x < height - 1:
        res.append((Direction.DOWN, x + 1, y))
    if y < width - 1:
        res.append((Direction.RIGHT, x, y + 1))
    return res
