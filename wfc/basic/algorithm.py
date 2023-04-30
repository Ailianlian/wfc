import random

import numpy as np
from IPython import display
from numpy.typing import NDArray
from PIL import Image as mig
from PIL.Image import Image

from wfc.basic.compute import find_true, get_neighbors
from wfc.basic.definitions import Direction, Tile


def blend_many(ims: list) -> Image:
    """
    Blends a sequence of images.
    """
    current, *ims = ims
    for i, im in enumerate(ims):
        current = mig.blend(current, im, 1 / (i + 2))
    return current


def blend_tiles(choices: NDArray, tiles: list[Tile]) -> Image:
    """
    Given a list of states (True if ruled out, False if not) for each tile,
    and a list of tiles, return a blend of all the tiles that haven't been
    ruled out.
    """
    to_blend = [tiles[i].bitmap for i in range(len(choices)) if choices[i]]
    return blend_many(to_blend)


def show_state(potential: NDArray, tiles: list[Tile]) -> Image:
    """
    Given a list of states for each tile for each position of the image, return
    an image representing the state of the global image.
    """
    rows = []
    for row in potential:
        rows.append([np.asarray(blend_tiles(t, tiles)) for t in row])

    np_rows = np.array(rows)
    n_rows, n_cols, tile_height, tile_width, _ = np_rows.shape
    images = np.swapaxes(rows, 1, 2)
    return mig.fromarray(images.reshape(n_rows * tile_height, n_cols * tile_width, 4))


def add_constraint(
    potential: NDArray,
    location: tuple[int, int],
    incoming_direction: Direction,
    possible_tiles: list[Tile],
    base_tiles: list[Tile],
) -> bool:
    neighbor_constraint = {t.sides[incoming_direction.value] for t in possible_tiles}
    outgoing_direction = incoming_direction.reverse()
    changed = False
    for i_p, p in enumerate(potential[location]):
        if not p:
            continue
        if base_tiles[i_p].sides[outgoing_direction.value] not in neighbor_constraint:
            potential[location][i_p] = False
            changed = True
    if not np.any(potential[location]):
        raise Exception(f"No patterns left at {location}")
    return changed


def propagate(
    potential: NDArray, start_location: tuple[int, int], base_tiles: list[Tile]
) -> None:
    height, width = potential.shape[:2]
    needs_update = np.full((height, width), False)
    needs_update[start_location] = True
    while np.any(needs_update):
        needs_update_next = np.full((height, width), False)
        locations = find_true(needs_update)
        for location in locations:
            possible_tiles = [base_tiles[n] for n in find_true(potential[location])]
            for neighbor in get_neighbors(location, height, width):
                neighbor_direction, neighbor_x, neighbor_y = neighbor
                neighbor_location = (neighbor_x, neighbor_y)
                was_updated = add_constraint(
                    potential=potential,
                    location=neighbor_location,
                    incoming_direction=neighbor_direction,
                    possible_tiles=possible_tiles,
                    base_tiles=base_tiles,
                )
                needs_update_next[location] |= was_updated
        needs_update = needs_update_next


def location_with_fewest_choices(potential: NDArray) -> tuple[int, int] | None:
    num_choices = np.sum(potential, axis=2, dtype="float32")
    num_choices[num_choices == 1] = np.inf
    candidate_locations = find_true(num_choices == num_choices.min())
    location = random.choice(candidate_locations)
    if num_choices[location] == np.inf:
        return None
    return location


def run_iteration(old_potential:NDArray, tiles_weights:NDArray, base_tiles:list[Tile])->NDArray:
    potential = old_potential.copy()
    to_collapse = location_with_fewest_choices(potential)  # 3
    if to_collapse is None:  # 1
        raise StopIteration()
    elif not np.any(potential[to_collapse]):  # 2
        raise Exception(f"No choices left at {to_collapse}")
    else:  # 4 ↓
        nonzero = find_true(potential[to_collapse])
        tile_probs = tiles_weights[nonzero] / sum(tiles_weights[nonzero])
        selected_tile = np.random.choice(nonzero, p=tile_probs)
        potential[to_collapse] = False
        potential[to_collapse][selected_tile] = True
        propagate(potential, to_collapse, base_tiles=base_tiles)  # 5
    return potential


def generate_wfc(height:int, width:int, base_tiles:list[Tile], fast_generation:bool)->list[Image]:
    p = np.full((height, width, len(base_tiles)), True)
    weights = np.asarray([t.weight for t in base_tiles])
    images = [show_state(p, base_tiles)]
    while True:
        try:
            p = run_iteration(p, tiles_weights=weights, base_tiles=base_tiles)
            if not fast_generation:
                images.append(show_state(p, base_tiles))  # Move me for speed
        except StopIteration as e:
            break
        except Exception as e:
            print(e)
            break
    if fast_generation:
        images = [show_state(p, base_tiles)]
    return images
