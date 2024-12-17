import random
import  numpy as np
from typing import Iterable

def get_object_pixels(object_size, top_left) -> List[Tuple[int, int]]:
    """
    Get the pixels occupied by a `object_size` by `object_size` square object
      whose top left most pixel is `top_left`
    
    Args:
        object_size - the side length of the square area that is the maximum
            area the object can occupy
        top_left - the top left pixel of the square to which
            `object_size` refers.
    
    Return:
        A list of tuples representing the coordinates of pixels that could
            potentially be taken up by an object.
    """
    object_pixels = []
    for i in range(object_size):
        for j in range(object_size):
            object_pixels.append((top_left[0] + i, top_left[1] + j))
    return object_pixels

def get_object_top_lefts(
    grid_size = 28,
    object_size = 4,
    number_of_objects = 5,
    partial_objects = True,
    overlapping_objects = False,
) -> List[Tuple[int, int]]:
    """
    Get the position of the top left pixel for `number_of_objects` objects
      on a square grid, checking that none will overlap

    Args:
        grid_size - how big will the visible space be?
        object_size - how big will objects in the visible space be?
        number_of_objects - how many objects will be in the visible space?
        partial_objects - can some objects be partially outside the
          visible space?
        overlapping_objects - can objects overlap each other?
    
    Return:
        A list of tuples representing the coordinates of the top left corner
            of the square containing each object.
    """

    filled_squares = [] # to check for collisions
    object_top_lefts = []

    if partial_objects:
        staging_grid_size = grid_size + 2 * object_size - 2
    else:
        staging_grid_size = grid_size
    
    # Assert no more than 1/2 of available area will be filled with objects:
    assert not number_of_objects * object_size**2 > staging_grid_size**2 / 2
    
    for o in range(number_of_objects):
        proposed_position = (
            random.randint(0, staging_grid_size - object_size),
            random.randint(0, staging_grid_size - object_size)
        )
        object_pixels = get_object_pixels(object_size, proposed_position)

        # If not `overlapping_objects` choose new point if there's a collision
        if not overlapping_objects:
            while any([o in filled_squares for o in object_pixels]):
                proposed_position = (
                    random.randrange(0, staging_grid_size - object_size),
                    random.randrange(0, staging_grid_size - object_size)
                )
                object_pixels = get_object_pixels(
                    object_size,
                    proposed_position
                )
        
        # Append satisfactory coordinate tuple to `object_top_lefts`
        object_top_lefts.append(
            (
                proposed_position[0] - (staging_grid_size - grid_size) // 2,
                proposed_position[1] - (staging_grid_size - grid_size) // 2
            )
        )
        filled_squares += object_pixels
    
    return object_top_lefts

def random_object_template(object_size: int) -> List[List[int]]:
    """
    Generate a random template for an object in the environment, given an
      object size.
    
    Args:
      object_size - the side length of the square area that is the maximum
        area the object can occupy
    
    Return:
      A list of lists representing a square grid, with 1s where the object
        will be filled and 0 where it will be transparent.
    """
    return [
        [0 if random.randint(0, 1) else 1 for _ in range(object_size)]
        for _ in range(object_size)
    ]

def random_colour(colour_vocabulary = None):
    if colour_vocabulary is not None:
        return random.choice(colour_vocabulary)
    else:
        return tuple(random.randint(0, 255) for _ in range(3))

def random_object(object_size: int = None, object_template_vocabulary = None, colour_vocabulary = None):
    assert object_size is not None or object_template_vocabulary is not None
    colour = random_colour(colour_vocabulary)
    if object_template_vocabulary is None:
        template = random_object_template(object_size)
    else:
        template = random.choice(object_template_vocabulary)
    return [[colour if j else (0, 0, 0) for j in i] for i in template]

def add_object(
    grid,
    top_left,
    object_to_add: List[List[Tuple[int, int, int]]]
):
    """
    Add an object to a grid
    """
    object_pixels = get_object_pixels(len(object_to_add), top_left)
    flattened = sum(object_to_add, [])
    for pixel, content in zip(object_pixels, flattened):
        x, y = pixel
        if x < 0 or y < 0 or x >= len(grid) or y >= len(grid):
            continue
        else:
            grid[y][x] = content
    return grid

def generate_view(
    grid_size,
    objects: Iterable[List[List[Tuple[int, int, int]]]],
    partial_objects = True
) -> List[Tuple[int, int]]:
    grid = [[(0, 0, 0)] * grid_size for _ in range(grid_size)]
    object_top_lefts = get_object_top_lefts(
        grid_size = grid_size,
        object_size = len(objects[0]),
        number_of_objects = len(objects),
        partial_objects = partial_objects
    )
    for tl, obj in zip(object_top_lefts, objects):
        grid = add_object(grid, tl, obj)
    return np.asarray(grid, dtype=np.uint8)