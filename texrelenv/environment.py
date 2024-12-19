import copy
import random
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .colour import get_kelly_colours
from .exceptions import NoSpace, TooManyColours


class ThingTemplate:
    """
    A random template for an object in the environment, based on an
      object size.

    Args:
      size - the side length of the square area that is the maximum
          area the object can occupy

    Attributes:
        pattern (List[List[int]]): A list of lists representing a square
            grid, with 1s where the object will be filled and 0 where
            it will be transparent.
    """

    def __init__(self, size):
        self.size = size
        self.pattern = [
            [0 if random.randint(0, 1) else 1 for _ in range(size)] for _ in range(size)
        ]

    def hash(self):
        return hash(str(self.pattern))


class Thing:
    def __init__(self, colour: Tuple[int, int, int], template: ThingTemplate) -> None:
        self.size = template.size
        self.body = [[colour if j else (0, 0, 0) for j in i] for i in template.pattern]


class ThingMaker:
    def __init__(self, size=4, distinct_shapes=9, distinct_colours=9, fix_colour=False):
        if distinct_colours > 22:
            raise TooManyColours("Up to 22 colours supported at this time.")

        self.fix_colour = fix_colour

        self.colours = random.choices(get_kelly_colours(), k=distinct_colours)

        self.templates = []
        while len(self.templates) < distinct_shapes:
            proposed_template = ThingTemplate(size)
            if proposed_template.hash() not in [t.hash() for t in self.templates]:
                self.templates.append(proposed_template)

        if fix_colour:
            self.cache = defaultdict(self.random_unused_colour)
        else:
            self.cache = defaultdict(self.random_colour)

    def random_colour(self):
        return random.choice(self.colours)

    def random_unused_colour(self):
        return random.choice([c for c in self.colours if c not in self.cache.values()])

    def thing(self) -> List[List[Tuple[int, int, int]]]:
        template = random.choice(self.templates)
        if self.fix_colour:
            # get the colour we chose for the template last time
            colour = self.cache[template.hash()]
        else:
            colour = random.choice(self.colours)
        return Thing(colour, template)


class Grid:
    def __init__(self, size=16, objects_can_overlap: bool = False) -> None:
        self.size = size
        self.objects_can_overlap = objects_can_overlap
        self.state = [[(0, 0, 0) for _ in range(size)] for _ in range(size)]

    def find_spaces(
        self, thing: Thing, state: List[List[Tuple[int, int, int]]]
    ) -> List[Tuple[int, int]]:
        """
        Find empty spaces for a square object to be added to the grid, where
            it will not overlap another object. Return the answer as a list of
            tuples giving the row and column coordinates of the top left pixel
            of each found square space
        """
        filled_squares = np.array(
            [[0 if c == (0, 0, 0) else 1 for c in r] for r in state]
        )
        # print('filled_squares:')
        # print(filled_squares)
        space_shape = (thing.size, thing.size)
        # print('space_shape:')
        # print(space_shape)
        space = np.zeros((thing.size, thing.size))
        # print('space:')
        # print(space)
        candidates = sliding_window_view(filled_squares, space_shape)
        # print('candidates:')
        # print(candidates)
        matches = np.all(candidates == space, axis=(2, 3))
        # print('matches:')
        # print(matches)
        coords = [tuple(coords) for coords in np.argwhere(matches).tolist()]
        # print('coords:')
        # print(coords)
        return coords

    def add_object(
        self, thing: Thing, top_left_pixel: Optional[Tuple[int, int]]
    ) -> None:
        self.state = self._functional_add_object(thing, top_left_pixel, self.state)

    def pack(self, things: Iterable[Thing]) -> None:
        self.state = self._functional_pack(things, self.state)

    def _functional_add_object(
        self,
        thing: Thing,
        top_left_pixel: Optional[Tuple[int, int]],
        state: List[List[Tuple[int, int, int]]],
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Add a `Thing` to a state such as self.state, in place, with the thing's
            top left pixel at `top_left_pixel` if provided, or in a random
            position otherwise, without violating `self.objects_can_overlap`
        """
        working_state = copy.deepcopy(state)
        if top_left_pixel is None:
            if self.objects_can_overlap:
                top_left_pixel = (
                    random.randrange(0, self.size),
                    random.randrange(0, self.size),
                )
            else:
                spaces = self.find_spaces(thing, working_state)
                if not spaces:
                    raise NoSpace("Not enough space to add that object")
                else:
                    top_left_pixel = random.choice(spaces)

        for thing_row, thing_columns in enumerate(thing.body):
            for thing_column, content in enumerate(thing_columns):
                working_row = top_left_pixel[0] + thing_row
                working_column = top_left_pixel[1] + thing_column
                if not self.objects_can_overlap and (
                    working_state[working_row][working_column] != (0, 0, 0)
                ):
                    raise NoSpace(
                        "There isn't space for that object to "
                        "have the specified `top_left_pixel`!"
                    )
                else:
                    working_state[working_row][working_column] = content
        return working_state

    def _functional_pack(
        self, things: Iterable[Thing], state: List[List[Tuple[int, int, int]]]
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Randomly pack some provided objects into a grid if possible,
            using recursion and backtracking.
        """
        if not things:
            return state
        working_state = copy.deepcopy(state)
        sorted_things = sorted(things, key=lambda x: x.size)
        thing_to_add = sorted_things.pop()  # i.e. largest thing
        spaces = self.find_spaces(thing_to_add, working_state)
        if not spaces:
            raise NoSpace("Not enough space to add largest object.")
        else:
            random.shuffle(spaces)
            for space in spaces:
                working_state = self._functional_add_object(
                    thing_to_add, space, working_state
                )
                try:
                    return self._functional_pack(sorted_things[1:], working_state)
                except NoSpace:
                    continue

        # If the method didn't return, packing is impossible
        raise NoSpace("Not enough space to pack all objects.")

    def array(self):
        return np.asarray(self.state, dtype=np.uint8)
