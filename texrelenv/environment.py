import copy
import random
from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .color import get_kelly_colors, ordinal_to_color
from .exceptions import BadSplit, NoSpace


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
    def __init__(self, color: Tuple[int, int, int], template: ThingTemplate) -> None:
        self.size = template.size
        self.body = [[color if j else 0 for j in i] for i in template.pattern]


class ThingMaker:
    def __init__(self, size=4, distinct_shapes=9, distinct_colors=9, hold_out=0.2):
        """
        Args:
            hold_out - what portion of `Thing`s (i.e. color-template
                combinations) are held out for testing?
        """
        self.hold_out = hold_out

        self.distinct_colors = distinct_colors

        self.templates = []

        while len(self.templates) < distinct_shapes:
            proposed_template = ThingTemplate(size)
            if proposed_template.hash() not in [t.hash() for t in self.templates]:
                self.templates.append(proposed_template)

        things = [
            Thing(color, template)
            for color in range(1, self.distinct_colors + 1)
            for template in self.templates
        ]

        random.shuffle(things)

        self.train_things = things[int(len(things) // (1 / hold_out)) :]
        self.test_things = things[: int(len(things) // (1 / hold_out))]

    def thing(self, split: str) -> List[List[Tuple[int, int, int]]]:
        assert split in ["train", "test"]
        if split == "train":
            return random.choice(self.train_things)
        elif split == "test":
            return random.choice(self.test_things + self.train_things)
        else:
            raise BadSplit("Split must be 'train' or 'test'")


class Grid:
    # TODO: Get grid to remember what things it has in what positions!
    def __init__(
        self, size=16, hard_boundary=True, objects_can_overlap: bool = False
    ) -> None:
        self.size = size
        self.hard_boundary = hard_boundary
        self.objects_can_overlap = objects_can_overlap
        self.state = []
        self.state_image = [[0] * size for _ in range(size)]

    def _add_thing(
        self,
        thing: Thing,
        top_left_pixel: Tuple[int, int],
        state: List[dict],
        state_image: List[List[Tuple[int, int]]],
    ) -> Tuple[List[dict], List[List[Tuple[int, int]]]]:
        state = copy.deepcopy(state)
        state_image = self._add_to_state_image(
            thing, top_left_pixel, copy.deepcopy(state_image)
        )
        state.append({"thing": thing, "top_left_pixel": top_left_pixel})
        return state, state_image

    def _add_to_state_image(
        self,
        thing: Thing,
        top_left_pixel: Tuple[int, int],
        state_image: List[List[Tuple[int, int]]],
    ) -> bool:
        state_image = copy.deepcopy(state_image)
        for thing_row, thing_columns in enumerate(thing.body):
            for thing_column, content in enumerate(thing_columns):
                working_row = top_left_pixel[0] + thing_row
                working_column = top_left_pixel[1] + thing_column
                if (
                    (working_row < 0)
                    or (working_column < 0)
                    or (working_row >= len(state_image))
                    or (working_column >= len(state_image))
                ):
                    # Out of frame
                    continue
                else:
                    state_image[working_row][working_column] = content
        return state_image

    def _find_spaces(
        self, thing: Thing, state_image: List[List[Tuple[int, int, int]]]
    ) -> List[Tuple[int, int]]:
        """
        Find empty spaces for a square object to be added to the grid, where
            it will not overlap another object. Return the answer as a list of
            tuples giving the row and column coordinates of the top left pixel
            of each found square space
        """
        filled_squares = np.array(state_image)

        if not self.hard_boundary:
            filled_squares = np.pad(
                filled_squares,
                pad_width=thing.size - 1,
                mode="constant",
                constant_values=0,
            )
        space_shape = (thing.size, thing.size)
        space = np.zeros((thing.size, thing.size))
        candidates = sliding_window_view(filled_squares, space_shape)
        matches = np.all(candidates == space, axis=(2, 3))
        coords = [tuple(coords) for coords in np.argwhere(matches).tolist()]
        if not self.hard_boundary:
            coords = [(a - (thing.size - 1), b - (thing.size - 1)) for a, b in coords]
        return coords

    # def _functional_add_object(
    #     self,
    #     thing: Thing,
    #     top_left_pixel: Optional[Tuple[int, int]],
    #     state: List[List[Tuple[int, int, int]]],
    # ) -> List[List[Tuple[int, int, int]]]:
    #     """
    #     Add a `Thing` to a state such as self.state, in place, with the thing's
    #         top left pixel at `top_left_pixel` if provided, or in a random
    #         position otherwise, without violating `self.objects_can_overlap`
    #     """
    #     working_state = copy.deepcopy(state)
    #     if top_left_pixel is None:
    #         if self.objects_can_overlap:
    #             top_left_pixel = (
    #                 random.randrange(0, self.size),
    #                 random.randrange(0, self.size),
    #             )
    #         else:
    #             spaces = self._find_spaces(thing, working_state)
    #             if not spaces:
    #                 raise NoSpace("Not enough space to add that object")
    #             else:
    #                 top_left_pixel = random.choice(spaces)

    #     for thing_row, thing_columns in enumerate(thing.body):
    #         for thing_column, content in enumerate(thing_columns):
    #             working_row = top_left_pixel[0] + thing_row
    #             working_column = top_left_pixel[1] + thing_column
    #             if (
    #                 (working_row < 0)
    #                 or (working_column < 0)
    #                 or (working_row >= len(working_state))
    #                 or (working_column >= len(working_state))
    #             ):
    #                 continue
    #             elif not self.objects_can_overlap and (
    #                 working_state[working_row][working_column] != 0
    #             ):
    #                 raise NoSpace(
    #                     "There isn't space for that object to "
    #                     "have the specified `top_left_pixel`!"
    #                 )
    #             else:
    #                 working_state[working_row][working_column] = content
    #     return working_state

    def _functional_pack(
        self,
        things: Iterable[Thing],
        state: List[dict],
        state_image: List[List[Tuple[int, int]]],
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Randomly pack some provided objects into a grid if possible,
            using recursion and backtracking.
        """
        if not things:
            return state, state_image
        state = copy.deepcopy(state)
        state_image = copy.deepcopy(state_image)
        sorted_things = sorted(things, key=lambda x: x.size)
        thing_to_add = sorted_things.pop()  # i.e. largest thing
        spaces = self._find_spaces(thing_to_add, state_image)
        if not spaces:
            raise NoSpace("Not enough space to add largest object.")
        else:
            random.shuffle(spaces)
            for space in spaces:
                state, state_image = self._add_thing(
                    thing_to_add, space, state, state_image
                )
                try:
                    return self._functional_pack(sorted_things[1:], state, state_image)
                except NoSpace:
                    continue

        # If the method didn't return, packing is impossible
        raise NoSpace("Not enough space to pack all objects.")

    # def state_to_image():
    #     image = [[0] * size for _ in range(size)]
    #     for entity in self.state:
    #         thing, top_left_pixel = entity.values()
    #         for thing_row, thing_columns in enumerate(thing.body):
    #                 for thing_column, content in enumerate(thing_columns):
    #                     working_row = top_left_pixel[0] + thing_row
    #                     working_column = top_left_pixel[1] + thing_column
    #                     if (
    #                         (working_row < 0)
    #                         or (working_column < 0)
    #                         or (working_row >= len(working_state))
    #                         or (working_column >= len(working_state))
    #                     ):
    #                         # Out of frame
    #                         continue
    #                     else:
    #                         image[working_row][working_column] = content

    # def add_object(
    #     self, thing: Thing, top_left_pixel: Optional[Tuple[int, int]]
    # ) -> None:
    #     self.state = self._functional_add_object(thing, top_left_pixel, self.state)

    def pack(self, things: Iterable[Thing]) -> None:
        self.state, self.state_image = self._functional_pack(
            things, self.state, self.state_image
        )

    def colorized_image(self):

        colors_used = set(sum(self.state_image, []))

        # get Kelly colors sorted darkest to lightest
        kelly_colors = sorted(get_kelly_colors(), key=lambda x: sum(x))

        if len(colors_used) < 22:
            color_map = defaultdict(lambda: random.choice(kelly_colors[1:]))
            color_map[0] = kelly_colors[0]
        else:
            color_map = {c: ordinal_to_color(c) for c in colors_used}

        return np.asarray(
            [[color_map[pixel] for pixel in row] for row in self.state_image],
            dtype=np.uint8,
        )
