import copy
import json
import random
from typing import Iterable, List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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


class Thing(dict):
    def __init__(self, color: int, template: ThingTemplate) -> None:
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
        state.append({"body": thing.body, "top_left": top_left_pixel})
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

    def pack(self, things: Iterable[Thing]) -> None:
        self.state, self.state_image = self._functional_pack(
            things, self.state, self.state_image
        )

    @property
    def dict(self) -> str:
        """
        Represent the grid as a dict, useful for creating datasets
        """
        return {"tokens": sum(self.state_image, []), "shapes": self.state}

    @property
    def json(self) -> str:
        """
        Represent the grid as a JSON string, useful for creating datasets
        """
        return json.dumps(self.dict)


class Environment:
    def __init__(
        self,
        grid_size=16,
        hard_boundary=True,
        objects_can_overlap=False,
        thing_size=4,
        distinct_shapes=9,
        distinct_colors=9,
        things_per_image=5,
        hold_out_things=0.2,
        hold_out_images=0.2,
    ):
        self.grid_size = grid_size
        self.hard_boundary = hard_boundary
        self.objects_can_overlap = objects_can_overlap
        self.things_per_image = things_per_image
        self.hold_out_images = hold_out_images
        self.hold_out_things = hold_out_things

        self.thingmaker = ThingMaker(
            size=thing_size,
            distinct_colors=distinct_colors,
            distinct_shapes=distinct_shapes,
            hold_out=hold_out_things,
        )

    def _sample_grid(self, split):
        grid = Grid(
            size=self.grid_size,
            hard_boundary=self.hard_boundary,
            objects_can_overlap=self.objects_can_overlap,
        )
        things = [self.thingmaker.thing(split) for _ in range(self.things_per_image)]

        state, state_image = grid._functional_pack(things, grid.state, grid.state_image)

        if split == "train":
            while abs(hash(str(state_image))) % 100 < self.hold_out_images * 100:
                state, state_image = grid._functional_pack(
                    things, grid.state, grid.state_image
                )
        elif split == "test":
            while abs(hash(str(state_image))) % 100 > self.hold_out_images * 100:
                state, state_image = grid._functional_pack(
                    things, grid.state, grid.state_image
                )

        grid.state, grid.state_image = state, state_image

        return grid

    def sample(self, split="train", n=1) -> List[dict]:
        return [self._sample_grid(split) for _ in range(n)]
