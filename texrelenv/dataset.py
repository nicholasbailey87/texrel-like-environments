import pickle
from typing import List

from .environment import generate_view, random_object_template
from .colour import get_kelly_colours

class ProblemSizeUndefined(Exception):
    pass

class TexRelEnv(Dataset):
    """ 
    PyTorch-DataLoader-compatible dataset for referential games and reconstruction games.
    """
    def __init__(
        self,
        split,
        distinct_objects=9,
        distinct_colours=9,
        object_vocabulary=None,
        colour_vocabulary=None,
        grid_size=16,
        object_size=4,
        objects_per_image=5,
        partial_objects=True,
        overlapping_objects=True
    ):
        
        assert split in {'train', 'test'}

        if colour_vocabulary is None:
            assert distinct_colours <= 22
            self.colour_vocabulary = get_kelly_colours()[:distinct_colours]
        else:
            self.colour_vocabulary = colour_vocabulary

        if object_vocabulary is None:
            self.object_vocabulary = []
        elif distinct_objects is not None:
            self.colour_vocabulary = colour_vocabulary

        self.split = split
        self.object_vocabulary = object_vocabulary
        self.grid_size = grid_size,
        self.object_size = object_size,
        self.objects_per_image = objects_per_image,
        self.distinct_objects = distinct_objects,
        self.distinct_colours = distinct_colours,
        self.partial_objects = partial_objects,
        self.overlapping_objects = overlapping_objects
    
    def generate_samples(k: int) -> List[List[List[List[int]]]]:
        samples = []
        while len(samples) < k:
            new_sample = generate_view(
                self.grid_size,
    objects: Iterable[List[List[Tuple[int, int, int]]]],
    partial_objects = True
)
    def __getitem__(self, idx):
    