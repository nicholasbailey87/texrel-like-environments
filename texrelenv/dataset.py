from .environment import Grid, ThingMaker


class DataSet:
    def __init__(
        self,
        grid_size=16,
        hard_boundary=True,
        objects_can_overlap=False,
        thing_size=4,
        distinct_shapes=9,
        distinct_colours=9,
        things_per_image=5,
        hold_out_things=0.2,
        hold_out_images=0.2,
    ):
        self.grid_size = grid_size
        self.hard_boundary = hard_boundary
        self.objects_can_overlap = objects_can_overlap
        self.things_per_image = things_per_image
        self.hold_out_images = hold_out_images

        self.thingmaker = ThingMaker(
            size=thing_size,
            distinct_colours=distinct_colours,
            distinct_shapes=distinct_shapes,
            hold_out=hold_out_things,
        )

    def sample(self, split, n):
        samples = []
        for _ in range(n):
            sample_dict = {}
            g = Grid(
                size=self.grid_size,
                hard_boundary=self.hard_boundary,
                objects_can_overlap=self.objects_can_overlap,
            )
            things = [
                self.thingmaker.thing(split) for _ in range(self.things_per_image)
            ]
            g.pack(things)
            samples.append(sample_dict)
