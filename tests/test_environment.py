import unittest
from unittest.mock import Mock

from texrelenv import environment as env


class TestThingTemplate(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.template = env.ThingTemplate(4)

    def test_pattern(self):
        self.assertEqual(len(self.template.pattern), 4)
        self.assertTrue(sum(sum(self.template.pattern, [])) <= 16)
        self.assertTrue(sum(sum(self.template.pattern, [])) >= 0)

    def test_hash(self):
        self.assertEqual(self.template.hash(), hash(str(self.template.pattern)))


class TestThing(unittest.TestCase):
    pass


class TestThingMaker(unittest.TestCase):
    pass


class TestGrid(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.maxDiff = None

    def test_init(self):
        grid = env.Grid(5, True)
        self.assertEqual(len(grid.state), len(grid.state[0]))
        self.assertEqual(len(grid.state), grid.size)

    def test_find_spaces_hard_boundary(self):
        mock_thing = Mock()
        mock_thing.size = 2
        grid = env.Grid(5, hard_boundary=True, objects_can_overlap=False)
        for pixel in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2)]:
            row, col = pixel
            grid.state[row][col] = (255, 255, 255)
            """
            We have filled these pixels:

            o o o o o
            o x x x o
            o x x o o
            o x x o o
            o o o o o
            """
        spaces = grid._find_spaces(mock_thing, grid.state)
        self.assertCountEqual(spaces, [(2, 3), (3, 3)])

    def test_find_spaces_soft_boundary(self):
        mock_thing = Mock()
        mock_thing.size = 2
        grid = env.Grid(5, hard_boundary=False, objects_can_overlap=False)
        for pixel in [(a, b) for a in range(4) for b in range(5)]:
            row, col = pixel
            grid.state[row][col] = (255, 255, 255)
            """
            We have filled these pixels:

            x x x x x
            x x x x x
            x x x x x
            x x x x x
            o o o o o
            """
        spaces = grid._find_spaces(mock_thing, grid.state)
        self.assertCountEqual(spaces, [(4, i) for i in range(-1, 5, 1)])

    def test_functional_add_object(self):
        mock_thing = Mock()
        mock_thing.size = 2
        mock_thing.body = [
            [(255, 255, 255), (255, 255, 255)],
            [(255, 255, 255), (255, 255, 255)],
        ]
        grid = env.Grid(4, hard_boundary=False, objects_can_overlap=False)
        after_add = grid._functional_add_object(mock_thing, (1, 1), grid.state)
        self.assertEqual(
            after_add,
            [
                [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
                [(0, 0, 0), (255, 255, 255), (255, 255, 255), (0, 0, 0)],
                [(0, 0, 0), (255, 255, 255), (255, 255, 255), (0, 0, 0)],
                [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
            ],
        )
