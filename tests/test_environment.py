import unittest

from texrelenv import environment as env


class TestThingTemplate(unittest.TestCase):
    def test_positive_number(self):
        template = env.ThingTemplate(4)
        self.assertEqual(abs(10), 10)
        assert template.pattern


if __name__ == "__main__":
    unittest.main()
