from typing import Tuple


def ordinal_to_color(ordinal: int) -> Tuple[int, int, int]:
    """
    Given an integer, deterministically produce a unique(ish) RGB color.
    """
    if ordinal == 0:
        # ensure 0 always comes out black
        return (0, 0, 0)
    else:
        hashstring = str(abs(hash(str(ordinal))))
        a, b, c = int(hashstring[:6]), int(hashstring[6:12]), int(hashstring[12:])
        return ((a % 26) * 10, (b % 26) * 10, (c % 26) * 10)


def get_kelly_colors():
    """
    See https://eleanormaclure.wordpress.com/wp-content/uploads/2011/03/...
      ...color-coding.pdf
    """
    return [
        (34, 34, 34),
        (242, 243, 244),
        (243, 195, 0),
        (135, 86, 146),
        (243, 132, 0),
        (161, 202, 241),
        (190, 0, 50),
        (194, 178, 128),
        (132, 132, 130),
        (0, 136, 86),
        (230, 143, 172),
        (0, 103, 165),
        (249, 147, 121),
        (96, 78, 151),
        (246, 166, 0),
        (179, 68, 108),
        (220, 211, 0),
        (136, 45, 23),
        (141, 182, 0),
        (101, 69, 34),
        (226, 88, 34),
        (43, 61, 38),
    ]
