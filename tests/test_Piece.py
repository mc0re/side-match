import pytest
from src.Piece import Piece


def test_init():
    p = Piece(1, [2, 3, 4, 5])
    assert p.number == 1
    assert p.patterns == [2, 3, 4, 5]


def test_rotate():
    p = Piece(1, [2, 3, 4, 5])
    p.rotate_clockwise()
    assert p.patterns == [5, 2, 3, 4]
