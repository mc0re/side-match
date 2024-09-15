import pytest
from src.Piece import Piece


def test_rotate():
    p = Piece(1, [2, 3, 4])
    assert p.number == 1


if __name__ == "__main__":
    pytest.main()
