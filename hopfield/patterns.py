"""Letras de 7x7 como patrones bipolares (# = +1, . = -1)."""
import numpy as np

SIZE = 7

LETTERS = {
    "A": ["..###..",
          ".#...#.",
          "#.....#",
          "#######",
          "#.....#",
          "#.....#",
          "#.....#"],
    "T": ["#######",
          "...#...",
          "...#...",
          "...#...",
          "...#...",
          "...#...",
          "...#..."],
    "X": ["#.....#",
          ".#...#.",
          "..#.#..",
          "...#...",
          "..#.#..",
          ".#...#.",
          "#.....#"],
    "L": ["#......",
          "#......",
          "#......",
          "#......",
          "#......",
          "#......",
          "#######"],
}


def to_vector(rows):
    return np.array([1 if c == "#" else -1 for row in rows for c in row])


def to_text(vector):
    grid = np.asarray(vector).reshape(SIZE, SIZE)
    return "\n".join("".join("#" if v > 0 else "." for v in row) for row in grid)


def letter_patterns(names=None):
    names = names or list(LETTERS)
    return names, np.array([to_vector(LETTERS[n]) for n in names])
