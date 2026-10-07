"""Figuras geométricas de 12x12 como patrones bipolares (+1 trazo, -1 fondo), generadas por fórmula."""
import numpy as np

SIZE = 12
_Y, _X = np.mgrid[0:SIZE, 0:SIZE]
_C = (SIZE - 1) / 2


def _circle():
    r = np.hypot(_Y - _C, _X - _C)
    return (r >= 4.0) & (r <= 5.3)


def _square():
    inside = (_Y >= 1) & (_Y <= 10) & (_X >= 1) & (_X <= 10)
    return inside & ((_Y == 1) | (_Y == 10) | (_X == 1) | (_X == 10))


def _triangle():
    half = (_Y - 1) * 0.5  # la mitad del ancho crece medio pixel por renglón
    sides = (_Y >= 1) & (_Y <= 10) & (np.abs(np.abs(_X - _C) - half) <= 0.5)
    return sides | ((_Y == 10) & (_X >= 1) & (_X <= 10))


def _cross():
    return (np.abs(_X - _C) <= 1) | (np.abs(_Y - _C) <= 1)


def _diamond():
    return np.abs(np.abs(_X - _C) + np.abs(_Y - _C) - 4.5) <= 0.6


SHAPES = {"círculo": _circle, "cuadrado": _square, "triángulo": _triangle, "cruz": _cross, "rombo": _diamond}


def shape_patterns(names=None):
    names = names or list(SHAPES)
    return names, np.array([np.where(SHAPES[n](), 1, -1).ravel() for n in names])


def to_text(vector):
    grid = np.asarray(vector).reshape(SIZE, SIZE)
    return "\n".join("".join("#" if v > 0 else "." for v in row) for row in grid)


def from_text(text):
    """Convierte un dibujo de 12 renglones de '#' y '.' en un vector bipolar."""
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if len(rows) != SIZE or any(len(r) != SIZE for r in rows):
        raise ValueError(f"el dibujo debe ser de {SIZE}x{SIZE}")
    if set("".join(rows)) - {"#", "."}:
        raise ValueError("solo se permiten '#' y '.'")
    return np.array([1 if c == "#" else -1 for r in rows for c in r])


def occlude(pattern, side):
    """Borra (pone en fondo) la mitad indicada de la figura: 'left', 'right', 'top' o 'bottom'."""
    grid = np.array(pattern, copy=True).reshape(SIZE, SIZE)
    half = SIZE // 2
    region = {"left": (slice(None), slice(0, half)), "right": (slice(None), slice(half, None)),
              "top": (slice(0, half), slice(None)), "bottom": (slice(half, None), slice(None))}
    if side not in region:
        raise ValueError("side debe ser left, right, top o bottom")
    grid[region[side]] = -1
    return grid.ravel()
