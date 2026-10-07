#!/usr/bin/env python3
"""Reconoce una figura dibujada a mano en un archivo de texto de 12x12 ('#' trazo, '.' fondo).

usage: python recognize.py examples/circulo_a_mano.txt [más archivos...]
"""
import sys

import numpy as np

from hopfield import HopfieldNetwork, recognize, shape_patterns
from hopfield.shapes import from_text, to_text


def main(paths):
    names, patterns = shape_patterns()
    net = HopfieldNetwork(patterns.shape[1]).train(patterns, rule="pseudoinverse")
    rng = np.random.default_rng(0)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            drawing = from_text(f.read())
        r = recognize(net, names, patterns, drawing, rng=rng)
        verdict = "no reconocida (estado espurio)" if r.name is None else r.name + (" invertida" if r.inverted else "")
        print(f"{path}: {verdict}  (traslape {r.overlap:.2f}, {r.sweeps} barridos)")
        for a, b in zip(to_text(drawing).split("\n"), to_text(r.state).split("\n")):
            print(f"  {a}   ->   {b}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
