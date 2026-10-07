"""Reconocimiento: dejar que la red converja y comparar el estado final con los patrones guardados."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Recognition:
    name: str | None    # None si la red cayó en un estado espurio
    overlap: float      # |m| = |s·p| / N del patrón más parecido (1.0 = idéntico)
    inverted: bool      # la red también guarda el negativo de cada patrón (-p es mínimo de la energía)
    state: np.ndarray
    sweeps: int


def recognize(net, names, patterns, state, threshold=0.95, rng=None):
    final, energies = net.recall(state, rng=rng)
    overlaps = patterns @ final / net.n
    best = int(np.argmax(np.abs(overlaps)))
    m = float(overlaps[best])
    name = names[best] if abs(m) >= threshold else None
    return Recognition(name, abs(m), m < 0, final, len(energies) - 1)
