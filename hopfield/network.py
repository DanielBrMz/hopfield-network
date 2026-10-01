"""Red de Hopfield discreta con neuronas bipolares (-1, +1)."""
import numpy as np


class HopfieldNetwork:
    """Memoria asociativa: guarda patrones y los recupera a partir de versiones ruidosas.

    Aprendizaje hebbiano:  W = (1/N) * sum_p x_p x_p^T,  con diagonal en cero.
    Actualización:         s_i <- sign(sum_j W_ij s_j)
    Energía:               E(s) = -1/2 * s^T W s
    Cada actualización asíncrona no aumenta la energía, así que la red siempre converge a un mínimo local.
    """

    def __init__(self, n_neurons):
        if n_neurons <= 0:
            raise ValueError("n_neurons debe ser positivo")
        self.n = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self._normalized = self.weights
        self.n_patterns = 0

    def train(self, patterns):
        """Regla de Hebb. `patterns` es una matriz (p, n) con valores -1/+1."""
        patterns = self._check(np.atleast_2d(patterns))
        for x in patterns:
            self.weights += np.outer(x, x)
        np.fill_diagonal(self.weights, 0)
        self.n_patterns += len(patterns)
        self._normalized = self.weights / self.n
        return self

    def energy(self, state):
        state = self._check(state)
        return -0.5 * state @ self._normalized @ state

    def recall(self, state, mode="async", max_iters=100, rng=None):
        """Itera hasta que el estado deja de cambiar (punto fijo) o se alcanza `max_iters`.

        Regresa (estado_final, historial_de_energia).
        mode="async": una neurona a la vez en orden aleatorio (converge siempre).
        mode="sync":  todas a la vez (puede oscilar entre dos estados).
        """
        if self.n_patterns == 0:
            raise RuntimeError("la red no tiene patrones entrenados")
        s = self._check(state).astype(float).copy()
        rng = rng or np.random.default_rng()
        energies = [self.energy(s)]
        for _ in range(max_iters):
            previous = s.copy()
            if mode == "async":
                for i in rng.permutation(self.n):
                    s[i] = 1.0 if self._normalized[i] @ s >= 0 else -1.0
            elif mode == "sync":
                s = np.where(self._normalized @ s >= 0, 1.0, -1.0)
            else:
                raise ValueError("mode debe ser 'async' o 'sync'")
            energies.append(self.energy(s))
            if np.array_equal(s, previous):
                break
        return s.astype(int), energies

    def capacity(self):
        """Capacidad teórica aproximada de la regla de Hebb: ~0.138 N patrones."""
        return 0.138 * self.n

    def _check(self, x):
        x = np.asarray(x)
        if x.shape[-1] != self.n:
            raise ValueError(f"se esperaban {self.n} neuronas, llegaron {x.shape[-1]}")
        if not np.all(np.isin(x, (-1, 1))):
            raise ValueError("los valores deben ser -1 o +1")
        return x


def add_noise(pattern, fraction, rng=None):
    """Invierte una fracción de los bits del patrón."""
    rng = rng or np.random.default_rng()
    noisy = np.array(pattern, copy=True)
    flip = rng.choice(len(noisy), size=int(round(fraction * len(noisy))), replace=False)
    noisy[flip] *= -1
    return noisy
