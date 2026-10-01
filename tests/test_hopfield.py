import unittest

import numpy as np

from hopfield import HopfieldNetwork, add_noise, letter_patterns


class TestHopfield(unittest.TestCase):
    def setUp(self):
        self.names, self.patterns = letter_patterns()
        self.net = HopfieldNetwork(self.patterns.shape[1]).train(self.patterns)
        self.rng = np.random.default_rng(0)

    def test_weights_symmetric_with_zero_diagonal(self):
        np.testing.assert_array_equal(self.net.weights, self.net.weights.T)
        self.assertTrue(np.all(np.diag(self.net.weights) == 0))

    def test_stored_patterns_are_fixed_points(self):
        for p in self.patterns:
            out, _ = self.net.recall(p, rng=self.rng)
            np.testing.assert_array_equal(out, p)

    def test_recovers_from_noise(self):
        for p in self.patterns:
            out, _ = self.net.recall(add_noise(p, 0.1, self.rng), rng=self.rng)
            np.testing.assert_array_equal(out, p)

    def test_energy_never_increases_async(self):
        for p in self.patterns:
            _, energies = self.net.recall(add_noise(p, 0.3, self.rng), rng=self.rng)
            self.assertTrue(all(b <= a + 1e-9 for a, b in zip(energies, energies[1:])))

    def test_add_noise_flips_exact_fraction(self):
        p = self.patterns[0]
        self.assertEqual(int(np.sum(add_noise(p, 0.2, self.rng) != p)), round(0.2 * len(p)))

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            self.net.recall(np.zeros(self.net.n))
        with self.assertRaises(ValueError):
            self.net.recall(np.ones(self.net.n + 1))
        with self.assertRaises(RuntimeError):
            HopfieldNetwork(4).recall(np.ones(4))


if __name__ == "__main__":
    unittest.main()
