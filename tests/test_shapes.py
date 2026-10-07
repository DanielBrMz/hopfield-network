import unittest

import numpy as np

from hopfield import HopfieldNetwork, add_noise, occlude, recognize, shape_patterns
from hopfield.shapes import SIZE, from_text, to_text


class TestShapes(unittest.TestCase):
    def setUp(self):
        self.names, self.patterns = shape_patterns()
        self.net = HopfieldNetwork(self.patterns.shape[1]).train(self.patterns, rule="pseudoinverse")
        self.rng = np.random.default_rng(0)

    def test_shapes_are_bipolar_grids(self):
        self.assertEqual(self.patterns.shape, (len(self.names), SIZE * SIZE))
        self.assertTrue(np.all(np.isin(self.patterns, (-1, 1))))

    def test_text_round_trip(self):
        for p in self.patterns:
            np.testing.assert_array_equal(from_text(to_text(p)), p)

    def test_pseudoinverse_recognizes_noisy_shapes(self):
        for name, p in zip(self.names, self.patterns):
            result = recognize(self.net, self.names, self.patterns, add_noise(p, 0.2, self.rng), rng=self.rng)
            self.assertEqual(result.name, name)
            self.assertFalse(result.inverted)

    def test_recognizes_half_occluded_shapes(self):
        for name, p in zip(self.names, self.patterns):
            result = recognize(self.net, self.names, self.patterns, occlude(p, "right"), rng=self.rng)
            self.assertEqual(result.name, name)

    def test_pseudoinverse_weights_symmetric_with_zero_diagonal(self):
        np.testing.assert_allclose(self.net.weights, self.net.weights.T, atol=1e-9)
        self.assertTrue(np.allclose(np.diag(self.net.weights), 0))

    def test_hebb_fails_on_correlated_shapes(self):
        # Shapes share most of the background, so Hebb's crosstalk breaks them; this is the reason for the projection rule.
        hebb = HopfieldNetwork(self.patterns.shape[1]).train(self.patterns)
        fixed = sum(np.array_equal(hebb.recall(p, rng=self.rng)[0], p) for p in self.patterns)
        self.assertLess(fixed, len(self.patterns))

    def test_inverted_shape_is_flagged(self):
        result = recognize(self.net, self.names, self.patterns, -self.patterns[0], rng=self.rng)
        self.assertEqual(result.name, self.names[0])
        self.assertTrue(result.inverted)

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            HopfieldNetwork(4).train(np.ones((1, 4)), rule="oja")


if __name__ == "__main__":
    unittest.main()
