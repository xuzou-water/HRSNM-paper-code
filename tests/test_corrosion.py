import unittest

import numpy as np

from hrsnm.corrosion import (
    corrosion_exceeds_assumed_wall,
    corrosion_exceedance_length_km,
    corrosion_rate_exceeds_threshold,
)


class CorrosionCriterionTests(unittest.TestCase):
    def test_strictly_above_one_is_exceedance(self):
        values = np.array([0.0, 0.999, 1.0, 1.0001, 2.0])
        actual = corrosion_rate_exceeds_threshold(values)
        np.testing.assert_array_equal(
            actual, np.array([False, False, False, True, True])
        )

    def test_nonfinite_rates_are_not_exceedances(self):
        values = np.array([np.nan, np.inf, -np.inf])
        self.assertFalse(corrosion_rate_exceeds_threshold(values).any())

    def test_custom_threshold(self):
        actual = corrosion_rate_exceeds_threshold([1.0, 1.5, 2.0], 1.5)
        np.testing.assert_array_equal(actual, [False, False, True])

    def test_scalar_returns_bool(self):
        self.assertIs(corrosion_rate_exceeds_threshold(1.1), True)
        self.assertIs(corrosion_rate_exceeds_threshold(1.0), False)

    def test_invalid_threshold_rejected(self):
        for threshold in (-1.0, np.nan, np.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    corrosion_rate_exceeds_threshold([1.0], threshold)

    def test_exceedance_length_uses_only_rates_strictly_above_one(self):
        rates = [0.5, 1.0, 1.1, 2.0, np.nan]
        lengths_m = [100, 200, 300, 400, 500]
        self.assertAlmostEqual(
            corrosion_exceedance_length_km(rates, lengths_m), 0.7
        )

    def test_exceedance_length_ignores_invalid_lengths(self):
        self.assertEqual(
            corrosion_exceedance_length_km([2, 2, 2], [np.nan, -10, 0]), 0.0
        )

    def test_exceedance_length_requires_matching_shapes(self):
        with self.assertRaises(ValueError):
            corrosion_exceedance_length_km([2.0], [100.0, 200.0])

    def test_fifty_year_wall_failure_is_strict(self):
        rates = np.array([0.75, 0.7501, 1.0])
        actual = corrosion_exceeds_assumed_wall(rates, [0.5, 0.5, 1.0])
        np.testing.assert_array_equal(actual, [False, True, False])

    def test_fifty_year_wall_failure_rejects_invalid_inputs(self):
        actual = corrosion_exceeds_assumed_wall(
            [1.0, np.nan, -1.0, 1.0],
            [0.0, 0.5, 0.5, np.nan],
        )
        self.assertFalse(actual.any())

    def test_fifty_year_wall_failure_validates_parameters(self):
        with self.assertRaises(ValueError):
            corrosion_exceeds_assumed_wall(1.0, 0.5, years=0)
        with self.assertRaises(ValueError):
            corrosion_exceeds_assumed_wall(1.0, 0.5, wall_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
