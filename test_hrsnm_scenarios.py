import unittest

from hrsnm_scenarios import scenario_axes, scenario_count


class ScenarioGridTests(unittest.TestCase):
    def test_small8_uses_parameter_cube_corners(self):
        self.assertEqual(scenario_axes("small8"), ((15, 25), (5, 25), (250, 800)))
        self.assertEqual(scenario_count("small8"), 8)

    def test_full27_is_complete_manuscript_grid(self):
        self.assertEqual(
            scenario_axes("full27"),
            ((15, 20, 25), (5, 15, 25), (250, 525, 800)),
        )
        self.assertEqual(scenario_count("full27"), 27)

    def test_unknown_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_axes("other")


if __name__ == "__main__":
    unittest.main()
