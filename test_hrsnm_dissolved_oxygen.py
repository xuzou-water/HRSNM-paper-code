import inspect
import unittest
from unittest.mock import patch

import numpy as np

import hk_HRSNM_v7_5_test2 as hk_model
import la_HRSNM_v7_5_test1 as la_model
import toronto_HRSNM_v7_5_test1 as toronto_model
from hrsnm_dissolved_oxygen import saturation_do_mg_l


CITY_MODELS = (hk_model, toronto_model, la_model)


class DissolvedOxygenTests(unittest.TestCase):
    def test_project_temperature_values(self):
        expected = {
            15.0: 10.083858,
            18.0: 9.467000,
            20.0: 9.092426,
            25.0: 8.263457,
            26.0: 8.113626,
            29.0: 7.691292,
        }
        for temp_c, expected_do in expected.items():
            with self.subTest(temp_c=temp_c):
                self.assertAlmostEqual(
                    saturation_do_mg_l(temp_c), expected_do, places=6
                )

    def test_saturation_decreases_with_temperature(self):
        values = [saturation_do_mg_l(temp_c) for temp_c in range(41)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_valid_range_boundaries(self):
        self.assertGreater(saturation_do_mg_l(0.0), 0.0)
        self.assertGreater(saturation_do_mg_l(40.0), 0.0)

    def test_invalid_temperatures_raise_value_error(self):
        for temp_c in (-0.01, 40.01, np.nan, np.inf, -np.inf, None, "bad"):
            with self.subTest(temp_c=temp_c):
                with self.assertRaises(ValueError):
                    saturation_do_mg_l(temp_c)

    def test_each_city_reaction_uses_current_temperature(self):
        concentration = np.ones((1, 11), dtype=float)
        hydraulic_value = np.ones(1, dtype=float)
        for model in CITY_MODELS:
            with self.subTest(model=model.__name__):
                with patch.object(
                    model, "saturation_do_mg_l", return_value=7.5
                ) as saturation:
                    model._derivatives(
                        concentration,
                        hydraulic_value,
                        hydraulic_value,
                        hydraulic_value,
                        hydraulic_value,
                        23.0,
                    )
                saturation.assert_called_once_with(23.0)

    def test_each_city_air_dosage_and_postprocess_use_shared_formula(self):
        for model in CITY_MODELS:
            with self.subTest(model=model.__name__):
                simulation_source = inspect.getsource(
                    next(
                        value for name, value in vars(model).items()
                        if name.startswith("run_simulation_")
                    )
                )
                postprocess_source = inspect.getsource(
                    next(
                        value for name, value in vars(model).items()
                        if name.startswith("postprocess_and_save_")
                    )
                )
                self.assertIn("saturation_do_mg_l(temp_c)", simulation_source)
                self.assertIn("saturation_do=saturation_do", simulation_source)
                self.assertIn("saturation_do_mg_l(temp_c)", postprocess_source)


if __name__ == "__main__":
    unittest.main()
