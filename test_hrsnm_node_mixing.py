import unittest

import numpy as np
import pandas as pd

from hk_HRSNM_v7_5_test2 import generate_source_concentrations_hk
from la_HRSNM_v7_5_test1 import generate_source_concentrations_la
from toronto_HRSNM_v7_5_test1 import generate_source_concentrations_toronto

from hrsnm_node_mixing import (
    apply_do_overrides,
    build_node_dwf,
    concentration_node_order,
    mix_node_concentration,
    node_flow_balance_summary,
)


class NodeMixingTests(unittest.TestCase):
    def test_source_with_local_dwf(self):
        local = np.arange(11, dtype=float)
        actual = mix_node_concentration([], np.empty((0, 11)), 2.0, local)
        np.testing.assert_allclose(actual, local)

    def test_internal_node_with_local_dwf(self):
        upstream = np.full(11, 10.0)
        local = np.full(11, 40.0)
        actual = mix_node_concentration([3.0], [upstream], 1.0, local)
        np.testing.assert_allclose(actual, np.full(11, 17.5))

    def test_multiple_upstreams_and_component_mass_balance(self):
        c1 = np.arange(11, dtype=float)
        c2 = np.arange(11, dtype=float) + 10.0
        local = np.arange(11, dtype=float) + 20.0
        actual = mix_node_concentration([2.0, 3.0], [c1, c2], 5.0, local)
        expected_mass = 2.0 * c1 + 3.0 * c2 + 5.0 * local
        np.testing.assert_allclose(actual * 10.0, expected_mass)

    def test_no_local_dwf_matches_legacy_weighted_mix(self):
        c1 = np.full(11, 2.0)
        c2 = np.full(11, 8.0)
        actual = mix_node_concentration([1.0, 3.0], [c1, c2], 0.0, np.zeros(11))
        np.testing.assert_allclose(actual, np.full(11, 6.5))

    def test_zero_total_flow_returns_zero(self):
        actual = mix_node_concentration([0.0], [np.full(11, 99.0)], 0.0, np.ones(11))
        np.testing.assert_array_equal(actual, np.zeros(11))

    def test_override_order(self):
        actual = apply_do_overrides(
            np.zeros(11), air_dosage=True, saturation_do=10.0, forced_do=4.56
        )
        self.assertEqual(actual[3], 4.56)

    def test_air_dosage_requires_explicit_saturation_do(self):
        with self.assertRaises(ValueError):
            apply_do_overrides(np.zeros(11), air_dosage=True)

    def test_source_order_preserved(self):
        self.assertEqual(
            concentration_node_order(["s2", "s1"], ["s1", "n2", "n1"]),
            ["s2", "s1", "n2", "n1"],
        )

    def test_fixed_seed_keeps_legacy_source_samples(self):
        sources = ["s2", "s1", "s3"]
        extended = concentration_node_order(sources, ["s1", "n2", "n1"])
        for generator in (
            generate_source_concentrations_hk,
            generate_source_concentrations_toronto,
            generate_source_concentrations_la,
        ):
            legacy = generator(sources)
            revised = generator(extended)
            np.testing.assert_array_equal(revised[:len(sources)], legacy)

    def test_dwf_validation_and_flow_audit(self):
        nodes = pd.DataFrame(
            {
                "node": ["a", "b", "c"],
                "DWF_cms": [1.0, 2.0, np.nan],
                "is_original": [True, True, False],
            }
        )
        dwf, positive = build_node_dwf(nodes, ["a", "b", "c"])
        self.assertEqual(positive, ["a", "b"])
        pipes = pd.DataFrame(
            {
                "start": ["a", "b"],
                "end": ["b", "c"],
                "flowrate": [1.0, 3.0],
            }
        )
        summary = node_flow_balance_summary(pipes, dwf)
        self.assertEqual(summary["n_flow_balance_failures"], 0)
        self.assertEqual(summary["n_internal_positive_dwf_nodes"], 1)

    def test_invalid_original_missing_dwf_rejected(self):
        nodes = pd.DataFrame(
            {"node": ["a"], "DWF_cms": [np.nan], "is_original": [True]}
        )
        with self.assertRaises(ValueError):
            build_node_dwf(nodes, ["a"])


if __name__ == "__main__":
    unittest.main()
