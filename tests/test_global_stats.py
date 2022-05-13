from unittest import TestCase

import numpy as np

from schema_mechanism.core import GlobalStats
from schema_mechanism.core import get_global_stats
from schema_mechanism.core import set_global_stats
from test_share.test_func import common_test_setup


class TestGlobalStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.stats = GlobalStats()

    def test_init(self):
        for initial_baseline_value in np.linspace(-10.0, 10.0):
            stats = GlobalStats(initial_baseline_value=initial_baseline_value)
            self.assertEqual(initial_baseline_value, stats.baseline_value)

    def test_global_stats_accessor_functions(self):
        stats = GlobalStats()
        set_global_stats(stats)

        self.assertEqual(stats, get_global_stats())
