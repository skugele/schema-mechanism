from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import ECItemStats
from schema_mechanism.data_structures import ERItemStats
from schema_mechanism.data_structures import ReadOnlyECItemStats
from schema_mechanism.data_structures import ReadOnlyERItemStats
from schema_mechanism.data_structures import SchemaStats


class TestECItemStatistics(TestCase):
    def test_init(self):
        schema_stats = SchemaStats()
        item_stats = ECItemStats(schema_stats)

        self.assertEqual(item_stats.n_success_and_on, 0)
        self.assertEqual(item_stats.n_success_and_off, 0)
        self.assertEqual(item_stats.n_fail_and_on, 0)
        self.assertEqual(item_stats.n_fail_and_off, 0)

        self.assertIs(item_stats.success_corr, np.NAN)
        self.assertIs(item_stats.failure_corr, np.NAN)

    def test_update_1(self):
        # Single Update - All Base Cases
        ################################
        schema_stats = SchemaStats()
        item_stats = ECItemStats(schema_stats)

        schema_stats.update(activated=True)
        item_stats.update(item_on=True, success=True)

        self.assertEqual(item_stats.n_on, 1)
        self.assertEqual(item_stats.n_off, 0)

        self.assertEqual(item_stats.n_success_and_on, 1)
        self.assertEqual(item_stats.n_success_and_off, 0)
        self.assertEqual(item_stats.n_fail_and_on, 0)
        self.assertEqual(item_stats.n_fail_and_off, 0)

        self.assertEqual(item_stats.success_corr, 1.0)
        self.assertIs(item_stats.failure_corr, np.NAN)

    def test_update_2(self):
        schema_stats = SchemaStats()
        item_stats = ECItemStats(schema_stats)

        schema_stats.update(activated=True)
        item_stats.update(item_on=False, success=True)

        self.assertEqual(item_stats.n_on, 0)
        self.assertEqual(item_stats.n_off, 1)

        self.assertEqual(item_stats.n_success_and_on, 0)
        self.assertEqual(item_stats.n_success_and_off, 1)
        self.assertEqual(item_stats.n_fail_and_on, 0)
        self.assertEqual(item_stats.n_fail_and_off, 0)

        self.assertEqual(item_stats.success_corr, 0.0)
        self.assertIs(item_stats.failure_corr, np.NAN)

    def test_update_3(self):
        schema_stats = SchemaStats()
        item_stats = ECItemStats(schema_stats)

        schema_stats.update(activated=True)
        item_stats.update(item_on=True, success=False)

        self.assertEqual(item_stats.n_on, 1)
        self.assertEqual(item_stats.n_off, 0)

        self.assertEqual(item_stats.n_success_and_on, 0)
        self.assertEqual(item_stats.n_success_and_off, 0)
        self.assertEqual(item_stats.n_fail_and_on, 1)
        self.assertEqual(item_stats.n_fail_and_off, 0)

        self.assertIs(item_stats.success_corr, np.NAN)
        self.assertEqual(item_stats.failure_corr, 1.0)

    def test_update_4(self):
        schema_stats = SchemaStats()
        item_stats = ECItemStats(schema_stats)

        schema_stats.update(activated=True)
        item_stats.update(item_on=False, success=False)

        self.assertEqual(item_stats.n_on, 0)
        self.assertEqual(item_stats.n_off, 1)

        self.assertEqual(item_stats.n_success_and_on, 0)
        self.assertEqual(item_stats.n_success_and_off, 0)
        self.assertEqual(item_stats.n_fail_and_on, 0)
        self.assertEqual(item_stats.n_fail_and_off, 1)

        self.assertIs(item_stats.success_corr, np.NAN)
        self.assertEqual(item_stats.failure_corr, 0.0)

    def test_update_5(self):
        # Multiple Updates
        ##################
        schema_stats = SchemaStats()
        item_stats = ECItemStats(schema_stats)

        schema_stats.update(activated=True, count=40)
        schema_stats.update(activated=False, count=10)

        item_stats.update(item_on=True, success=True, count=32)
        item_stats.update(item_on=False, success=True, count=2)
        item_stats.update(item_on=True, success=False, count=8)
        item_stats.update(item_on=False, success=False, count=8)

        self.assertEqual(item_stats.n_on, 40)
        self.assertEqual(item_stats.n_off, 10)

        self.assertEqual(item_stats.n_success_and_on, 32)
        self.assertEqual(item_stats.n_success_and_off, 2)
        self.assertEqual(item_stats.n_fail_and_on, 8)
        self.assertEqual(item_stats.n_fail_and_off, 8)

        self.assertEqual(item_stats.success_corr, 0.8)
        self.assertEqual(item_stats.failure_corr, 0.2)


class TestERItemStatistics(TestCase):
    def test_init(self):
        schema_stats = SchemaStats()
        item_stats = ERItemStats(schema_stats)

        self.assertEqual(item_stats.n_on_and_activated, 0)
        self.assertEqual(item_stats.n_on_and_not_activated, 0)
        self.assertEqual(item_stats.n_off_and_activated, 0)
        self.assertEqual(item_stats.n_off_and_not_activated, 0)

        self.assertIs(item_stats.positive_transition_corr, np.NAN)
        self.assertIs(item_stats.negative_transition_corr, np.NAN)

    def test_update_1(self):
        # Single Update - All Base Cases
        ################################
        schema_stats = SchemaStats()
        item_stats = ERItemStats(schema_stats)

        schema_stats.update(activated=True)
        item_stats.update(item_on=True, activated=True)

        self.assertEqual(item_stats.n_on, 1)
        self.assertEqual(item_stats.n_off, 0)

        self.assertEqual(item_stats.n_on_and_activated, 1)
        self.assertEqual(item_stats.n_on_and_not_activated, 0)
        self.assertEqual(item_stats.n_off_and_activated, 0)
        self.assertEqual(item_stats.n_off_and_not_activated, 0)
        self.assertEqual(item_stats.positive_transition_corr, 1.0)
        self.assertIs(item_stats.negative_transition_corr, np.NAN)

    def test_update_2(self):
        schema_stats = SchemaStats()
        item_stats = ERItemStats(schema_stats)

        schema_stats.update(activated=False)
        item_stats.update(item_on=True, activated=False)

        self.assertEqual(item_stats.n_on, 1)
        self.assertEqual(item_stats.n_off, 0)

        self.assertEqual(item_stats.n_on_and_activated, 0)
        self.assertEqual(item_stats.n_on_and_not_activated, 1)
        self.assertEqual(item_stats.n_off_and_activated, 0)
        self.assertEqual(item_stats.n_off_and_not_activated, 0)

        self.assertEqual(item_stats.positive_transition_corr, 0.0)
        self.assertIs(item_stats.negative_transition_corr, np.NAN)

    def test_update_3(self):
        schema_stats = SchemaStats()
        item_stats = ERItemStats(schema_stats)

        schema_stats.update(activated=True)
        item_stats.update(item_on=False, activated=True)

        self.assertEqual(item_stats.n_on, 0)
        self.assertEqual(item_stats.n_off, 1)

        self.assertEqual(item_stats.n_on_and_activated, 0)
        self.assertEqual(item_stats.n_on_and_not_activated, 0)
        self.assertEqual(item_stats.n_off_and_activated, 1)
        self.assertEqual(item_stats.n_off_and_not_activated, 0)

        self.assertIs(item_stats.positive_transition_corr, np.NAN)
        self.assertEqual(item_stats.negative_transition_corr, 1.0)

    def test_update_4(self):
        schema_stats = SchemaStats()
        item_stats = ERItemStats(schema_stats)

        schema_stats.update(activated=False)
        item_stats.update(item_on=False, activated=False)

        self.assertEqual(item_stats.n_on, 0)
        self.assertEqual(item_stats.n_off, 1)

        self.assertEqual(item_stats.n_on_and_activated, 0)
        self.assertEqual(item_stats.n_on_and_not_activated, 0)
        self.assertEqual(item_stats.n_off_and_activated, 0)
        self.assertEqual(item_stats.n_off_and_not_activated, 1)

        self.assertIs(item_stats.positive_transition_corr, np.NAN)
        self.assertEqual(item_stats.negative_transition_corr, 0.0)

    def test_update_5(self):
        # Multiple Updates
        ##################
        schema_stats = SchemaStats()
        item_stats = ERItemStats(schema_stats)

        schema_stats.update(activated=True, count=40)
        schema_stats.update(activated=False, count=10)

        item_stats.update(item_on=True, activated=True, count=32)
        item_stats.update(item_on=True, activated=False, count=2)
        item_stats.update(item_on=False, activated=True, count=8)
        item_stats.update(item_on=False, activated=False, count=8)

        self.assertEqual(item_stats.n_on, 34)
        self.assertEqual(item_stats.n_off, 16)

        self.assertEqual(item_stats.n_on_and_activated, 32)
        self.assertEqual(item_stats.n_on_and_not_activated, 2)
        self.assertEqual(item_stats.n_off_and_activated, 8)
        self.assertEqual(item_stats.n_off_and_not_activated, 8)

        self.assertEqual(item_stats.positive_transition_corr, 0.8)
        self.assertEqual(item_stats.negative_transition_corr, 0.2)


class TestReadOnlyECItemStats(TestCase):
    def test_update(self):
        item_stats = ReadOnlyECItemStats(SchemaStats())
        self.assertRaises(NotImplementedError, lambda: item_stats.update(True, True))


class TestReadOnlyERItemStats(TestCase):
    def test_update(self):
        item_stats = ReadOnlyERItemStats(SchemaStats())
        self.assertRaises(NotImplementedError, lambda: item_stats.update(True, True))
