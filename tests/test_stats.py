from unittest import TestCase
from unittest import TestCase
from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import FrozenItemStatisticsDecorator
from schema_mechanism.data_structures import GlobalStatistics
from schema_mechanism.data_structures import ItemStatistics


class TestItemStatistics(TestCase):
    def setUp(self) -> None:
        GlobalStatistics().reset()

    def test_init(self):
        item_stats = ItemStatistics()

        self.assertEqual(item_stats.n_on_with_action, 0)
        self.assertEqual(item_stats.n_on_without_action, 0)
        self.assertEqual(item_stats.n_off_with_action, 0)
        self.assertEqual(item_stats.n_off_without_action, 0)

        self.assertIs(item_stats.positive_transition_corr, np.NAN)
        self.assertIs(item_stats.negative_transition_corr, np.NAN)

    def test_update_1(self):
        # Single Update - All Base Cases
        ################################
        item_stats = ItemStatistics()
        global_stats = GlobalStatistics()

        item_stats.update(item_on=True, action_taken=True)
        global_stats.n += 1

        self.assertEqual(item_stats.n_on, 1)
        self.assertEqual(item_stats.n_off, 0)
        self.assertEqual(item_stats.n_action, 1)
        self.assertEqual(item_stats.n_not_action, 0)

        self.assertEqual(item_stats.n_on_with_action, 1)
        self.assertEqual(item_stats.n_on_without_action, 0)
        self.assertEqual(item_stats.n_off_with_action, 0)
        self.assertEqual(item_stats.n_off_without_action, 0)
        self.assertEqual(item_stats.positive_transition_corr, 1.0)
        self.assertIs(item_stats.negative_transition_corr, np.NAN)

    def test_update_2(self):
        item_stats = ItemStatistics()
        global_stats = GlobalStatistics()

        item_stats.update(item_on=True, action_taken=False)
        global_stats.n += 1

        self.assertEqual(item_stats.n_on, 1)
        self.assertEqual(item_stats.n_off, 0)
        self.assertEqual(item_stats.n_action, 0)
        self.assertEqual(item_stats.n_not_action, 1)

        self.assertEqual(item_stats.n_on_with_action, 0)
        self.assertEqual(item_stats.n_on_without_action, 1)
        self.assertEqual(item_stats.n_off_with_action, 0)
        self.assertEqual(item_stats.n_off_without_action, 0)
        self.assertEqual(item_stats.positive_transition_corr, 0.0)
        self.assertIs(item_stats.negative_transition_corr, np.NAN)

    def test_update_3(self):
        item_stats = ItemStatistics()
        global_stats = GlobalStatistics()

        item_stats.update(item_on=False, action_taken=True)
        global_stats.n += 1

        self.assertEqual(item_stats.n_on, 0)
        self.assertEqual(item_stats.n_off, 1)
        self.assertEqual(item_stats.n_action, 1)
        self.assertEqual(item_stats.n_not_action, 0)

        self.assertEqual(item_stats.n_on_with_action, 0)
        self.assertEqual(item_stats.n_on_without_action, 0)
        self.assertEqual(item_stats.n_off_with_action, 1)
        self.assertEqual(item_stats.n_off_without_action, 0)
        self.assertIs(item_stats.positive_transition_corr, np.NAN)
        self.assertEqual(item_stats.negative_transition_corr, 1.0)

    def test_update_4(self):
        item_stats = ItemStatistics()
        global_stats = GlobalStatistics()

        item_stats.update(item_on=False, action_taken=False)
        global_stats.n += 1

        self.assertEqual(item_stats.n_on, 0)
        self.assertEqual(item_stats.n_off, 1)
        self.assertEqual(item_stats.n_action, 0)
        self.assertEqual(item_stats.n_not_action, 1)

        self.assertEqual(item_stats.n_on_with_action, 0)
        self.assertEqual(item_stats.n_on_without_action, 0)
        self.assertEqual(item_stats.n_off_with_action, 0)
        self.assertEqual(item_stats.n_off_without_action, 1)
        self.assertIs(item_stats.positive_transition_corr, np.NAN)
        self.assertEqual(item_stats.negative_transition_corr, 0.0)

    def test_update_5(self):
        # Multiple Updates
        ##################
        item_stats = ItemStatistics()
        global_stats = GlobalStatistics()

        item_stats.update(item_on=True, action_taken=True, count=32)
        item_stats.update(item_on=True, action_taken=False, count=2)
        item_stats.update(item_on=False, action_taken=True, count=8)
        item_stats.update(item_on=False, action_taken=False, count=8)
        global_stats.n += 50

        self.assertEqual(item_stats.n_on, 34)
        self.assertEqual(item_stats.n_off, 16)
        self.assertEqual(item_stats.n_action, 40)
        self.assertEqual(item_stats.n_not_action, 10)

        self.assertEqual(item_stats.n_on_with_action, 32)
        self.assertEqual(item_stats.n_on_without_action, 2)
        self.assertEqual(item_stats.n_off_with_action, 8)
        self.assertEqual(item_stats.n_off_without_action, 8)

        self.assertEqual(item_stats.positive_transition_corr, 0.8)
        self.assertEqual(item_stats.negative_transition_corr, 0.2)


class TestFrozenItemStatisticsDecorator(TestCase):
    def setUp(self) -> None:
        GlobalStatistics().reset()

    def test_update(self):
        stats = ItemStatistics()
        global_stats = GlobalStatistics()

        stats.update(item_on=False, action_taken=False)
        stats.update(item_on=False, action_taken=True)
        stats.update(item_on=True, action_taken=False)
        stats.update(item_on=True, action_taken=True)
        global_stats.n += 4

        frozen_stats = FrozenItemStatisticsDecorator(stats)
        self.assertEqual(frozen_stats.n_on, 2)
        self.assertEqual(frozen_stats.n_off, 2)
        self.assertEqual(frozen_stats.n_action, 2)
        self.assertEqual(frozen_stats.n_not_action, 2)

        self.assertRaises(NotImplementedError, lambda: frozen_stats.update(True, True))


class TestGlobalStatistics(TestCase):
    def setUp(self) -> None:
        GlobalStatistics().reset()

    def test_init(self):
        global_stats = GlobalStatistics()
        self.assertEqual(0, global_stats.n)

    def test_property_setters(self):
        global_stats = GlobalStatistics()
        global_stats.n += 1

        self.assertEqual(1, global_stats.n)
