import itertools
import os
from copy import deepcopy
from pathlib import Path
from random import sample
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np

from schema_mechanism.core import ExtendedContext
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_EC_ITEM_STATS
from schema_mechanism.core import SchemaSpinOffType
from schema_mechanism.core import SupportedFeature
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_items
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.correlation_test import DrescherCorrelationTest
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written


class TestExtendedContext(TestCase):
    N_ITEMS = 1000

    def setUp(self) -> None:
        common_test_setup()

        params = get_global_params()

        params.set('ext_context.correlation_test', DrescherCorrelationTest())
        params.set('ext_context.positive_correlation_threshold', 0.65)
        params.set('ext_context.negative_correlation_threshold', 0.65)

        self.pool = ItemPool()

        for i in range(self.N_ITEMS):
            self.pool.get(str(i), item_type=SymbolicItem)

        self.context = sym_state_assert('100,101')
        self.ec = ExtendedContext(suppressed_items=self.context.items)

        self.obs = MockObserver()
        self.ec.register(self.obs)

    def test_init(self):
        ec = ExtendedContext(suppressed_items=self.context.items)

        # test: stats for all items in item pool should start as NULL_EC_ITEM_STATS
        for item in self.pool:
            self.assertIs(NULL_EC_ITEM_STATS, ec.stats[item])

        new_item = self.pool.get('NEW_ITEM')

        # test: if a new item is added to the item pool it should also have a value of NULL_EC_ITEM_STATS
        self.assertIs(NULL_EC_ITEM_STATS, ec.stats[new_item])

        # test: verify that suppressed items were set properly
        for item in self.context:
            self.assertIn(item, ec.suppressed_items)

        # test: initial pending max specificity should be -np.inf
        self.assertEqual(-np.inf, ec.pending_max_specificity)

    def test_update(self):
        state = list(map(str, sample(range(self.N_ITEMS), k=10)))

        # update an item from this state assuming the action was taken
        new_item = sym_item(state[0])
        self.ec.update(new_item, on=True, success=True)

        item_stats = self.ec.stats[new_item]
        for i in ItemPool():
            if i.source != state[0]:
                self.assertIs(NULL_EC_ITEM_STATS, self.ec.stats[i])
            else:
                self.assertIsNot(NULL_EC_ITEM_STATS, item_stats)

                for value in [item_stats.n_on,
                              item_stats.n_success_and_on]:
                    self.assertEqual(1, value)

                for value in [item_stats.n_off,
                              item_stats.n_success_and_off,
                              item_stats.n_fail_and_on,
                              item_stats.n_fail_and_off]:
                    self.assertEqual(0, value)

    def test_update_all_1(self):
        state = tuple(sample(range(self.N_ITEMS), k=10))

        # update all items in this state simulating case of action taken
        self.ec.update_all(state, success=True)

        # test that all items in the state have been updated
        for i in ItemPool():
            item_stats = self.ec.stats[i]

            if i.source in state:
                self.assertEqual(1, item_stats.n_on)
                self.assertEqual(1, item_stats.n_success_and_on)

                self.assertEqual(0, item_stats.n_off)
                self.assertEqual(0, item_stats.n_success_and_off)

                self.assertEqual(0, item_stats.n_fail_and_on)
                self.assertEqual(0, item_stats.n_fail_and_off)
            else:

                self.assertEqual(0, item_stats.n_on)
                self.assertEqual(0, item_stats.n_success_and_on)

                self.assertEqual(1, item_stats.n_off)
                self.assertEqual(1, item_stats.n_success_and_off)

                self.assertEqual(0, item_stats.n_fail_and_on)
                self.assertEqual(0, item_stats.n_fail_and_off)

    def test_update_all_2(self):
        params = get_global_params()
        features = params.get('features')

        # test update_all without SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE enabled
        if SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE in features:
            features.remove(SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE)

        self.ec.update_all(sym_state('3'), success=False)
        self.ec.update_all(sym_state('2'), success=False, count=5)

        self.assertEqual(0, len(self.ec.relevant_items))

        # three pending relevant items from this update:
        # [1 -> SC=1.0; FC=0.0]   [2 -> SC=1.0; FC=0.25]   [3 -> SC=0.0; FC=0.75]
        self.ec.update_all(sym_state('1,2'), success=True, count=10)

        # test: 2 pending items (items 1 and 2) should be relevant items in extended context
        self.assertEqual(2, len(self.ec.relevant_items))
        self.assertSetEqual(set(sym_items('1;2')), self.ec.relevant_items)

    def test_update_all_without_ec_defer_to_more_specific(self):
        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')
        features.remove(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)

        success_values = [True, False]
        selection_state_values = [
            sym_state('1'),
            sym_state('2'),
            sym_state('1,2'),
            sym_state('2,3'),
            sym_state('1,2,3')
        ]

        loop_iterables = itertools.product(
            success_values,
            selection_state_values,
        )

        for success, selection_state in loop_iterables:
            ec_before = deepcopy(self.ec)
            self.ec.update_all(success=success, selection_state=selection_state)
            ec_after = deepcopy(self.ec)

            for item in ItemPool():
                stats_before = ec_before.stats[item]
                stats_after = ec_after.stats[item]

                before = {
                    'n': stats_before.n,
                    'n_on': stats_before.n_on,
                    'n_off': stats_before.n_off,
                    'n_success_and_on': stats_before.n_success_and_on,
                    'n_success_and_off': stats_before.n_success_and_off,
                    'n_fail_and_on': stats_before.n_fail_and_on,
                    'n_fail_and_off': stats_before.n_fail_and_off,
                    'n_success': stats_before.n_success,
                    'n_fail': stats_before.n_fail,
                }

                after = {
                    'n': stats_after.n,
                    'n_on': stats_after.n_on,
                    'n_off': stats_after.n_off,
                    'n_success_and_on': stats_after.n_success_and_on,
                    'n_success_and_off': stats_after.n_success_and_off,
                    'n_fail_and_on': stats_after.n_fail_and_on,
                    'n_fail_and_off': stats_after.n_fail_and_off,
                    'n_success': stats_after.n_success,
                    'n_fail': stats_after.n_fail,
                }

                is_on = item.is_on(selection_state)
                is_off = item.is_off(selection_state)

                if success:
                    self.assertEqual(before['n_success'] + 1, after['n_success'])
                else:
                    self.assertEqual(before['n_fail'] + 1, after['n_fail'])

                if is_on:
                    self.assertEqual(before['n_on'] + 1, after['n_on'])
                elif is_off:
                    self.assertEqual(before['n_off'] + 1, after['n_off'])

                if is_on and success:
                    self.assertEqual(before['n_success_and_on'] + 1, after['n_success_and_on'])
                elif is_off and success:
                    self.assertEqual(before['n_success_and_off'] + 1, after['n_success_and_off'])
                elif is_on and not success:
                    self.assertEqual(before['n_fail_and_on'] + 1, after['n_fail_and_on'])
                elif is_off and not success:
                    self.assertEqual(before['n_fail_and_off'] + 1, after['n_fail_and_off'])

    def test_update_all_with_ec_defer_to_more_specific(self):
        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')
        features.add(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)

        success_values = [True, False]
        selection_state_values = [
            sym_state('1'),
            sym_state('2'),
            sym_state('1,2'),
            sym_state('2,3'),
            sym_state('1,2,3')
        ]

        loop_iterables = itertools.product(
            success_values,
            selection_state_values,
        )

        for success, selection_state in loop_iterables:
            ec_before = deepcopy(self.ec)
            defer_to_spinoff = ec_before.defer_update_to_spin_offs(selection_state)

            self.ec.update_all(success=success, selection_state=selection_state)
            ec_after = deepcopy(self.ec)

            new_relevant_items = ec_after.relevant_items.difference(ec_before.relevant_items)

            for item in ItemPool():
                stats_before = ec_before.stats[item]
                stats_after = ec_after.stats[item]

                before = {
                    'n': stats_before.n,
                    'n_on': stats_before.n_on,
                    'n_off': stats_before.n_off,
                    'n_success_and_on': stats_before.n_success_and_on,
                    'n_success_and_off': stats_before.n_success_and_off,
                    'n_fail_and_on': stats_before.n_fail_and_on,
                    'n_fail_and_off': stats_before.n_fail_and_off,
                    'n_success': stats_before.n_success,
                    'n_fail': stats_before.n_fail,
                }

                after = {
                    'n': stats_after.n,
                    'n_on': stats_after.n_on,
                    'n_off': stats_after.n_off,
                    'n_success_and_on': stats_after.n_success_and_on,
                    'n_success_and_off': stats_after.n_success_and_off,
                    'n_fail_and_on': stats_after.n_fail_and_on,
                    'n_fail_and_off': stats_after.n_fail_and_off,
                    'n_success': stats_after.n_success,
                    'n_fail': stats_after.n_fail,
                }

                # test: if there was a new relevant item identified, then the values of ALL stats in ALL extended
                #     : context slots should be reset to zero
                if new_relevant_items:
                    self.assertTrue(all((value == 0 for value in after.values())))
                    break

                is_on = item.is_on(selection_state)
                is_off = item.is_off(selection_state)

                # test: if updates have been deferred to the spinoff schemas then the before and after stats should be
                #     : equal
                if defer_to_spinoff:
                    self.assertDictEqual(before, after)
                    break

                if success:
                    self.assertEqual(before['n_success'] + 1, after['n_success'])
                else:
                    self.assertEqual(before['n_fail'] + 1, after['n_fail'])

                if is_on:
                    self.assertEqual(before['n_on'] + 1, after['n_on'])
                elif is_off:
                    self.assertEqual(before['n_off'] + 1, after['n_off'])

                if is_on and success:
                    self.assertEqual(before['n_success_and_on'] + 1, after['n_success_and_on'])
                elif is_off and success:
                    self.assertEqual(before['n_success_and_off'] + 1, after['n_success_and_off'])
                elif is_on and not success:
                    self.assertEqual(before['n_fail_and_on'] + 1, after['n_fail_and_on'])
                elif is_off and not success:
                    self.assertEqual(before['n_fail_and_off'] + 1, after['n_fail_and_off'])

    def test_notify_all_from_update_all(self):
        self.ec.update_all(success=True, selection_state=sym_state('1'))
        self.ec.update_all(success=False, selection_state=sym_state('2'))

        # test: observer should have been notified of new relevant items
        self.assertTrue(self.obs.n_received > 0)

    def test_defer_update_to_spin_offs(self):
        # SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA enabled
        GlobalParams().get('features').add(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)

        update_state_1 = sym_state('4,6,8')
        update_state_2 = sym_state('3,6')
        defer_state_1 = sym_state('1,5')
        defer_state_2 = sym_state('1,7,8')

        with patch(target='schema_mechanism.core.ExtendedContext.relevant_items',
                   new_callable=PropertyMock) as mock_relevant_items:
            mock_relevant_items.return_value = set(sym_items('5;7'))
            ec = ExtendedContext(suppressed_items=sym_items('1;2'))

            # test: update should NOT be deferred if relevant items do not match state
            self.assertFalse(ec.defer_update_to_spin_offs(update_state_1))
            self.assertFalse(ec.defer_update_to_spin_offs(update_state_2))

            # test: relevant items should defer updates when satisfied
            self.assertTrue(ec.defer_update_to_spin_offs(defer_state_1))
            self.assertTrue(ec.defer_update_to_spin_offs(defer_state_2))

    def test_pending_max_specificity(self):
        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')

        # the EC_MOST_SPECIFIC_ON_MULTIPLE feature must be enabled for this test
        features.add(SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE)

        # removing this feature to simplify testing (otherwise, stats would revert to zero for every relevant item)
        features.remove(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)

        item_1, item_2 = sym_item('1'), sym_item('2')

        # updating item 1 stats for testing
        self.ec.update(item_1, on=True, success=True)
        self.ec.update(item_1, on=False, success=False, count=9)

        # updating item 2 stats for testing
        self.ec.update(item_2, on=True, success=True)
        self.ec.update(item_2, on=False, success=False, count=4)

        item_1_specificity, item_2_specificity = self.ec.stats[item_1].specificity, self.ec.stats[item_2].specificity
        self.assertEqual(max(item_1_specificity, item_2_specificity), self.ec.pending_max_specificity)

    def test_clear_pending_relevant_items(self):
        pass

    def test_register_and_unregister(self):
        observer = MockObserver()
        self.ec.register(observer)
        self.assertIn(observer, self.ec.observers)

        self.ec.unregister(observer)
        self.assertNotIn(observer, self.ec.observers)

    def test_relevant_items_1(self):
        items = [sym_item(str(i)) for i in range(5)]

        i1 = items[0]

        # test updates that SHOULD NOT result in relevant item determination
        self.ec.update(i1, on=False, success=True)

        self.assertEqual(0, len(self.ec.pending_relevant_items))
        self.ec.update(i1, on=False, success=False, count=2)
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        self.ec.update(i1, on=True, success=False)
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        self.ec.update(i1, on=True, success=True)
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        # test update that SHOULD result in a relevant item
        self.ec.update(i1, on=True, success=True)

        i1_stats = self.ec.stats[i1]

        params = get_global_params()
        pos_threshold = params.get('ext_context.positive_correlation_threshold')
        neg_threshold = params.get('ext_context.negative_correlation_threshold')

        self.assertTrue(i1_stats.positive_correlation_stat > pos_threshold)
        self.assertTrue(i1_stats.negative_correlation_stat <= neg_threshold)

        self.assertEqual(1, len(self.ec.pending_relevant_items))
        self.ec.check_pending_relevant_items()
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        # verify only one relevant item
        self.assertEqual(1, len(self.ec.relevant_items))
        self.assertIn(i1, self.ec.relevant_items)

        # should add a 2nd relevant item
        i2 = items[1]

        self.ec.update(i2, on=True, success=True)
        self.ec.update(i2, on=False, success=False)

        self.assertEqual(1, len(self.ec.pending_relevant_items))
        self.ec.check_pending_relevant_items()
        self.assertEqual(0, len(self.ec.pending_relevant_items))

        self.assertEqual(2, len(self.ec.relevant_items))
        self.assertIn(i2, self.ec.relevant_items)

        # number of new relevant items SHOULD be reset to zero after notifying observers
        self.ec.notify_all(spin_off_type=SchemaSpinOffType.CONTEXT)
        self.assertEqual(0, len(self.ec.new_relevant_items))

    def test_relevant_items_2(self):
        i1 = sym_item('100')

        self.assertIn(i1, self.ec.suppressed_items)

        # test updates that SHOULD NOT result in relevant item determination
        self.ec.update(i1, on=True, success=True, count=10)
        self.ec.update(i1, on=False, success=False, count=10)

        i1_stats = self.ec.stats[i1]

        pos_threshold = GlobalParams().get('ext_context.positive_correlation_threshold')
        neg_threshold = GlobalParams().get('ext_context.negative_correlation_threshold')

        self.assertTrue(i1_stats.positive_correlation_stat > pos_threshold)
        self.assertTrue(i1_stats.negative_correlation_stat <= neg_threshold)

        # verify suppressed item NOT in relevant items list
        self.assertEqual(0, len(self.ec.pending_relevant_items))
        self.assertEqual(0, len(self.ec.relevant_items))
        self.assertNotIn(i1, self.ec.relevant_items)

    def test_serialize(self):
        # update extended context before serialize
        items = [sym_item(str(i)) for i in range(10)]

        for item in items:
            self.ec.update(item, on=False, success=True, count=100)
            self.ec.update(item, on=False, success=False, count=200)
            self.ec.update(item, on=True, success=False, count=50)
            self.ec.update(item, on=True, success=True, count=100)

        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-extended_context-serialize.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.ec, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered = deserialize(path)

            self.assertEqual(self.ec, recovered)
