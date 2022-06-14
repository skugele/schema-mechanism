import itertools
from copy import deepcopy
from random import sample
from typing import Any
from unittest import TestCase

import test_share
from schema_mechanism.core import ERItemStats
from schema_mechanism.core import ExtendedResult
from schema_mechanism.core import FROZEN_ER_ITEM_STATS
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_ER_ITEM_STATS
from schema_mechanism.core import SchemaSpinOffType
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.serialization.json.decoders import decode
from schema_mechanism.serialization.json.encoders import encode
from schema_mechanism.share import SupportedFeature
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup


class TestExtendedResult(TestCase):
    N_ITEMS = 1000

    def setUp(self) -> None:
        common_test_setup()

        params = get_global_params()
        params.set('ext_result.correlation_test', 'DrescherCorrelationTest')
        params.set('ext_result.positive_correlation_threshold', 0.65)
        params.set('ext_result.negative_correlation_threshold', 0.65)

        # populate item pool
        self._item_pool = ItemPool()
        for i in range(self.N_ITEMS):
            _ = self._item_pool.get(str(i))

        self.result = sym_state_assert('100,101')
        self.er = ExtendedResult(suppressed_items=self.result.items)

        # for convenience, register a single observer
        self.obs = MockObserver()
        self.er.register(self.obs)

    def test_init(self):
        for item in ItemPool():
            self.assertIs(NULL_ER_ITEM_STATS, self.er.stats[item])

        for item in self.result:
            self.assertIn(item, self.er.suppressed_items)

    def test_update(self):
        # testing updates for Items
        state = list(map(str, sample(range(self.N_ITEMS), k=10)))

        # update an item from this state assuming the action was taken
        new_item = sym_item(state[0])
        self.er.update(new_item, on=True, activated=True)

        item_stats = self.er.stats[new_item]
        for i in ItemPool():
            if i != new_item:
                self.assertIs(NULL_ER_ITEM_STATS, self.er.stats[i])
            else:
                self.assertIsNot(NULL_ER_ITEM_STATS, item_stats)

                for value in [item_stats.n_on,
                              item_stats.n_on_and_activated]:
                    self.assertEqual(1, value)

                for value in [item_stats.n_off,
                              item_stats.n_on_and_not_activated,
                              item_stats.n_off_and_activated,
                              item_stats.n_off_and_not_activated]:
                    self.assertEqual(0, value)

    def test_update_with_item_stats_freeze_enabled(self):
        params = get_global_params()

        # this test requires FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION to be enabled
        features = params.get('features')
        features.add(SupportedFeature.FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION)

        item = sym_item('1')

        self.er.update(item=item, on=True, activated=True, count=10)
        self.er.update(item=item, on=False, activated=False, count=10)

        self.assertIn(item, self.er.new_relevant_items)
        self.assertIs(self.er.stats[item], FROZEN_ER_ITEM_STATS)

        stats_before = deepcopy(self.er.stats[item])
        self.er.update(item=item, on=False, activated=False, count=10)
        stats_after = deepcopy(self.er.stats[item])

        # test: no item stats updates should occur after item frozen
        self.assertEqual(stats_before, stats_after)

    @test_share.disable_test
    def test_update_with_no_progress_and_item_stats_freeze_enabled(self):
        params = get_global_params()

        # this test requires FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION to be enabled
        features = params.get('features')
        features.add(SupportedFeature.FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION)

        item = sym_item('1')

        self.er.update(item=item, on=True, activated=True, count=500)
        self.er.update(item=item, on=False, activated=True, count=500)

        self.assertNotIn(item, self.er.new_relevant_items)
        self.assertIs(self.er.stats[item], FROZEN_ER_ITEM_STATS)

        stats_before = deepcopy(self.er.stats[item])
        self.er.update(item=item, on=False, activated=False, count=10)
        stats_after = deepcopy(self.er.stats[item])

        # test: no item stats updates should occur after item frozen
        self.assertEqual(stats_before, stats_after)

    def test_update_all(self):
        # ExtendedResult.update_all (3 cases -> new items, lost items, and new_relevant_items calling notify_all)

        activated_values = [True, False]
        new_items_values = [[sym_item('1'), sym_item('2')], [sym_item('3')]]
        lost_items_values = [[sym_item('4')], [sym_item('5'), sym_item('6')]]

        loop_iterables = itertools.product(
            activated_values,
            new_items_values,
            lost_items_values
        )

        for activated, new_items, lost_items in loop_iterables:
            er_before = deepcopy(self.er)
            self.er.update_all(activated=activated, new=new_items, lost=lost_items)
            er_after = deepcopy(self.er)

            for item in [*new_items, *lost_items]:
                stats_before = er_before.stats[item]
                stats_after = er_after.stats[item]

                before = {
                    'n_on': stats_before.n_on,
                    'n_off': stats_before.n_off,
                    'n_activated': stats_before.n_activated,
                    'n_not_activated': stats_before.n_not_activated,
                    'n_on_and_activated': stats_before.n_on_and_activated,
                    'n_on_and_not_activated': stats_before.n_on_and_not_activated,
                    'n_off_and_activated': stats_before.n_off_and_activated,
                    'n_off_and_not_activated': stats_before.n_off_and_not_activated,
                }
                after = {
                    'n_on': stats_after.n_on,
                    'n_off': stats_after.n_off,
                    'n_activated': stats_after.n_activated,
                    'n_not_activated': stats_after.n_not_activated,
                    'n_on_and_activated': stats_after.n_on_and_activated,
                    'n_on_and_not_activated': stats_after.n_on_and_not_activated,
                    'n_off_and_activated': stats_after.n_off_and_activated,
                    'n_off_and_not_activated': stats_after.n_off_and_not_activated,
                }

                is_on = item in new_items
                is_off = item in lost_items

                if activated:
                    self.assertEqual(before['n_activated'] + 1, after['n_activated'])
                else:
                    self.assertEqual(before['n_not_activated'] + 1, after['n_not_activated'])

                if is_on:
                    self.assertEqual(before['n_on'] + 1, after['n_on'])
                elif is_off:
                    self.assertEqual(before['n_off'] + 1, after['n_off'])

                if is_on and activated:
                    self.assertEqual(before['n_on_and_activated'] + 1, after['n_on_and_activated'])
                elif is_off and activated:
                    self.assertEqual(before['n_off_and_activated'] + 1, after['n_off_and_activated'])
                elif is_on and not activated:
                    self.assertEqual(before['n_on_and_not_activated'] + 1, after['n_on_and_not_activated'])
                elif is_off and not activated:
                    self.assertEqual(before['n_off_and_not_activated'] + 1, after['n_off_and_not_activated'])

    def test_update_all_value_errors(self):
        self.assertRaises(ValueError, lambda: self.er.update_all(
            activated=True, new=[sym_item('1')], lost=[sym_item('1')]))

        self.assertRaises(ValueError, lambda: self.er.update_all(
            activated=True, new=[sym_item('1'), sym_item('2')], lost=[sym_item('2')]))

    def test_notify_all_from_update_all(self):
        self.er.update_all(activated=True, new=[sym_item('1')], lost=[])
        self.er.update_all(activated=False, new=[], lost=[sym_item('1')])

        # test: observer should have been notified of new relevant items
        self.assertTrue(self.obs.n_received > 0)

    def test_register_and_unregister(self):
        observer = MockObserver()
        self.er.register(observer)
        self.assertIn(observer, self.er.observers)

        self.er.unregister(observer)
        self.assertNotIn(observer, self.er.observers)

    def test_relevant_items_1(self):
        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')

        # removing these features to simplify testing (otherwise, stats would revert to zero for every relevant item)
        features.remove(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)
        features.remove(SupportedFeature.FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION)

        items = [sym_item(str(i)) for i in range(5)]

        i1 = items[0]

        # test updates that SHOULD NOT result in relevant item determination
        self.er.update(i1, on=True, activated=False)
        self.assertEqual(0, len(self.er.new_relevant_items))

        self.er.update(i1, on=False, activated=False, count=2)
        self.assertEqual(0, len(self.er.new_relevant_items))

        self.er.update(i1, on=False, activated=True)
        self.assertEqual(0, len(self.er.new_relevant_items))

        self.er.update(i1, on=True, activated=True)
        self.assertEqual(0, len(self.er.new_relevant_items))

        # test update that SHOULD result is a relevant item
        self.er.update(i1, on=True, activated=True)
        self.assertEqual(1, len(self.er.new_relevant_items))

        i1_stats = self.er.stats[i1]

        params = get_global_params()

        pos_threshold = params.get('ext_result.positive_correlation_threshold')
        neg_threshold = params.get('ext_result.negative_correlation_threshold')

        self.assertTrue(i1_stats.positive_correlation_stat > pos_threshold)
        self.assertTrue(i1_stats.negative_correlation_stat <= neg_threshold)

        # verify only one relevant item
        self.assertEqual(1, len(self.er.relevant_items))
        self.assertIn(i1, self.er.relevant_items)

        # should add a 2nd relevant item
        i2 = items[1]

        self.er.update(i2, on=True, activated=True)
        self.er.update(i2, on=False, activated=False)

        self.assertEqual(2, len(self.er.relevant_items))
        self.assertIn(i2, self.er.relevant_items)

        # number of new relevant items SHOULD be reset to zero after notifying observers
        self.er.notify_all(spin_off_type=SchemaSpinOffType.RESULT)
        self.assertEqual(0, len(self.er.new_relevant_items))

    def test_relevant_items_2(self):
        params = get_global_params()
        features: set[SupportedFeature] = params.get('features')

        # removing these features to simplify testing (otherwise, stats would revert to zero for every relevant item)
        features.remove(SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA)
        features.remove(SupportedFeature.FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION)

        i1 = sym_item('100')

        self.assertIn(i1, self.er.suppressed_items)

        # test updates that SHOULD NOT result in relevant item determination
        self.er.update(i1, on=True, activated=True, count=10)
        self.er.update(i1, on=False, activated=False, count=10)

        i1_stats = self.er.stats[i1]

        params = get_global_params()
        pos_threshold = params.get('ext_result.positive_correlation_threshold')
        neg_threshold = params.get('ext_result.negative_correlation_threshold')

        self.assertTrue(i1_stats.positive_correlation_stat > pos_threshold)
        self.assertTrue(i1_stats.negative_correlation_stat <= neg_threshold)

        # verify suppressed item NOT in relevant items list
        self.assertEqual(0, len(self.er.relevant_items))
        self.assertNotIn(i1, self.er.relevant_items)

    def test_encode_and_decode(self):
        extended_result = ExtendedResult(
            suppressed_items=[
                sym_item('A'),
                sym_item('B'),
            ],
            relevant_items=[
                sym_item('C'),
                sym_item('D'),
            ],
            stats={
                sym_item('A'): ERItemStats(
                    n_on_and_activated=1,
                    n_on_and_not_activated=2,
                    n_off_and_activated=3,
                    n_off_and_not_activated=4,
                ),
                sym_item('B'): ERItemStats(
                    n_on_and_activated=100,
                    n_on_and_not_activated=250,
                    n_off_and_activated=65,
                    n_off_and_not_activated=9,
                ),
            }
        )

        object_registry: dict[int, Any] = dict()
        encoded_obj = encode(extended_result, object_registry=object_registry)
        decoded_obj: ExtendedResult = decode(encoded_obj, object_registry=object_registry)

        self.assertEqual(extended_result, decoded_obj)
