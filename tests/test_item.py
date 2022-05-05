import os
import unittest
from copy import copy
from pathlib import Path
from random import sample
from tempfile import TemporaryDirectory
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import ItemPool
from schema_mechanism.core import SymbolicItem
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestSymbolicItem(TestCase):

    def setUp(self) -> None:
        common_test_setup()

        self.item = SymbolicItem(source='1234', primitive_value=1.0)

    def test_init(self):
        # both the state element and the primitive value should be set properly
        self.assertEqual('1234', self.item.source)
        self.assertEqual(1.0, self.item.primitive_value)

        # default primitive value should be 0.0
        i = SymbolicItem(source='1234')
        self.assertEqual(0.0, i.primitive_value)

    def test_state_elements(self):
        self.assertSetEqual({'1234'}, self.item.state_elements)

    def test_primitive_value(self):
        # an item's primitive value should be settable
        self.item.primitive_value = -2.0
        self.assertEqual(-2.0, self.item.primitive_value)

    def test_is_on(self):
        # item expected to be ON for these states
        self.assertTrue(self.item.is_on(sym_state('1234')))
        self.assertTrue(self.item.is_on(sym_state('123,1234')))

        # item expected to be OFF for these states
        self.assertFalse(self.item.is_on(sym_state('')))
        self.assertFalse(self.item.is_on(sym_state('123')))
        self.assertFalse(self.item.is_on(sym_state('123,4321')))

    def test_eq(self):
        self.assertTrue(satisfies_equality_checks(obj=self.item, other=SymbolicItem('123')))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.item))

    def test_serialize(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-symbolic_item-serialize.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.item, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered = deserialize(path)

            self.assertEqual(self.item, recovered)

    @test_share.performance_test
    def test_performance(self):
        n_items = 100_000
        n_runs = 100_000
        n_distinct_states = 100_000_000_000
        n_state_elements = 25

        items = [SymbolicItem(str(value)) for value in range(n_items)]
        state = tuple(sample(range(n_distinct_states), k=n_state_elements))

        start = time()
        for run in range(n_runs):
            _ = map(lambda i: self.item.is_on(state), items)
        end = time()

        print(f'Time for {n_runs * n_items:,} SymbolicItem.is_on calls: {end - start}s')


class TestCompositeItem(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.pool = ItemPool()

        # fill ItemPool
        self.item_1 = self.pool.get('1', primitive_value=-1.0)
        self.item_2 = self.pool.get('2', primitive_value=0.0)
        self.item_3 = self.pool.get('3', primitive_value=3.0)
        self.item_4 = self.pool.get('4', primitive_value=-3.0)

        self.source = frozenset(['1', '2', '3', '4'])
        self.item = CompositeItem(source=self.source)

    def test_init(self):
        # test: the source should be set properly
        self.assertEqual(self.source, self.item.source)

        # test: primitive value should unknown state elements should be 0.0
        item = CompositeItem(source=['UNK1', 'UNK2'])
        self.assertEqual(0.0, item.primitive_value)

        # test: by default, the primitive value should be the sum of item primitive values corresponding to known
        #     : state elements
        item = CompositeItem(source=['1', '2'])
        self.assertEqual(sum(item.primitive_value for item in [self.item_1, self.item_2]), item.primitive_value)

        # test: when a primitive value is supplied to initializer, it should be properly set and override the
        #     : primitive value of the underlying state assertion
        primitive_value = -100.0
        item = CompositeItem(source=self.source, primitive_value=primitive_value)
        self.assertEqual(primitive_value, item.primitive_value)

    def test_state_elements(self):
        # test: all state elements should be returned
        self.assertSetEqual({'1', '2'}, sym_item('(1,2)').state_elements)
        self.assertSetEqual({'1', '2', '3'}, sym_item('(1,2,3)').state_elements)

    def test_is_on(self):
        # item expected to be ON for these states
        self.assertTrue(self.item.is_on(sym_state('1,2,3,4')))
        self.assertTrue(self.item.is_on(sym_state('1,2,3,4,5,6')))

        # item expected to be OFF for these states
        self.assertFalse(self.item.is_on(sym_state('')))
        self.assertFalse(self.item.is_on(sym_state('1,2,3')))
        self.assertFalse(self.item.is_on(sym_state('1,4')))
        self.assertFalse(self.item.is_on(sym_state('2,4')))

    def test_equal(self):
        obj = self.item
        other = CompositeItem(source=['A', 'B'])
        copy_ = copy(obj)

        self.assertEqual(obj, obj)
        self.assertEqual(obj, copy_)
        self.assertNotEqual(obj, other)

        self.assertTrue(is_eq_reflexive(obj))
        self.assertTrue(is_eq_symmetric(x=obj, y=copy_))
        self.assertTrue(is_eq_transitive(x=obj, y=copy_, z=copy(copy_)))
        self.assertTrue(is_eq_consistent(x=obj, y=copy_))
        self.assertTrue(is_eq_with_null_is_false(obj))

    def test_eq(self):
        self.assertTrue(satisfies_equality_checks(obj=self.item, other=CompositeItem(source=['A', 'B'])))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.item))

    def test_item_from_pool(self):
        source_1 = frozenset(['2', '3', '4'])

        len_before = len(self.pool)
        item_1 = self.pool.get(source_1)

        # test: item provided by item pool should have correct instance type of CompositeItem
        self.assertIsInstance(item_1, CompositeItem)

        # test: item provided by item pool should have a source containing all requested state elements
        self.assertEqual(source_1, item_1.source)

        # test: item provided by item pool should have a primitive value equal to sum of state elements' items' values
        expected_primitive_value = sum(self.pool.get(se).primitive_value for se in source_1)
        self.assertEqual(expected_primitive_value, item_1.primitive_value)

        # test: the pool's size should have been increased by one
        self.assertEqual(len_before + 1, len(self.pool))

        # test: the requested source (set of state elements) should now exist in the pool
        self.assertIn(source_1, self.pool)

        len_before = len(self.pool)
        item_2 = self.pool.get(source_1)

        # test: identical item should be retrieved from pool when given the same source again
        self.assertIs(item_1, item_2)

        # test: pool size should not increase when retrieving a previously gotten item
        self.assertEqual(len_before, len(self.pool))

        source_2 = frozenset(['1', '2', '3'])

        len_before = len(self.pool)
        item_3 = self.pool.get(source_2)

        # test: requesting an item from a different source should return a different item
        self.assertNotEqual(item_2, item_3)

        # test: item provided by item pool should have a source containing all requested state elements
        self.assertEqual(source_2, item_3.source)

        # test: the pool's size should have been increased by one
        self.assertEqual(len_before + 1, len(self.pool))

        # test: the requested source (set of state elements) should now exist in the pool
        self.assertIn(source_2, self.pool)

    def test_serialize(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-composite_item-serialize.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.item, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered = deserialize(path)

            self.assertEqual(self.item, recovered)
