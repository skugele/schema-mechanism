import unittest
from copy import copy
from random import sample
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import ItemPool
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from test_share.test_func import common_test_setup
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
        _ = self.pool.get('1', primitive_value=-1.0)
        _ = self.pool.get('2', primitive_value=0.0)
        _ = self.pool.get('3', primitive_value=3.0)
        _ = self.pool.get('4', primitive_value=-3.0)

        self.sa = sym_state_assert('1,2,~3,4')
        self.item = CompositeItem(source=self.sa)

    def test_init(self):
        # test: the source should be set properly
        self.assertEqual(self.sa, self.item.source)

        # test: default primitive value should be 0.0
        i = SymbolicItem(source='UNK1')
        self.assertEqual(0.0, i.primitive_value)

    def test_state_elements(self):
        # test: all state elements from positive assertions should be returned
        self.assertSetEqual({'1', '2'}, sym_item('(1,2)').state_elements)
        self.assertSetEqual({'1', '2', '3'}, sym_item('(1,2,3)').state_elements)
        self.assertSetEqual({'1', '2', '3'}, sym_item('(1,2,3,~4)').state_elements)
        self.assertSetEqual(set(), sym_item('(~1,~2,~3,~4)').state_elements)

    def test_primitive_value(self):
        # test: a item's primitive value should equal the primitive value of its source assertion
        self.assertEqual(calc_primitive_value(self.sa), self.item.primitive_value)

    def test_is_on(self):
        # item expected to be ON for these states
        self.assertTrue(self.item.is_on(sym_state('1,2,4')))
        self.assertTrue(self.item.is_on(sym_state('1,2,4,5,6')))

        # item expected to be OFF for these states
        self.assertFalse(self.item.is_on(sym_state('')))
        self.assertFalse(self.item.is_on(sym_state('1,2,3,4')))
        self.assertFalse(self.item.is_on(sym_state('1,4')))
        self.assertFalse(self.item.is_on(sym_state('2,4')))

    def test_equal(self):
        obj = self.item
        other = CompositeItem(sym_state_assert('~7,8,9'))
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
        self.assertTrue(satisfies_equality_checks(obj=self.item, other=CompositeItem(sym_state_assert('~7,8,9'))))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.item))

    def test_equal_with_state_assert(self):
        self.assertEqual(self.item, self.sa)
        self.assertTrue(is_eq_symmetric(x=self.item, y=self.sa))

    def test_item_from_pool(self):
        sa1 = sym_state_assert('2,~3,~4')

        len_before = len(self.pool)
        item1 = self.pool.get(sa1, item_type=CompositeItem)

        self.assertIsInstance(item1, CompositeItem)
        self.assertEqual(sa1, item1.source)
        self.assertEqual(calc_primitive_value(sa1), item1.primitive_value)
        self.assertEqual(len_before + 1, len(self.pool))
        self.assertIn(sa1, self.pool)

        len_before = len(self.pool)
        item2 = self.pool.get(sa1, item_type=CompositeItem)

        self.assertIs(item1, item2)
        self.assertEqual(len_before, len(self.pool))

        sa2 = sym_state_assert('2,3,4')

        len_before = len(self.pool)
        item3 = self.pool.get(sa2, item_type=CompositeItem)

        self.assertNotEqual(item2, item3)
        self.assertEqual(sa2, item3.source)
        self.assertEqual(len_before + 1, len(self.pool))
        self.assertIn(sa2, self.pool)
