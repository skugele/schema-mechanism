import unittest
from copy import copy
from random import sample
from time import time
from typing import Optional
from unittest import TestCase

import test_share
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import EligibilityTraceDelegatedValueHelper
from schema_mechanism.core import Item
from schema_mechanism.core import ItemPool
from schema_mechanism.core import State
from schema_mechanism.core import StateElement
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import get_global_params
from schema_mechanism.core import set_delegated_value_helper
from schema_mechanism.func_api import sym_composite_item
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.serialization.json.decoders import decode
from schema_mechanism.serialization.json.encoders import encode
from schema_mechanism.strategies.decay import NoDecayStrategy
from schema_mechanism.strategies.trace import ReplacingTrace
from schema_mechanism.util import repr_str
from test_share.test_func import common_test_setup
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestItem(TestCase):
    class IntItem(Item):
        """ A simple, concrete Item implementation used for testing Item's methods. """

        def __init__(
                self,
                source: int,
                primitive_value: Optional[float] = 0.0,
                delegated_value: Optional[float] = 0.0,
        ) -> None:
            super().__init__(
                source=source,
                primitive_value=primitive_value,
                delegated_value=delegated_value,
            )

        def __eq__(self, other) -> bool:
            return self.source == other.source

        def __hash__(self) -> int:
            return hash(self.source)

        def is_on(self, state: State, **kwargs) -> bool:
            return self.source in state

        @property
        def state_elements(self) -> set[StateElement]:
            return {self.source}

    def setUp(self) -> None:
        common_test_setup()

        self.pool: ItemPool = ItemPool()

        self.item_2 = self.pool.get(2, item_type=TestItem.IntItem, primitive_value=10.0)

    def test_init(self):
        source = 1
        primitive_value = 17.2
        delegated_value = -4.2

        item = self.pool.get(
            source,
            item_type=TestItem.IntItem,
            primitive_value=primitive_value,
            delegated_value=delegated_value)

        # test: attributes should have been set correctly by Item's initializer
        self.assertEqual(source, item.source)
        self.assertEqual(primitive_value, item.primitive_value)
        self.assertEqual(delegated_value, item.delegated_value)

    def test_str(self):
        item = TestItem.IntItem(source=1, primitive_value=5.0)
        item_str = str(item)

        self.assertEqual(str(item.source), item_str)

    def test_repr(self):
        item = TestItem.IntItem(source=1, primitive_value=5.0)
        item_repr = repr(item)

        expected_str = repr_str(item, {'source': str(item.source),
                                       'pv': item.primitive_value,
                                       'dv': item.delegated_value})

        self.assertEqual(expected_str, item_repr)

    def test_update_delegated_value(self):
        # test: verify that delegated value updates when delegated value helper's state is updated
        item = self.pool.get(
            1,
            item_type=TestItem.IntItem,
            primitive_value=0.0,
            delegated_value=0.0)

        # setting learning rate to 1.0 to simplify testing
        params = get_global_params()
        params.set('learning_rate', 1.0)

        # no discount or decay in this instance, so delegated value updates will be equal to the primitive values of
        # new result state items
        delegated_value_helper = EligibilityTraceDelegatedValueHelper(
            discount_factor=1.0,
            eligibility_trace=ReplacingTrace(
                decay_strategy=NoDecayStrategy())
        )
        delegated_value_helper.update(selection_state=(1,), result_state=(2,))
        set_delegated_value_helper(delegated_value_helper)

        # test: if delegated value helper were set properly, the item's delegated value should be equal to item 2's
        #       primitive value
        self.assertEqual(self.item_2.primitive_value, item.delegated_value)


class TestSymbolicItem(TestCase):

    def setUp(self) -> None:
        common_test_setup()

        self.item = SymbolicItem(source='1234', primitive_value=1.0, delegated_value=0.1)

    # noinspection PyTypeChecker
    def test_init(self):
        source = '1234'
        primitive_value = 1.0
        delegated_value = 0.6

        # test: attributes should be set properly when argument passed to initializer
        item = SymbolicItem(source=source, primitive_value=primitive_value, delegated_value=delegated_value)
        self.assertEqual(source, item.source)
        self.assertEqual(primitive_value, item.primitive_value)
        self.assertEqual(delegated_value, item.delegated_value)

        i = SymbolicItem(source='1234')

        # test: default primitive value should be 0.0
        self.assertEqual(0.0, i.primitive_value)

        # test: default delegated value should be 0.0
        self.assertEqual(0.0, i.delegated_value)

        # test: ValueError raised when source is not a str)
        self.assertRaises(ValueError, lambda: SymbolicItem(source=1.0))
        self.assertRaises(ValueError, lambda: SymbolicItem(source=frozenset(('1', '2'))))
        self.assertRaises(ValueError, lambda: SymbolicItem(source=[]))

    def test_state_elements(self):
        self.assertSetEqual({'1234'}, self.item.state_elements)

    def test_is_on(self):
        # test: items should be On for these states
        self.assertTrue(self.item.is_on(sym_state('1234')))
        self.assertTrue(self.item.is_on(sym_state('123,1234')))

        # test: items should be Off for these states
        self.assertFalse(self.item.is_on(sym_state('')))
        self.assertFalse(self.item.is_on(sym_state('123')))
        self.assertFalse(self.item.is_on(sym_state('123,4321')))

    def test_is_off(self):
        # test: items should be Off for these states
        self.assertTrue(self.item.is_off(sym_state('')))
        self.assertTrue(self.item.is_off(sym_state('123')))
        self.assertTrue(self.item.is_off(sym_state('123,4321')))

        # test: items should be On for these states
        self.assertFalse(self.item.is_off(sym_state('1234')))
        self.assertFalse(self.item.is_off(sym_state('123,1234')))

    def test_update_delegated_value(self):
        # test: verify that delegated value updates when delegated value helper's state is updated
        item_1 = sym_item(
            '1',
            item_type=SymbolicItem,
            primitive_value=0.0,
            delegated_value=6.0
        )

        item_2 = sym_item(
            '2',
            item_type=SymbolicItem,
            primitive_value=10.0,
            delegated_value=0.0
        )

        # sanity check: delegated values should have been initialized
        self.assertEqual(6.0, item_1.delegated_value)
        self.assertEqual(10.0, item_2.primitive_value)

        # setting learning rate to 1.0 to simplify testing
        params = get_global_params()
        params.set('learning_rate', 1.0)

        # no discount or decay in this instance, so delegated value updates will be equal to the primitive values of
        # new result state items
        delegated_value_helper = EligibilityTraceDelegatedValueHelper(
            discount_factor=1.0,
            eligibility_trace=ReplacingTrace(
                decay_strategy=NoDecayStrategy())
        )
        delegated_value_helper.update(selection_state=('1',), result_state=('2',))
        set_delegated_value_helper(delegated_value_helper)

        # test: if delegated value helper were set properly, the item's delegated value should be equal to item 2's
        #       primitive value
        self.assertEqual(item_2.primitive_value, item_1.delegated_value)

    def test_eq(self):
        self.assertTrue(satisfies_equality_checks(
            obj=self.item,
            other=SymbolicItem('123'),
            other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.item))

    def test_encode_and_decode(self):
        encoded_obj = encode(self.item)
        decoded_obj: SymbolicItem = decode(encoded_obj)

        self.assertEqual(self.item, decoded_obj)
        self.assertEqual(self.item.primitive_value, decoded_obj.primitive_value)
        self.assertEqual(self.item.delegated_value, decoded_obj.delegated_value)

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
        self.item_1 = self.pool.get('1', primitive_value=-1.0, delegated_value=0.25)
        self.item_2 = self.pool.get('2', primitive_value=0.0, delegated_value=-0.6)
        self.item_3 = self.pool.get('3', primitive_value=3.0)
        self.item_4 = self.pool.get('4', primitive_value=-3.0)

        self.state_elements = ['1', '2', '3', '4']
        self.source = frozenset(self.state_elements)
        self.item = CompositeItem(source=self.source)

    # noinspection PyTypeChecker
    def test_init(self):
        source = ['1', '2', '3', '4']
        primitive_value = 1.0
        delegated_value = 0.6

        # test: the attributes should be set properly
        item = CompositeItem(source=source, primitive_value=primitive_value, delegated_value=delegated_value)
        self.assertSetEqual(set(source), set(item.source))
        self.assertEqual(primitive_value, item.primitive_value)
        self.assertEqual(delegated_value, item.delegated_value)

        # test: primitive value should unknown state elements should be 0.0
        item = CompositeItem(source=['UNK1', 'UNK2'])
        self.assertEqual(0.0, item.primitive_value)

        item = CompositeItem(source=['1', '2'])

        # test: by default, the primitive value should be the sum of item primitive values corresponding to known
        #     : state elements
        self.assertEqual(sum(item.primitive_value for item in [self.item_1, self.item_2]), item.primitive_value)

        # test: by default, the delegated value should be the sum of item delegated values corresponding to known
        #     : state elements
        self.assertEqual(sum(item.delegated_value for item in [self.item_1, self.item_2]), item.delegated_value)

        # test: TypeError raised when source is not iterable
        self.assertRaises(TypeError, lambda: CompositeItem(source=1.0))
        self.assertRaises(TypeError, lambda: CompositeItem(source=None))

        # test: ValueError raised when source is an iterable with less than 2 elements
        self.assertRaises(ValueError, lambda: CompositeItem(source=[]))
        self.assertRaises(ValueError, lambda: CompositeItem(source=['1']))

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

    def test_is_off(self):
        # item expected to be ON for these states
        self.assertFalse(self.item.is_off(sym_state('1,2,3,4')))
        self.assertFalse(self.item.is_off(sym_state('1,2,3,4,5,6')))

        # item expected to be OFF for these states
        self.assertTrue(self.item.is_off(sym_state('')))
        self.assertTrue(self.item.is_off(sym_state('1,2,3')))
        self.assertTrue(self.item.is_off(sym_state('1,4')))
        self.assertTrue(self.item.is_off(sym_state('2,4')))

    def test_contains(self):
        # test: all of these state elements SHOULD be contained in this composite item
        for state_element in self.state_elements:
            self.assertIn(state_element, self.item)

        # test: all of these state elements SHOULD NOT be contained in this composite item
        other_state_elements = set(range(100)).difference(self.source)
        for state_element in other_state_elements:
            self.assertNotIn(state_element, self.item)

        # test: contains should always return False when parameter is not a state element
        self.assertNotIn(set(), self.item)
        self.assertNotIn(list(), self.item)
        self.assertNotIn(dict(), self.item)

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
        self.assertTrue(satisfies_equality_checks(
            obj=self.item,
            other=CompositeItem(source=['A', 'B']),
            other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.item))

    def test_str(self):
        expected_str = ','.join(sorted([str(element) for element in self.source]))
        self.assertEqual(expected_str, str(self.item))

    def test_repr(self):
        attr_values = {'source': ','.join(sorted([str(element) for element in self.source])),
                       'pv': self.item.primitive_value,
                       'dv': self.item.delegated_value}

        expected_repr = repr_str(self.item, attr_values)
        self.assertEqual(expected_repr, repr(self.item))

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

    def test_update_delegated_value(self):
        # test: verify that delegated value updates when delegated value helper's state is updated
        item_1 = sym_composite_item('(A,B)', primitive_value=0.0, delegated_value=6.0)

        item_2 = sym_item('X', primitive_value=10.0, delegated_value=0.0)

        # sanity check: delegated values should have been initialized
        self.assertEqual(6.0, item_1.delegated_value)
        self.assertEqual(10.0, item_2.primitive_value)

        # setting learning rate to 1.0 to simplify testing
        params = get_global_params()
        params.set('learning_rate', 1.0)

        # no discount or decay in this instance, so delegated value updates will be equal to the primitive values of
        # new result state items
        delegated_value_helper = EligibilityTraceDelegatedValueHelper(
            discount_factor=1.0,
            eligibility_trace=ReplacingTrace(
                decay_strategy=NoDecayStrategy())
        )
        delegated_value_helper.update(selection_state=sym_state('A,B'), result_state=sym_state('X'))
        set_delegated_value_helper(delegated_value_helper)

        # test: if delegated value helper were set properly, the item's delegated value should be equal to item 2's
        #       primitive value
        self.assertEqual(item_2.primitive_value, item_1.delegated_value)

    # def test_serialize(self):
    #     with TemporaryDirectory() as tmp_dir:
    #         path = Path(os.path.join(tmp_dir, 'test-file-composite_item-serialize.sav'))
    #
    #         # sanity check: file SHOULD NOT exist
    #         self.assertFalse(path.exists())
    #
    #         serialize(self.item, path)
    #
    #         # test: file SHOULD exist after call to save
    #         self.assertTrue(file_was_written(path))
    #
    #         recovered = deserialize(path)
    #
    #         self.assertEqual(self.item, recovered)
