import itertools
from unittest import TestCase

import numpy as np

from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Item
from schema_mechanism.core import ItemPool
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import advantage
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.core import calc_value
from schema_mechanism.core import get_global_stats
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from test_share.test_classes import MockCompositeItem
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


class TestCalcPrimitiveValue(TestCase):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        super().setUp()
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, primitive_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, primitive_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, primitive_value=100.0)

        self.ci1: CompositeItem = sym_item('(A,B)')
        self.ci2: CompositeItem = sym_item('(B,C)')
        self.ci3: CompositeItem = sym_item('(~A,C)')
        self.ci4: CompositeItem = sym_item('(A,~C)')

        self.items: list[Item] = [self.i1, self.i2, self.i3]
        self.composite_items: list[CompositeItem] = [self.ci1, self.ci2, self.ci3, self.ci4]
        self.state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.items]))

    def test_pv_other(self):
        self.assertRaises(TypeError, lambda: calc_primitive_value(set()))
        self.assertRaises(TypeError, lambda: calc_primitive_value(dict()))

    def test_pv_item(self):
        # test: calc_primitive_value(item) should equal the value of the item's primitive value property
        for item in [*self.items, *self.composite_items]:
            self.assertEqual(item.primitive_value, calc_primitive_value(item))

    def test_pv_state_element(self):
        # test: calc_primitive_value(state element) should return the primitive value of the corresponding item in pool
        for se in self.state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.primitive_value, calc_primitive_value(se))

        # test: calc_primitive_value(state element) should return a ValueError if item does not exist in pool
        self.assertEqual(0.0, calc_primitive_value('UNK'))

    def test_pv_state(self):
        s_empty = sym_state('')
        s_none = None

        # test: empty state should have zero primitive value
        self.assertEqual(0.0, calc_primitive_value(s_empty))
        self.assertEqual(0.0, calc_primitive_value(s_none))

        # test: state with unknown state elements should have zero primitive value
        self.assertEqual(0.0, calc_primitive_value(sym_state('UNK,NOPE,NADA')))

        # test: single element states should have that element's corresponding non-composite item's primitive value
        for state_element in 'ABC':
            state = sym_state(state_element)
            item = sym_item(state_element)

            self.assertEqual(item.primitive_value, calc_primitive_value(state))

        # test: calc_primitive_value(state) should return the SUM of state elements' primitive values
        for elements in itertools.combinations('ABC', 2):
            state_str = ','.join(elements)
            state = sym_state(state_str)

            items = [ReadOnlyItemPool().get(e) for e in elements]
            item_sum = sum(item.primitive_value for item in items if item)

            self.assertEqual(item_sum, calc_primitive_value(state),
                             msg=f'Primitive value of items differs from state\'s {state}\'s primitive value!')

    def test_pv_state_assertion(self):
        # test: state assertions about previously unknown state elements should have zero primitive value
        self.assertEqual(0.0, calc_primitive_value(sym_state_assert('UNK,NOPE,NADA')))

        # test: state assertions' primitive values should equal the SUM of their corresponding items' primitive value
        for r in range(1, 4):
            for state_str in [','.join(elements) for elements in itertools.combinations('ABC', r)]:
                state_assert = sym_state_assert(state_str)

                expected_value = sum(item.primitive_value for item in state_assert)
                self.assertEqual(expected_value, calc_primitive_value(state_assert))


class TestCalcDelegatedValue(TestCase):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, delegated_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, delegated_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, delegated_value=100.0)

        self.ci1: CompositeItem = sym_item('(A,B)', item_type=MockCompositeItem, delegated_value=-100.0)
        self.ci2: CompositeItem = sym_item('(B,C)', item_type=MockCompositeItem, delegated_value=0.0)
        self.ci3: CompositeItem = sym_item('(C,D)', item_type=MockCompositeItem, delegated_value=100.0)

        self.items: list[Item] = [self.i1, self.i2, self.i3]
        self.composite_items: list[CompositeItem] = [self.ci1, self.ci2, self.ci3]
        self.state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.items]))

    def test_dv_other(self):
        self.assertRaises(TypeError, lambda: calc_delegated_value(set()))
        self.assertRaises(TypeError, lambda: calc_delegated_value(dict()))

    def test_dv_item(self):
        # test: delegated_value(item) should equal the value of the item's delegated value property
        for item in [*self.items, *self.composite_items]:
            self.assertEqual(item.delegated_value, calc_delegated_value(item))

    def test_dv_state_element(self):
        # test: delegated_value(state element) should return the delegated value of the corresponding item in pool
        for se in self.state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.delegated_value, calc_delegated_value(se))

        # test: delegated_value(state element) should return 0.0 if item does not exist in pool
        self.assertEqual(0.0, calc_delegated_value('UNK'))

    def test_dv_state(self):
        s_empty = sym_state('')
        s_none = None

        # test: empty state should have zero delegated value
        self.assertEqual(0.0, calc_delegated_value(s_empty))
        self.assertEqual(0.0, calc_delegated_value(s_none))

        # test: state with unknown state elements should have zero delegated value
        self.assertEqual(0.0, calc_delegated_value(sym_state('UNK,NOPE,NADA')))

        # test: single element states should have that element's corresponding non-composite item's delegated value
        for state_element in 'ABC':
            state = sym_state(state_element)
            item = sym_item(state_element)

            self.assertEqual(item.delegated_value, calc_delegated_value(state))

        # test: delegated_value(state) should return the SUM of state elements' delegated values
        for elements in itertools.combinations('ABC', 2):
            state_str = ','.join(elements)
            state = sym_state(state_str)

            items = [ReadOnlyItemPool().get(e) for e in elements]
            item_sum = sum(item.delegated_value for item in items if item)

            self.assertEqual(item_sum, calc_delegated_value(state),
                             msg=f'delegated value of items differs from state\'s {state}\'s delegated value!')

    def test_dv_state_assertion(self):
        # test: state assertions about previously unknown state elements should have zero delegated value
        self.assertEqual(0.0, calc_delegated_value(sym_state_assert('UNK,NOPE,NADA')))

        # test: state assertions' delegated values should equal the SUM of their corresponding items' delegated value
        for r in range(1, 4):
            for state_str in [','.join(elements) for elements in itertools.combinations('ABC', r)]:
                state_assert = sym_state_assert(state_str)

                expected_value = sum(item.delegated_value for item in state_assert)
                self.assertEqual(expected_value, calc_delegated_value(state_assert))


class TestCalcValue(TestCase):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, delegated_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, delegated_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, delegated_value=100.0)

        self.ci1: CompositeItem = sym_item('(A,B)', item_type=MockCompositeItem, delegated_value=-100.0)
        self.ci2: CompositeItem = sym_item('(B,C)', item_type=MockCompositeItem, delegated_value=0.0)
        self.ci3: CompositeItem = sym_item('(C,D)', item_type=MockCompositeItem, delegated_value=100.0)

        self.items: list[Item] = [self.i1, self.i2, self.i3]
        self.composite_items: list[CompositeItem] = [self.ci1, self.ci2, self.ci3]
        self.state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.items]))

    def test_calc_value_other(self):
        self.assertRaises(TypeError, lambda: calc_value(set()))
        self.assertRaises(TypeError, lambda: calc_value(dict()))

    def test_calc_value_item(self):
        # test: calc_value(item) should equal the sum of the item's primitive and delegated values
        for item in [*self.items, *self.composite_items]:
            expected_value = item.primitive_value + item.delegated_value
            self.assertEqual(expected_value, calc_value(item))

    def test_calc_value_state_element(self):
        # test: calc_value(state element) should equal the sum of the state element's corresponding item's primitive
        #     : and delegated values
        for se in self.state_elements:
            item = ReadOnlyItemPool().get(se)
            expected_value = item.primitive_value + item.delegated_value
            self.assertEqual(expected_value, calc_value(se))

        # test: calc_value(state element) should return 0.0 if the item does not exist in pool
        self.assertEqual(0.0, calc_value('UNK'))

    def test_calc_value_state(self):
        s_empty = sym_state('')
        s_none = None

        # test: empty state should have zero value
        self.assertEqual(0.0, calc_value(s_empty))
        self.assertEqual(0.0, calc_value(s_none))

        # test: state with unknown state elements should have zero value
        self.assertEqual(0.0, calc_value(sym_state('UNK,NOPE,NADA')))

        # test: single element states should equal the sum of that element's corresponding non-composite item's
        #     : primitive and delegated values
        for state_element in 'ABC':
            state = sym_state(state_element)
            item = sym_item(state_element)

            expected_value = item.primitive_value + item.delegated_value
            self.assertEqual(expected_value, calc_value(state))

        # test: calc_value(state) should return the SUM of the state elements' primitive and delegated values
        for elements in itertools.combinations('ABC', 2):
            state_str = ','.join(elements)
            state = sym_state(state_str)

            items = [ReadOnlyItemPool().get(e) for e in elements]
            item_sum = sum(calc_value(item) for item in items if item)

            self.assertEqual(item_sum, calc_value(state),
                             msg=f'calc_value over items differs from state\'s {state}\'s calc_value!')

    def test_calc_value_state_assertion(self):
        # test: state assertions about previously unknown state elements should have zero value
        self.assertEqual(0.0, calc_value(sym_state_assert('UNK,NOPE,NADA')))

        # test: state assertions' calc_value should equal the SUM of their corresponding items' calc_values
        for r in range(1, 4):
            for state_str in [','.join(elements) for elements in itertools.combinations('ABC', r)]:
                state_assert = sym_state_assert(state_str)

                item_sum = sum(calc_value(item) for item in state_assert)
                self.assertEqual(item_sum, calc_value(state_assert))


class TestAdvantage(TestCase):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, delegated_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, delegated_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, delegated_value=100.0)

        self.ci1: CompositeItem = sym_item('(A,B)', item_type=MockCompositeItem, delegated_value=-100.0)
        self.ci2: CompositeItem = sym_item('(B,C)', item_type=MockCompositeItem, delegated_value=0.0)
        self.ci3: CompositeItem = sym_item('(C,D)', item_type=MockCompositeItem, delegated_value=100.0)

        self.items: list[Item] = [self.i1, self.i2, self.i3]
        self.composite_items: list[CompositeItem] = [self.ci1, self.ci2, self.ci3]
        self.state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.items]))

    def test_advantage_other(self):
        self.assertRaises(TypeError, lambda: advantage(set()))
        self.assertRaises(TypeError, lambda: advantage(dict()))

    def test_advantage_item(self):
        # test: advantage(item) should equal the difference of the sum of the item's primitive and delegated values
        #     : and the baseline
        for baseline_value in np.linspace(-10.0, 10.0):
            global_stats = get_global_stats()
            global_stats.baseline_value = baseline_value

            for item in [*self.items, *self.composite_items]:
                expected_value = calc_value(item) - baseline_value
                self.assertEqual(expected_value, advantage(item))

    def test_advantage_state_element(self):
        # test: advantage(state element) should equal the difference of the sum of the state element's corresponding
        #     : item's primitive and delegated values and the baseline
        for baseline_value in np.linspace(-10.0, 10.0):
            global_stats = get_global_stats()
            global_stats.baseline_value = baseline_value

            for se in self.state_elements:
                item = ReadOnlyItemPool().get(se)
                expected_value = calc_value(item) - baseline_value
                self.assertEqual(expected_value, advantage(se))

            # test: advantage(state element) should return -baseline if the item does not exist in pool
            self.assertEqual(-baseline_value, advantage('UNK'))

    def test_advantage_state(self):
        s_empty = sym_state('')
        s_none = None

        for baseline_value in np.linspace(-10.0, 10.0):
            global_stats = get_global_stats()
            global_stats.baseline_value = baseline_value

            # test: empty state should have -baseline advantage
            self.assertEqual(-baseline_value, advantage(s_empty))
            self.assertEqual(-baseline_value, advantage(s_none))

            # test: state with unknown state elements should have -baseline advantage
            self.assertEqual(-baseline_value, advantage(sym_state('UNK,NOPE,NADA')))

            # test: single element states should have that element's corresponding non-composite item's advantage
            for state_element in 'ABC':
                state = sym_state(state_element)
                item = sym_item(state_element)

                self.assertEqual(advantage(item), advantage(state))

            # test: advantage(state) should return the difference between the SUM of the state elements' primitive
            #     : and delegated values, and the baseline
            for elements in itertools.combinations('ABC', 2):
                state_str = ','.join(elements)
                state = sym_state(state_str)

                items = [ReadOnlyItemPool().get(e) for e in elements]
                item_sum = sum(calc_value(item) for item in items if item)

                self.assertEqual(item_sum - baseline_value, advantage(state),
                                 msg=f'advantage over items differs from state\'s {state}\'s advantage!')

    def test_advantage_state_assertion(self):
        for baseline_value in np.linspace(-10.0, 10.0):
            global_stats = get_global_stats()
            global_stats.baseline_value = baseline_value

            # test: state assertions about previously unknown state elements should have zero delegated value
            self.assertEqual(-baseline_value, advantage(sym_state_assert('UNK,NOPE,NADA')))

            # test: state assertions' delegated values should equal the SUM of their corresponding items' delegated
            #     : value
            for r in range(1, 4):
                for state_str in [','.join(elements) for elements in itertools.combinations('ABC', r)]:
                    state_assert = sym_state_assert(state_str)

                    item_sum = sum(calc_value(item) for item in state_assert)
                    self.assertEqual(item_sum - baseline_value, advantage(state_assert))
