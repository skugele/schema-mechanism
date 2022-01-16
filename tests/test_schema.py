from random import sample
from time import time
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import Context
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import Result
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import make_assertion
from schema_mechanism.func_api import make_assertions
from schema_mechanism.modules import lost_state
from schema_mechanism.modules import new_state
from schema_mechanism.modules import update_schema
from test_share.test_classes import MockObserver


class TestSchema(TestCase):
    def setUp(self) -> None:
        self._item_pool = ItemPool()

        # populate pool
        for i in range(10):
            _ = self._item_pool.get(i, SymbolicItem)

        self.context = Context(make_assertions((0, 1)))
        self.action = Action()
        self.result = Result(make_assertions((2, 3, 4)))

        self.schema = Schema(
            context=self.context,
            action=self.action,
            result=self.result
        )

        self.obs = MockObserver()
        self.schema.register(self.obs)

    def test_init(self):
        # Action CANNOT be None
        try:
            Schema(action=None)
            self.fail('Action=None should generate a ValueError')
        except ValueError as e:
            self.assertEqual('Action cannot be None', str(e))

        # Context and Result CAN be None
        try:
            s = Schema(action=Action('My Action'))
            self.assertIsNone(s.context)
            self.assertIsNone(s.result)
        except Exception as e:
            self.fail(f'Unexpected exception raised: {e}')

        # Verify immutability
        s = Schema(context=Context(item_asserts=(make_assertion('1'),)),
                   action=Action(),
                   result=Result(item_asserts=(make_assertion('2'),)))

        try:
            s.context = Context()
            self.fail('Schema\'s context is not immutable as expected')
        except Exception as e:
            pass

        try:
            s.result = Result()
            self.fail('Schema\'s result is not immutable as expected')
        except Exception as e:
            pass

        try:
            s.action = Action()
            self.fail('Schema\'s action is not immutable as expected')
        except Exception as e:
            pass

    def test_is_context_satisfied(self):
        c = Context((
            make_assertion('1'),
            make_assertion('2', negated=True),
            make_assertion('3')
        ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be satisfied
        ##########################
        self.assertTrue(schema.context.is_satisfied(state=['1', '3']))
        self.assertTrue(schema.context.is_satisfied(state=['1', '3', '4']))

        # expected to NOT be satisfied
        ##############################
        # case 1: present negated item
        self.assertFalse(schema.context.is_satisfied(state=['1', '2', '3']))

        # case 2: missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=['1']))
        self.assertFalse(schema.context.is_satisfied(state=['3']))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.context.is_satisfied(state=['1', '2']))
        self.assertFalse(schema.context.is_satisfied(state=['2', '3']))

    def test_is_applicable(self):
        c = Context(
            item_asserts=(
                make_assertion('1'),
                make_assertion('2', negated=True),
                make_assertion('3'),
            ))

        schema = Schema(context=c, action=Action(), result=None)

        # expected to be applicable
        ##########################
        self.assertTrue(schema.is_applicable(state=['1', '3']))
        self.assertTrue(schema.is_applicable(state=['1', '3', '4']))

        # expected to NOT be applicable
        ###############################

        # case 1: present negated item
        self.assertFalse(schema.is_applicable(state=['1', '2', '3']))

        # case 2: missing non-negated item
        self.assertFalse(schema.is_applicable(state=['1']))
        self.assertFalse(schema.is_applicable(state=['3']))

        # case 3 : both present negated item and missing non-negated item
        self.assertFalse(schema.is_applicable(state=['1', '2']))
        self.assertFalse(schema.is_applicable(state=['2', '3']))

        # Tests overriding conditions
        #############################
        schema.overriding_conditions = StateAssertion((make_assertion('5'),))

        # expected to be applicable
        self.assertTrue(schema.is_applicable(state=['1', '3', '4']))

        # expected to NOT be applicable (due to overriding condition)
        self.assertFalse(schema.is_applicable(state=['1', '3', '4', '5']))

    def test_update_1(self):
        # activated + success
        s_prev = [0, 1, 2, 3]
        s_curr = [2, 3, 4, 5]

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=True, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(1, self.schema.stats.n_success)
        self.assertEqual(0, self.schema.stats.n_fail)
        self.assertEqual(1, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(1, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(1, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(1, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(1, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)

    def test_update_2(self):
        # activated and not success
        s_prev = [0, 1, 2, 3]
        s_curr = [2, 5]

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=True, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(0, self.schema.stats.n_success)
        self.assertEqual(1, self.schema.stats.n_fail)
        self.assertEqual(1, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(1, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(1, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(1, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(1, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)

    def test_update_3(self):
        # not activated
        s_prev = [0, 3]
        s_curr = [2, 4, 5]

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=False, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(0, self.schema.stats.n_success)
        self.assertEqual(0, self.schema.stats.n_fail)
        self.assertEqual(0, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(1, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(1, er_stats.n_off_and_not_activated)

    def test_update_4(self):

        s_prev = None
        s_curr = [2, 4, 5]

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        self.schema.update(activated=False, v_prev=v_prev, v_curr=v_curr, new=new, lost=lost)

        # check schema level statistics
        self.assertEqual(1, self.schema.stats.n)
        self.assertEqual(0, self.schema.stats.n_success)
        self.assertEqual(0, self.schema.stats.n_fail)
        self.assertEqual(0, self.schema.stats.n_activated)

        # check item level statistics
        for i in self._item_pool:
            ec_stats = self.schema.extended_context.stats[i]

            if v_prev.is_on(i):
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)
            else:
                self.assertEqual(0, ec_stats.n_success_and_on)
                self.assertEqual(0, ec_stats.n_success_and_off)
                self.assertEqual(0, ec_stats.n_fail_and_on)
                self.assertEqual(0, ec_stats.n_fail_and_off)

            er_stats = self.schema.extended_result.stats.get(i)

            if i in new:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(1, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(0, er_stats.n_off_and_not_activated)
            elif i in lost:
                self.assertEqual(0, er_stats.n_on_and_activated)
                self.assertEqual(0, er_stats.n_on_and_not_activated)
                self.assertEqual(0, er_stats.n_off_and_activated)
                self.assertEqual(1, er_stats.n_off_and_not_activated)

    def test_reliability_0(self):
        # reliability before update should be NAN
        self.assertIs(np.NAN, self.schema.reliability)

    def test_reliability_1(self):
        # success update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # failure update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 5], count=2)
        self.assertEqual(0.25, self.schema.reliability)

    def test_reliability_2(self):
        # failure update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertEqual(0.0, self.schema.reliability)

        # success update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertEqual(0.5, self.schema.reliability)

        # success update
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=2)
        self.assertEqual(0.75, self.schema.reliability)

    def test_reliability_3(self):
        # reliability stats SHOULD NOT affected when schema not activated

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertIs(np.NAN, self.schema.reliability)

        # success update WITH activation
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=1)
        self.assertEqual(1.0, self.schema.reliability)

        # failure update WITHOUT activation
        update_schema(self.schema, activated=False, s_prev=[0, 1], s_curr=[2, 3, 5], count=1)
        self.assertEqual(1.0, self.schema.reliability)

    def test_notify_all(self):
        self.schema.notify_all = MagicMock()

        # activated + success
        update_schema(self.schema, activated=True, s_prev=[0, 1], s_curr=[2, 3, 4, 5], count=10)
        self.schema.notify_all.assert_not_called()

        # activated and not success
        update_schema(self.schema, activated=True, s_prev=[0, 1, 7], s_curr=[3, 4, 7], count=10)
        self.schema.notify_all.assert_called()

        # not activated
        update_schema(self.schema, activated=False, s_prev=[0, 3], s_curr=[4, 7, 8], count=10)
        self.schema.notify_all.assert_called()

    def test_performance(self):
        n_items = 10_000
        n_state_elements = 25

        # populate item pool
        self._item_pool.clear()
        for i in range(n_items):
            self._item_pool.get(i)

        s_prev = sample(range(n_items), k=n_state_elements)
        s_curr = sample(range(n_items), k=n_state_elements)

        v_prev = ItemPoolStateView(s_prev)
        v_curr = ItemPoolStateView(s_curr)

        new = new_state(s_prev, s_curr)
        lost = lost_state(s_prev, s_curr)

        # TODO: This is WAY too slow...
        start = time()
        self.schema.update(activated=True,
                           v_prev=v_prev,
                           v_curr=v_curr,
                           new=new,
                           lost=lost)
        end = time()
        elapsed_time = end - start

        print(f'Time updating a schema with {n_items:,} items : {elapsed_time}s ')
